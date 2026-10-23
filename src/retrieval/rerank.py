"""Step 5.3: re-rank PubMed candidates with MedCPT (NCBI's PubMed-trained cross-encoder).

PubMed's own "best match" order is decent, but it only sees the query words. A cross-encoder
reads the query AND the whole title + abstract together and outputs one relevance score
(higher = more relevant). MedCPT was trained by NCBI on real PubMed search logs.
Model card: https://huggingface.co/ncbi/MedCPT-Cross-Encoder

Scores are cached in data/processed/pubmed/rerank_scores.json, keyed by (query text, PMID).
"""
import hashlib
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config

MODEL = "ncbi/MedCPT-Cross-Encoder"
SCORES_PATH = config.PROCESSED_DIR / "pubmed" / "rerank_scores.json"


def _key(text, pmid):
    return hashlib.sha1(f"{text}|{pmid}".encode()).hexdigest()


class Reranker:
    def __init__(self, device="cuda"):
        self.device = device
        self.cache = json.loads(SCORES_PATH.read_text()) if SCORES_PATH.exists() else {}
        self.model = None                                  # loaded only if something is not cached

    def _load(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device).eval()

    @torch.no_grad()
    def score(self, pairs, batch=16):
        """pairs: list of (query text, article dict). Returns one score per pair."""
        todo = [(q, a) for q, a in pairs if _key(q, a["pmid"]) not in self.cache]
        if todo and self.model is None:
            self._load()
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            enc = self.tok([q for q, _ in part], [f"{a['title']} {a['abstract']}" for _, a in part],
                           truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", enabled=self.device == "cuda"):
                logits = self.model(**enc).logits.squeeze(1).float().cpu().tolist()
            for (q, a), s in zip(part, logits):
                self.cache[_key(q, a["pmid"])] = s
        if todo:
            SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
            SCORES_PATH.write_text(json.dumps(self.cache))
        return [self.cache[_key(q, a["pmid"])] for q, a in pairs]
