"""Step 5.1: tiny PubMed client (NCBI E-utilities) with a disk cache.

E-utilities = NCBI's public web API for PubMed. Two calls:
  esearch : query text -> list of PMIDs (PubMed article IDs), best matches first
  efetch  : PMIDs -> title, abstract, journal, year, publication types (as XML)

Every answer is saved in data/processed/pubmed/cache/, keyed by the request. Asking the same
thing again reads the file instead of the internet: reruns are fast, reproducible (PubMed
changes daily), and we stay within NCBI's rate limit (3 requests/s, 10/s with an API key).

Identity: NCBI asks for a tool name and an email. NCBI_EMAIL and NCBI_API_KEY are read from
the gitignored .env if present; they are sent to NCBI only and never printed or cached.

Try it from the repo root:
    conda run -n derm python -m src.retrieval.pubmed "melanoma AND blue-whitish veil AND dermoscopy"
"""
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

from src import config

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CACHE_DIR = config.PROCESSED_DIR / "pubmed" / "cache"
FILTERS = 'english[lang] AND hasabstract AND humans[mh] NOT "retracted publication"[pt]'

load_dotenv(config.ROOT / ".env")
_IDENTITY = {"tool": "swayam-xai-derm"}
if os.getenv("NCBI_EMAIL"):
    _IDENTITY["email"] = os.getenv("NCBI_EMAIL")
if os.getenv("NCBI_API_KEY"):
    _IDENTITY["api_key"] = os.getenv("NCBI_API_KEY")
_PAUSE = 0.11 if "api_key" in _IDENTITY else 0.34
_last_call = 0.0


def _get(endpoint, params):
    """GET with cache + rate limit + 3 retries. Cache key ignores the identity fields."""
    key = hashlib.sha1(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["body"]
    global _last_call
    for attempt in range(3):
        time.sleep(max(0.0, _PAUSE - (time.time() - _last_call)))
        _last_call = time.time()
        r = requests.get(BASE + endpoint, params={**params, **_IDENTITY}, timeout=30)
        if r.status_code == 200:
            break
        time.sleep(2 ** attempt)                     # 429 / 5xx: wait and retry
    r.raise_for_status()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"endpoint": endpoint, "params": params, "body": r.text}), encoding="utf-8")
    return r.text


def search(query, retmax=20, filters=True):
    """PMIDs for a query, PubMed 'best match' order."""
    term = f"({query}) AND {FILTERS}" if filters else query
    body = _get("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": retmax,
                                 "sort": "relevance", "retmode": "json"})
    return json.loads(body)["esearchresult"]["idlist"]


def fetch(pmids):
    """{pmid: {title, abstract, journal, year, types}} for a list of PMIDs."""
    out = {}
    for start in range(0, len(pmids), 100):
        ids = sorted(pmids[start:start + 100])        # sorted -> same request -> cache hit
        root = ET.fromstring(_get("efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}))
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//MedlineCitation/PMID")
            parts = []
            for a in art.iterfind(".//Abstract/AbstractText"):
                text = "".join(a.itertext()).strip()
                parts.append(f"{a.get('Label')}: {text}" if a.get("Label") else text)
            out[pmid] = {
                "pmid": pmid,
                "title": "".join(art.find(".//ArticleTitle").itertext()).strip(),
                "abstract": " ".join(parts),
                "journal": art.findtext(".//Journal/Title"),
                "year": art.findtext(".//JournalIssue/PubDate/Year") or art.findtext(".//JournalIssue/PubDate/MedlineDate"),
                "types": [t.text for t in art.iterfind(".//PublicationType")],
            }
    return out


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "melanoma AND blue-whitish veil AND dermoscopy"
    print("identity sent:", sorted(k for k in _IDENTITY), "(values hidden)")
    ids = search(q, retmax=5)
    print(f"query: {q}\n{len(ids)} PMIDs: {ids}")
    for p in fetch(ids).values():
        print(f"- PMID {p['pmid']} ({p['year']}) {p['title'][:90]}  [{len(p['abstract'])} chars]")
