"""Step 6.2: generate clinician-readable reports with gpt-oss-120b, from retrieved evidence ONLY.

For RQ1 every arm gets the SAME case information and SHAP explanation; only the 6 abstracts
differ (shap / random / diagnosis arm, Phase 5). So any difference between the reports comes
from the evidence retrieval.

The report is JSON so it can be checked by code:
  summary         what the model suggested and why (from the explanation; no literature claims)
  findings        one claim per item: the factor it is about, the factor's SHAP direction
                  (toward / against the suggestion), and the PMIDs supporting it
  evidence_gaps   factors the abstracts do not support, or conflicting evidence
  recommendation  next step for the clinician (review; no treatment orders)
  disclaimer      fixed sentence, checked word for word

Run from the repo root:
    conda run -n derm python -m src.llm.report --split valid --n 10 --tag pilot
    conda run -n derm python -m src.llm.report --split test --n 100 --tag main
"""
import argparse
import json

import numpy as np

from src import config
from src.llm.groq_client import chat

WRITER = "openai/gpt-oss-120b"
WRITER_SETTINGS = {"reasoning_effort": "low"}
ARMS = ["shap", "random", "diagnosis"]
ABSTRACT_CHARS = 2000          # long abstracts are cut to keep each prompt under the token limit
OUT_DIR = config.PROCESSED_DIR / "llm"
DISCLAIMER = ("This is AI-generated decision support, not a diagnosis. "
              "A qualified clinician must make all diagnostic and treatment decisions.")

SYSTEM = f"""You write short reports for clinicians about the output of an AI research prototype that analyses skin lesions (dermoscopic image, clinical image, patient metadata).
You explain the AI model's suggestion using ONLY the PubMed abstracts provided.

Rules:
1. Every medical claim must come from the provided abstracts. Do not use outside knowledge for medical claims.
2. Each finding cites 1-3 PMIDs from the provided list that directly support that claim. Never cite a PMID that is not in the list.
3. If the abstracts do not support a factor, or contradict each other, say so in "evidence_gaps" instead of making a claim.
4. Call the result "the model's suggestion", never a diagnosis. Do not give probabilities, percentages or confidence levels for the diagnosis.
5. Do not recommend specific treatments or drugs. You may recommend clinical review, dermatologist referral or considering further assessment.
6. If the suggestion is melanoma or basal cell carcinoma, the recommendation must advise timely specialist review.
7. Report the model's explanation exactly as given: a factor listed under TOWARD pushed toward the suggestion, a factor under AGAINST pushed against it, even if this seems medically surprising. Never move a factor to the other side. If the abstracts disagree with how the model used a factor, say so in "evidence_gaps".
8. Findings are only about factors in the explanation (use their id) or "general" statements about the suggested diagnosis.
9. Write plainly for a busy clinician. Keep each claim to one sentence.

Return only JSON with exactly these keys:
{{"summary": "2-3 sentences: the model's suggestion and the main factors in its explanation. No literature claims here.",
 "findings": [{{"factor": "factor id from the explanation (e.g. concept_streaks), or general", "direction": "toward, against, or none for general", "claim": "one sentence", "pmids": ["PMID"]}}],
 "evidence_gaps": "which factors lack support in the abstracts, and any conflicts",
 "recommendation": "one or two sentences",
 "disclaimer": "{DISCLAIMER}"}}"""


def factor_line(item, suggestion):
    """Explanation item -> one line without probabilities."""
    f, v = item["factor"], item["value"]
    if f in ("diag_derm", "diag_clinic"):
        who = "dermoscopic" if f == "diag_derm" else "clinical"
        lean = "toward" if item["shap"] > 0 else "away from"
        return f"- [{f}] the {who} image model's own output leaned {lean} {suggestion}"
    v = v.split(" (p=")[0]
    if f.startswith("concept_"):
        v += " (as detected by the image model)"
    return f"- [{f}] {v}"


def explanation_text(expl):
    """The case part of the prompt: suggestion + SHAP factors (same for all arms; the judge sees it too)."""
    sug = expl["pred"]
    sup = [factor_line(s, sug) for s in expl["supporting"]]
    opp = [factor_line(s, sug) for s in expl["opposing"]]
    return (f"Model suggestion: {sug}\n\n"
            "Model explanation (SHAP) - factors that pushed the model TOWARD this suggestion, strongest first:\n"
            + "\n".join(sup) + "\n\nFactors that pushed AGAINST it:\n" + ("\n".join(opp) or "- none"))


def build_messages(expl, evidence, abstracts):
    docs = []
    for c in evidence["evidence"]:
        a = abstracts[c["pmid"]]
        text = a["abstract"][:ABSTRACT_CHARS] + (" [...]" if len(a["abstract"]) > ABSTRACT_CHARS else "")
        docs.append(f"[PMID {a['pmid']}] {a['title']} ({a['year']}, {a['journal']})\n{text}")
    user = (explanation_text(expl) +
            "\n\nPubMed abstracts (the only allowed evidence):\n\n" + "\n\n".join(docs))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text):
    """JSON report or None if it is not valid / misses keys."""
    try:
        r = json.loads(text)
    except json.JSONDecodeError:
        return None
    keys = {"summary", "findings", "evidence_gaps", "recommendation", "disclaimer"}
    return r if isinstance(r, dict) and keys <= set(r) and isinstance(r["findings"], list) else None


def generate(expl, evidence, abstracts, sample=0, temperature=0.0):
    """One report. Retries once (new sample) if the JSON is broken."""
    msgs = build_messages(expl, evidence, abstracts)
    for s in [sample, sample + 1000]:
        out = chat(WRITER, msgs, temperature=temperature, max_tokens=3000, sample=s, **WRITER_SETTINGS)
        rep = parse(out["text"])
        if rep is not None:
            return rep, out["usage"]
    return None, out["usage"]


def select_cases(expl, split, n, seed=config.SEED):
    """Stratified by TRUE class: small classes (<= 20 cases) all in, the rest shared equally.
    Within a class, correct and wrong predictions are mixed by random draw."""
    rows = [e for e in expl if e["split"] == split]
    by_class = {}
    for e in rows:
        by_class.setdefault(e["true"], []).append(e)
    rng = np.random.default_rng(seed)
    if n >= len(rows):
        return rows
    small = {c: v for c, v in by_class.items() if len(v) <= 20 and n > 20}
    chosen = [e for v in small.values() for e in v]
    rest = [c for c in by_class if c not in small]
    left = n - len(chosen)
    for k, c in enumerate(sorted(rest, key=lambda c: len(by_class[c]))):
        take = min(len(by_class[c]), left // (len(rest) - k))
        chosen += [by_class[c][j] for j in rng.choice(len(by_class[c]), take, replace=False)]
        left -= take
    return sorted(chosen, key=lambda e: e["row"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    expl = [json.loads(l) for l in open(config.PROCESSED_DIR / "explanations" / "explanations.jsonl", encoding="utf-8")]
    evid = {(r["row"], r["arm"]): r for r in map(json.loads, open(config.PROCESSED_DIR / "pubmed" / "evidence.jsonl", encoding="utf-8"))}
    abstracts = json.loads((config.PROCESSED_DIR / "pubmed" / "abstracts.json").read_text(encoding="utf-8"))
    cases = select_cases(expl, args.split, args.n)
    print(f"{len(cases)} cases:", {c: sum(e['true'] == c for e in cases) for c in sorted({e['true'] for e in cases})})

    out_path = OUT_DIR / f"reports_{args.tag}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokens = []
    with open(out_path, "w", encoding="utf-8") as f:
        for n, e in enumerate(cases, 1):
            for arm in ARMS:
                rep, usage = generate(e, evid[(e["row"], arm)], abstracts)
                tokens.append(usage.get("total_tokens", 0))
                f.write(json.dumps({"row": e["row"], "case_num": e["case_num"], "split": e["split"], "true": e["true"],
                                    "pred": e["pred"], "arm": arm, "report": rep, "usage": usage}) + "\n")
                f.flush()
            print(f"case {n}/{len(cases)} done (tokens/report so far: mean {np.mean(tokens):.0f})")
    print("wrote", out_path.relative_to(config.ROOT), "| broken reports:",
          sum(json.loads(l)["report"] is None for l in open(out_path, encoding="utf-8")))


if __name__ == "__main__":
    main()
