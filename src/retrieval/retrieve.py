"""Step 5.4: retrieve evidence for every valid + test case, for the 3 RQ1 arms, and check it.

For each query: PubMed top 20 (filters: English, abstract, humans, not retracted). If fewer
than 3 hits, retry without the dermoscopy term (marked as fallback). Candidates are re-ranked
with MedCPT. Each arm keeps 6 abstracts per case:
  shap / random : 2 best per factor query (3 factors), no duplicates
  diagnosis     : 6 best from its single query (same budget)

Checks (no LLM yet), on TEST:
  - how many queries had no / few hits
  - diagnosis mentioned: share of abstracts naming the predicted diagnosis
  - SHAP-factor coverage: share of the case's SHAP factors that appear in at least one of
    the arm's abstracts (the evidence an explanation-based report would need)
  - overlap between arms (Jaccard = shared / all PMIDs), diversity, review share, year

Output: data/processed/pubmed/evidence.jsonl (one line per case x arm), abstracts.json,
reports/phase5_retrieval.md, MLflow experiment "phase5-retrieval".

Run from the repo root:
    conda run -n derm python -m src.retrieval.retrieve
"""
import json
import re

import mlflow
import numpy as np
import pandas as pd

from src import config
from src.models.unimodal import md_table
from src.retrieval.pubmed import fetch, search
from src.retrieval.queries import DERMOSCOPY, build_queries
from src.retrieval.rerank import MODEL, Reranker

OUT_DIR = config.PROCESSED_DIR / "pubmed"
RETMAX, PER_FACTOR, PER_ARM, MIN_HITS = 20, 2, 6, 3
ARMS = ["shap", "random", "diagnosis"]
DIAG_WORDS = {"nevus": ["nevus", "nevi", "naevus", "naevi"], "melanoma": ["melanoma"],
              "miscellaneous": ["lentigo", "dermatofibroma", "vascular lesion", "angioma", "melanosis"],
              "seborrheic keratosis": ["seborrheic keratos", "seborrhoeic keratos"],
              "basal cell carcinoma": ["basal cell carcinoma", "bcc"]}


def norm(text):
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "


def term_phrases(term):
    """'("arborizing vessels" OR comma)' -> ['arborizing vessels', 'comma']"""
    parts = re.findall(r'"([^"]+)"|([A-Za-z][A-Za-z-]+)', term)
    return [norm(a or b).strip() for a, b in parts if (a or b) not in ("OR", "AND")]


def mentions(text, phrases):
    t = norm(text)
    return any(f" {p}" in t for p in phrases)       # word start match: "keratos" matches "keratoses"


def run_query(q):
    """PMIDs for one query, with the no-dermoscopy fallback."""
    ids = search(q, RETMAX)
    if len(ids) >= MIN_HITS:
        return ids, False
    return search(q.replace(f" AND {DERMOSCOPY}", ""), RETMAX), True


def main():
    expl = [json.loads(l) for l in open(config.PROCESSED_DIR / "explanations" / "explanations.jsonl", encoding="utf-8")]
    plans = {e["row"]: build_queries(e) for e in expl}

    # 1. search every unique query once
    unique = sorted({q["query"] for p in plans.values() for qs in p.values() for q in qs})
    hits = {}
    for n, q in enumerate(unique, 1):
        hits[q] = run_query(q)
        if n % 20 == 0:
            print(f"searched {n}/{len(unique)}")
    articles = fetch(sorted({i for ids, _ in hits.values() for i in ids}))
    print(f"{len(unique)} queries, {len(articles)} unique abstracts")

    # 2. re-rank each query's candidates with MedCPT
    rr = Reranker()
    pairs = sorted({(q["rerank_text"], q["query"]) for p in plans.values() for qs in p.values() for q in qs})
    ranked = {}
    for text, query in pairs:
        cands = [articles[i] for i in hits[query][0] if i in articles]
        scores = rr.score([(text, a) for a in cands])
        ranked[(text, query)] = sorted(zip(scores, cands), key=lambda t: -t[0])

    # 3. pick evidence per case and arm
    records = []
    for e in expl:
        for arm in ARMS:
            chosen, qinfo = [], []
            per_q = PER_ARM if arm == "diagnosis" else PER_FACTOR
            for q in plans[e["row"]][arm]:
                ids, fallback = hits[q["query"]]
                qinfo.append({**q, "n_hits": len(ids), "fallback": fallback})
                taken = 0
                for s, a in ranked[(q["rerank_text"], q["query"])]:
                    if taken == per_q:
                        break
                    if a["pmid"] in {c["pmid"] for c in chosen}:
                        continue
                    chosen.append({"pmid": a["pmid"], "score": round(s, 3), "factor": q["factor"], "value": q["value"]})
                    taken += 1
            records.append({"row": e["row"], "case_num": e["case_num"], "split": e["split"], "true": e["true"],
                            "pred": e["pred"], "arm": arm, "queries": qinfo, "evidence": chosen})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "evidence.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (OUT_DIR / "abstracts.json").write_text(json.dumps(articles), encoding="utf-8")

    # 4. checks on TEST
    shap_terms = {e["row"]: [term_phrases(q["term"]) for q in plans[e["row"]]["shap"]] for e in expl}
    rows, pm = [], {}
    for arm in ARMS:
        recs = [r for r in records if r["split"] == "test" and r["arm"] == arm]
        n_ev = [len(r["evidence"]) for r in recs]
        qs = [q for r in recs for q in r["queries"]]
        diag_m, cover, reviews, years = [], [], [], []
        for r in recs:
            texts = [articles[c["pmid"]]["title"] + " " + articles[c["pmid"]]["abstract"] for c in r["evidence"]]
            diag_m += [mentions(t, DIAG_WORDS[r["pred"]]) for t in texts]
            if shap_terms[r["row"]]:
                cover.append(np.mean([any(mentions(t, ph) for t in texts) for ph in shap_terms[r["row"]]]))
            for c in r["evidence"]:
                a = articles[c["pmid"]]
                reviews.append(any("Review" in t for t in a["types"]))
                y = re.match(r"\d{4}", a["year"] or "")
                if y:
                    years.append(int(y.group()))
        pm[arm] = [{c["pmid"] for c in r["evidence"]} for r in recs]
        rows.append({"arm": arm, "abstracts/case": f"{np.mean(n_ev):.2f}",
                     "queries <3 hits (fallback)": f"{np.mean([q['fallback'] for q in qs]):.2f}",
                     "queries 0 hits": f"{np.mean([q['n_hits'] == 0 for q in qs]):.2f}",
                     "diagnosis mentioned": f"{np.mean(diag_m):.2f}",
                     "SHAP-factor coverage": f"{np.mean(cover):.2f}",
                     "unique PMIDs (all cases)": len(set().union(*pm[arm])),
                     "reviews": f"{np.mean(reviews):.2f}", "median year": int(np.median(years))})
    table = pd.DataFrame(rows)
    jac = lambda a, b: np.mean([len(x & y) / max(1, len(x | y)) for x, y in zip(pm[a], pm[b])])
    overlap = pd.DataFrame([{"arms": f"{a} vs {b}", "mean Jaccard": f"{jac(a, b):.2f}"}
                            for a, b in [("shap", "diagnosis"), ("shap", "random"), ("random", "diagnosis")]])

    ex = next(r for r in records if r["split"] == "test" and r["arm"] == "shap" and r["pred"] == "melanoma")
    example = "\n".join(f"- [{c['value']}] PMID {c['pmid']} ({articles[c['pmid']]['year']}): "
                        f"{articles[c['pmid']]['title']} (MedCPT {c['score']:.2f})" for c in ex["evidence"])

    out = config.REPORTS_DIR / "phase5_retrieval.md"
    out.write_text(
        "# Phase 5: PubMed evidence retrieval (3 arms for RQ1)\n\n"
        f"PubMed E-utilities, top {RETMAX} per query (English, abstract, humans, not retracted), "
        f"re-ranked with `{MODEL}`. {PER_ARM} abstracts per case and arm. "
        f"{len(unique)} unique queries, {len(articles)} unique abstracts. Retrieved on {pd.Timestamp.now():%Y-%m-%d} "
        "(PubMed changes daily; raw answers are cached in data/processed/pubmed/cache).\n\n"
        "## Checks on test cases (395)\n\n" + md_table(table) +
        "\n\nSHAP-factor coverage = share of a case's SHAP factors (top 3) named in at least one of the arm's "
        "abstracts; it is high for the SHAP arm by construction, so it is a coverage check, not the RQ1 result. "
        "RQ1 is answered in Phase 6 (report faithfulness and relevance).\n\n"
        "## Overlap between arms\n\n" + md_table(overlap) +
        f"\n\n## Example: test case {ex['case_num']} (true {ex['true']}, predicted {ex['pred']}), SHAP arm\n\n" + example + "\n",
        encoding="utf-8")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase5-retrieval")
    with mlflow.start_run(run_name="pubmed_medcpt"):
        mlflow.log_params({"retmax": RETMAX, "per_factor": PER_FACTOR, "per_arm": PER_ARM, "reranker": MODEL,
                           "n_queries": len(unique), "n_abstracts": len(articles)})
        for r in rows:
            mlflow.log_metrics({f"{r['arm']}_diag_mentioned": float(r["diagnosis mentioned"]),
                                f"{r['arm']}_shap_coverage": float(r["SHAP-factor coverage"]),
                                f"{r['arm']}_fallback": float(r["queries <3 hits (fallback)"])})
        mlflow.log_artifact(str(out))
    print(table.to_string(index=False))
    print(overlap.to_string(index=False))
    print(example)
    print("wrote", out.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
