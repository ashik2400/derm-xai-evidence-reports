"""Step 6.4: evaluate the reports and compare the 3 retrieval arms (RQ1).

Per report, checked by CODE (exact, free):
  json_ok, disclaimer_ok        report parsed; disclaimer is word for word
  pmid_valid                    share of cited PMIDs that were really retrieved for this arm
  any_fake_pmid                 1 if the report cites any PMID it was not given
  direction_ok                  share of factor findings whose toward/against matches SHAP
  prob_leak, treatment_words    probabilities/percentages, or treatment words, in the text
  referral_ok                   melanoma/BCC suggestion -> specialist/dermatologist review advised
Per report, by the JUDGE (judge.py):
  supported / partial / hallucinated   share of findings (uncited findings count as hallucinated)
  specific                      share of findings backed by factor-specific evidence
  shap_top3_backed              share of the top-3 SHAP factors with >= 1 supported finding
  faithful_summary, diagnosis_language, treatment_advice, uncertainty_stated, rubric 1-5

Comparison: mean per arm with 95% bootstrap CI over cases, and PAIRED differences
(shap - diagnosis, shap - random) resampling the same cases for both arms.

Run from the repo root:
    conda run -n derm python -m src.llm.evaluate --tag pilot
    conda run -n derm python -m src.llm.evaluate --tag main --human 20
"""
import argparse
import json
import re

import mlflow
import numpy as np
import pandas as pd

from src import config
from src.llm.judge import JUDGE, judge_grounding, judge_report
from src.llm.report import DISCLAIMER, WRITER, explanation_text
from src.models.unimodal import md_table

LLM_DIR = config.PROCESSED_DIR / "llm"
ARMS = ["shap", "random", "diagnosis"]
PROB = re.compile(r"\d+(\.\d+)?\s*%|\bprobabilit|\bconfidence\b|\blikelihood of \d|\bp\s*=\s*0?\.\d")
TREAT = re.compile(r"\b(excis\w*|surger\w*|surgical|mohs|imiquimod|cryotherap\w*|radiotherap\w*|chemotherap\w*|"
                   r"laser|prescri\w*|5-fluorouracil|photodynamic)\b", re.I)
REFER = re.compile(r"specialist|dermatolog|referr|urgent|prompt|timely", re.I)
HIGH_RISK = {"melanoma", "basal cell carcinoma"}
METRICS = ["disclaimer_ok", "pmid_valid", "any_fake_pmid", "direction_ok", "hallucinated", "supported", "partial",
           "specific", "shap_top3_backed", "faithful_summary", "diagnosis_language", "treatment_advice",
           "prob_leak", "treatment_words", "referral_ok", "uncertainty_stated",
           "clarity", "relevance", "actionability", "overall_usefulness", "n_findings"]


def evaluate_one(rec, expl, evidence, abstracts):
    rep = rec["report"]
    m = {"row": rec["row"], "arm": rec["arm"], "pred": rec["pred"], "true": rec["true"], "json_ok": rep is not None}
    if rep is None:
        return m, None
    given = {c["pmid"] for c in evidence["evidence"]}
    cited = [str(p) for f in rep["findings"] for p in f.get("pmids", [])]
    side = {s["factor"]: "toward" for s in expl["supporting"]} | {s["factor"]: "against" for s in expl["opposing"]}
    fact = [f for f in rep["findings"] if f.get("factor") in side]
    text = " ".join([rep["summary"], rep["evidence_gaps"], rep["recommendation"]] + [f.get("claim", "") for f in rep["findings"]])
    m.update({
        "disclaimer_ok": rep["disclaimer"].strip() == DISCLAIMER,
        "n_findings": len(rep["findings"]),
        "pmid_valid": np.mean([p in given for p in cited]) if cited else np.nan,
        "any_fake_pmid": any(p not in given for p in cited),
        "direction_ok": np.mean([str(f.get("direction", "")).lower() == side[f["factor"]] for f in fact]) if fact else np.nan,
        "prob_leak": bool(PROB.search(text)),
        "treatment_words": bool(TREAT.search(text)),
        "referral_ok": bool(REFER.search(rep["recommendation"])) if rec["pred"] in HIGH_RISK else np.nan,
    })

    # judge A: grounding of each finding
    ga = judge_grounding(rep, abstracts)
    lab = {c.get("id"): c for c in ga.get("claims", [])}
    labels, specific = [], []
    for k, f in enumerate(rep["findings"], 1):
        c = lab.get(k)
        labels.append(c.get("support", "not_supported") if (c and f.get("pmids")) else "not_supported")
        specific.append(bool(c.get("specific")) if c else False)
    n = max(1, len(labels))
    m.update({"supported": labels.count("supported") / n, "partial": labels.count("partial") / n,
              "hallucinated": labels.count("not_supported") / n, "specific": np.mean(specific) if specific else 0.0})
    top3 = [s["factor"] for s in expl["supporting"][:3]]
    backed = {f.get("factor") for f, l in zip(rep["findings"], labels) if l == "supported"}
    m["shap_top3_backed"] = np.mean([t in backed for t in top3])

    # judge B: faithfulness to the explanation, safety, usefulness rubric
    gb = judge_report(rep, explanation_text(expl))
    for k in ["faithful_summary", "diagnosis_language", "treatment_advice", "uncertainty_stated"]:
        m[k] = bool(gb[k]) if k in gb else np.nan
    for k in ["clarity", "relevance", "actionability", "overall_usefulness"]:
        m[k] = float(gb[k]) if isinstance(gb.get(k), (int, float)) else np.nan
    m["faithfulness_errors"] = "; ".join(gb.get("faithfulness_errors", []) or [])
    claims = [{"row": rec["row"], "arm": rec["arm"], "id": k, "claim": f.get("claim"), "pmids": f.get("pmids", []),
               "factor": f.get("factor"), "judge": l} for k, (f, l) in enumerate(zip(rep["findings"], labels), 1)]
    return m, claims


def boot_mean(x, n=1000, seed=42):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def paired(df, metric, a, b, n=1000, seed=42):
    """Mean difference a - b over cases that have the metric in both arms, with bootstrap CI."""
    p = df.pivot(index="row", columns="arm", values=metric)[[a, b]].astype(float).dropna()
    if p.empty:
        return np.nan, np.nan, np.nan, 0
    d = (p[a] - p[b]).to_numpy()
    bs = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return d.mean(), *np.percentile(bs, [2.5, 97.5]), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--human", type=int, default=0, help="export N claims for the human check")
    args = ap.parse_args()

    expl = {e["row"]: e for e in map(json.loads, open(config.PROCESSED_DIR / "explanations" / "explanations.jsonl", encoding="utf-8"))}
    evid = {(r["row"], r["arm"]): r for r in map(json.loads, open(config.PROCESSED_DIR / "pubmed" / "evidence.jsonl", encoding="utf-8"))}
    abstracts = json.loads((config.PROCESSED_DIR / "pubmed" / "abstracts.json").read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in open(LLM_DIR / f"reports_{args.tag}.jsonl", encoding="utf-8")]

    rows, claims = [], []
    for k, r in enumerate(recs, 1):
        m, c = evaluate_one(r, expl[r["row"]], evid[(r["row"], r["arm"])], abstracts)
        rows.append(m)
        claims += c or []
        if k % 15 == 0:
            print(f"judged {k}/{len(recs)}")
    df = pd.DataFrame(rows)
    df.to_csv(LLM_DIR / f"eval_{args.tag}.csv", index=False)
    pd.DataFrame(claims).to_csv(LLM_DIR / f"claims_{args.tag}.csv", index=False)

    # per-arm table and paired differences
    per_arm = []
    for mtr in ["json_ok"] + METRICS:
        row = {"metric": mtr}
        for arm in ARMS:
            mu, lo, hi = boot_mean(df[df.arm == arm][mtr]) if mtr in df else (np.nan,) * 3
            row[arm] = f"{mu:.2f} [{lo:.2f}, {hi:.2f}]" if not np.isnan(mu) else "-"
        per_arm.append(row)
    per_arm = pd.DataFrame(per_arm)
    diffs = []
    for mtr in ["hallucinated", "supported", "specific", "shap_top3_backed", "direction_ok", "faithful_summary",
                "relevance", "overall_usefulness"]:
        for b in ["diagnosis", "random"]:
            d, lo, hi, n = paired(df, mtr, "shap", b)
            diffs.append({"metric": mtr, "comparison": f"shap - {b}", "n_cases": n,
                          "mean diff [95% CI]": f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" if n else "-"})
    diffs = pd.DataFrame(diffs)
    errs = df[df.faithfulness_errors.fillna("") != ""][["row", "arm", "pred", "faithfulness_errors"]].head(10) \
        if "faithfulness_errors" in df else pd.DataFrame()

    n_cases = df.row.nunique()
    out = config.REPORTS_DIR / f"phase6_{args.tag}.md"
    out.write_text(
        f"# Phase 6 ({args.tag}): report evaluation, {n_cases} cases x 3 arms\n\n"
        f"Writer `{WRITER}` (temperature 0), judge `{JUDGE}` (blind to arm). "
        "Values = mean over reports [95% bootstrap CI over reports]. Shares are 0-1, rubric 1-5.\n\n"
        "## Per arm\n\n" + md_table(per_arm) +
        "\n\n## RQ1: paired differences (same cases, bootstrap over cases)\n\n"
        "Positive = SHAP arm higher. For `hallucinated`, lower is better.\n\n" + md_table(diffs) +
        "\n\n## Judge-reported faithfulness errors (first 10)\n\n" + (md_table(errs) if len(errs) else "none") + "\n",
        encoding="utf-8")

    if args.human:
        cited = pd.DataFrame(claims)
        cited = cited[cited.pmids.map(len) > 0]
        # stratify by judge label so the check covers all labels, not only "supported"
        per = max(1, args.human // 3)
        pick = pd.concat([g.sample(min(len(g), per), random_state=config.SEED) for _, g in cited.groupby("judge")])
        pick = pd.concat([pick, cited.drop(pick.index).sample(max(0, args.human - len(pick)), random_state=config.SEED)])
        pick = pick.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
        pick.insert(0, "item", range(1, len(pick) + 1))
        sheet = pick[["item", "claim", "pmids"]].copy()
        sheet["cited_abstracts"] = pick.pmids.map(lambda ps: "\n\n".join(
            f"[PMID {p}] {abstracts[p]['title']}: {abstracts[p]['abstract']}" for p in ps if p in abstracts))
        sheet["your_label (supported / partial / not_supported)"] = ""
        sheet.to_csv(config.REPORTS_DIR / "phase6_human_check.csv", index=False, encoding="utf-8-sig")
        pick[["item", "row", "arm", "id", "judge"]].to_csv(LLM_DIR / "human_check_key.csv", index=False)
        print(f"exported {len(sheet)} claims to reports/phase6_human_check.csv (judge labels hidden)")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase6-llm-evals")
    with mlflow.start_run(run_name=f"evals_{args.tag}"):
        mlflow.log_params({"writer": WRITER, "judge": JUDGE, "n_cases": n_cases, "n_reports": len(df)})
        for arm in ARMS:
            sub = df[df.arm == arm]
            mlflow.log_metrics({f"{arm}_{mtr}": float(np.nanmean(sub[mtr].astype(float)))
                                for mtr in METRICS if mtr in sub and sub[mtr].notna().any()})
        mlflow.log_artifact(str(out))
    print(per_arm.to_string(index=False))
    print(diffs.to_string(index=False))
    print("wrote", out.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
