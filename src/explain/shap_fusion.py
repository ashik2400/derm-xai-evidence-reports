"""Phase 4.1: SHAP explanations of the fusion model.

SHAP splits one prediction into a contribution per feature: the model's score for a class
= base score (average case) + sum of feature contributions. For logistic regression this
is exact: contribution = weight x (value - training average), in log-odds units (the
scale the model adds things up in; +1 means the odds of that class go up ~2.7x).

What this script makes:
  1. SHAP values for every case (shap.LinearExplainer), cross-checked with the hand formula
  2. factor-level explanations for valid + test cases (JSON lines; input for Phase 5 queries)
  3. global importance (features, factors, modalities) on TEST
  4. RQ2: SHAP modality ranking vs the real drop when a modality is removed (Phase 3, MLflow)
  5. faithfulness (deletion test): removing SHAP's top factors should hurt more than random ones

Run from the repo root:
    conda run -n derm python -m src.explain.shap_fusion
"""
import json

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config
from src.explain.factors import MODALITY_OF_FACTOR, column_words, describe, factor_columns
from src.models.unimodal import md_table

OUT_DIR = config.PROCESSED_DIR / "explanations"
BLUE, RED, INK, MUTED, SURFACE = "#2a78d6", "#d0453a", "#1f1f1e", "#6b6b66", "#fcfcfb"
N_RANDOM = 50       # random draws for the deletion baseline


def bar_chart(names, values, title, xlabel, path, colors=None):
    """Thin horizontal bars, biggest on top, values written at the bar end."""
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, height=0.6, color=colors or BLUE)
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:+.2f}" if colors else f" {v:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.set_yticks(y, names, fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, color=INK)   # title spans the whole figure, not just the plot
    ax.grid(axis="x", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    bundle = joblib.load(config.ROOT / "models" / "fusion_lr.joblib")
    model, columns, classes = bundle["model"], bundle["features"], bundle["classes"]
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    order = [list(lr.classes_).index(c) for c in classes]            # our class order

    feats = pd.read_csv(config.PROCESSED_DIR / "fusion_features.csv")
    cases = pd.read_csv(config.CASES_CSV)
    split, y = feats["split"].to_numpy(), feats["diagnosis_5"].to_numpy()
    X = feats[columns].to_numpy()
    Xs = scaler.transform(X)
    tr, te = split == "train", split == "test"

    # 1. SHAP values: (cases, classes, features), background = train rows
    explainer = shap.LinearExplainer(lr, shap.maskers.Independent(Xs[tr], max_samples=int(tr.sum())))
    sv = np.asarray(explainer.shap_values(Xs))
    if sv.shape[1] == len(columns):                                  # (n, features, classes) -> (n, classes, features)
        sv = sv.transpose(0, 2, 1)
    sv = sv[:, order, :]
    hand = (lr.coef_[order][None, :, :] * (Xs - Xs[tr].mean(0))[:, None, :])
    print("max |shap - hand formula| =", float(np.abs(sv - hand).max()))
    base = np.asarray(explainer.expected_value)[order]
    logits = lr.decision_function(Xs)[:, order]
    print("additivity check (base + sum = model score), max error =", float(np.abs(base + sv.sum(2) - logits).max()))

    proba = model.predict_proba(X)[:, order]
    pred = proba.argmax(1)
    fcols = factor_columns(columns)
    factors = list(fcols)
    fsv = np.stack([sv[:, :, idx].sum(2) for idx in fcols.values()], axis=2)   # (n, classes, factors)
    own = fsv[np.arange(len(X)), pred]                                          # toward the predicted class

    # 2. per-case explanations for valid + test
    mean_tr = X[tr].mean(0)

    def driver(i, j):
        """The single column that pushed factor j most for case i, vs the train average."""
        idx = fcols[factors[j]]
        c = idx[int(np.argmax(np.abs(sv[i, pred[i], idx])))]
        side = "higher" if X[i, c] > mean_tr[c] else "lower"
        return f"{column_words(columns[c])} = {X[i, c]:.2f}, {side} than average {mean_tr[c]:.2f}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "shap_values.npy", sv.astype(np.float32))
    with open(OUT_DIR / "explanations.jsonl", "w", encoding="utf-8") as f:
        for i in np.where(~tr)[0]:
            ranked = np.argsort(-own[i])
            item = lambda j: {"factor": factors[j], "modality": MODALITY_OF_FACTOR[factors[j]],
                              "value": describe(factors[j], columns, X[i], classes[pred[i]]),
                              "driver": driver(i, j), "shap": round(float(own[i, j]), 3)}
            f.write(json.dumps({
                "row": int(i), "case_num": int(cases.case_num[i]), "split": split[i], "true": y[i],
                "pred": classes[pred[i]], "proba": {c: round(float(p), 3) for c, p in zip(classes, proba[i])},
                "supporting": [item(j) for j in ranked if own[i, j] > 0][:5],
                "opposing": [item(j) for j in ranked[::-1] if own[i, j] < 0][:3],
                "all": [item(j) for j in ranked],                       # all 13 factors, most supporting first
            }) + "\n")

    # 3. global importance on TEST: mean |SHAP| toward the predicted class
    feat_imp = pd.Series(np.abs(sv[te][np.arange(te.sum()), pred[te]]).mean(0), index=columns)
    fact_imp = pd.Series(np.abs(own[te]).mean(0), index=factors).sort_values(ascending=False)
    mods = ["dermoscopic", "clinical", "metadata"]
    msv = np.stack([own[:, [k for k, fa in enumerate(factors) if MODALITY_OF_FACTOR[fa] == m]].sum(1) for m in mods], 1)
    mod_imp = pd.Series(np.abs(msv[te]).mean(0), index=mods)

    # 4. RQ2: compare with Phase 3 removal drops (read from MLflow)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=["phase3-fusion"])
    score = dict(zip(runs["tags.mlflow.runName"], runs["metrics.test_bal_acc"]))
    full = score[f"fusion_{bundle['derm_source']}"]
    rq2 = pd.DataFrame({"mean_abs_shap": mod_imp,
                        "drop_retrain": [full - score[f"retrain_without_{m}"] for m in mods],
                        "drop_occlude": [full - score[f"occlude_{m}"] for m in mods]})
    for c in rq2:
        rq2[f"rank_{c}"] = rq2[c].rank(ascending=False).astype(int)

    # factor level: SHAP importance vs balanced-accuracy drop when that factor is occluded
    base_acc = balanced_accuracy_score(y[te], np.asarray(classes)[pred[te]])
    occ_drop = {}
    for fa, idx in fcols.items():
        Xo = X[te].copy()
        Xo[:, idx] = mean_tr[idx]
        occ_drop[fa] = base_acc - balanced_accuracy_score(y[te], np.asarray(classes)[model.predict_proba(Xo)[:, order].argmax(1)])
    occ_drop = pd.Series(occ_drop)
    rho, p_rho = spearmanr(fact_imp[factors], occ_drop[factors])

    # 5. deletion test: occlude top-k factors (by SHAP toward the predicted class) vs k random factors
    rng = np.random.default_rng(config.SEED)
    te_idx = np.where(te)[0]
    p0 = proba[te_idx, pred[te_idx]]
    rows = []
    for k in [1, 2, 3, 5]:
        top, rnd = np.zeros(len(te_idx)), np.zeros(len(te_idx))
        flip_top, flip_rnd = 0.0, 0.0
        for n, i in enumerate(te_idx):
            def drop(chosen):
                x = X[i].copy()
                for j in chosen:
                    x[fcols[factors[j]]] = mean_tr[fcols[factors[j]]]
                p = model.predict_proba(x[None])[0, order]
                return p0[n] - p[pred[i]], p.argmax() != pred[i]
            top[n], f_t = drop(np.argsort(-own[i])[:k])
            flip_top += f_t
            r = [drop(rng.choice(len(factors), k, replace=False)) for _ in range(N_RANDOM)]
            rnd[n] = np.mean([a for a, _ in r])
            flip_rnd += np.mean([b for _, b in r])
        rows.append({"k": k, "prob_drop_top": f"{top.mean():.3f}", "prob_drop_random": f"{rnd.mean():.3f}",
                     "flip_rate_top": f"{flip_top / len(te_idx):.2f}", "flip_rate_random": f"{flip_rnd / len(te_idx):.2f}",
                     "top_beats_random": f"{(top > rnd).mean():.2f}"})
    deletion = pd.DataFrame(rows)

    # 6. figures + report + MLflow
    fig_dir = config.FIGURES_DIR
    bar_chart(list(fact_imp.index), list(fact_imp.values), "Fusion: mean |SHAP| per factor (test)",
              "mean |SHAP| toward predicted class (log-odds)", fig_dir / "phase4_shap_factors.png")
    bar_chart(mods, list(mod_imp.values), "Fusion: mean |SHAP| per modality (test)",
              "mean |SHAP| toward predicted class (log-odds)", fig_dir / "phase4_shap_modalities.png")
    mel = np.where(te & (y == "melanoma") & (pred == classes.index("melanoma")))[0][0]   # first correctly caught melanoma
    js = np.argsort(-np.abs(own[mel]))
    bar_chart([describe(factors[j], columns, X[mel], "melanoma") for j in js], list(own[mel, js]),
              f"Test case {cases.case_num[mel]}: true melanoma, predicted melanoma (p={proba[mel, pred[mel]]:.2f})",
              "SHAP toward melanoma (log-odds)\nred = pushes toward melanoma, blue = pushes against",
              fig_dir / "phase4_shap_example_melanoma.png", colors=[RED if v > 0 else BLUE for v in own[mel, js]])

    rq2_out = rq2.copy()
    for c in ["mean_abs_shap", "drop_retrain", "drop_occlude"]:
        rq2_out[c] = rq2_out[c].map(lambda v: f"{v:.3f}")
    fact_tab = pd.DataFrame({"factor": factors, "modality": [MODALITY_OF_FACTOR[f] for f in factors],
                             "mean_abs_shap": fact_imp[factors].map(lambda v: f"{v:.3f}").values,
                             "bal_acc_drop_if_occluded": occ_drop[factors].map(lambda v: f"{v:+.3f}").values})
    fact_tab = fact_tab.iloc[np.argsort(-fact_imp[factors].values)]
    top_feats = feat_imp.sort_values(ascending=False).head(10)

    out = config.REPORTS_DIR / "phase4_shap.md"
    out.write_text(
        "# Phase 4.1: SHAP explanations of the fusion model\n\n"
        "Exact linear SHAP (shap.LinearExplainer, background = train rows), log-odds units, "
        "toward each case's PREDICTED class. Factor = sum of its columns. Test set (395 cases).\n\n"
        f"Checks: max |shap - hand formula| = {np.abs(sv - hand).max():.1e}; "
        f"additivity error = {np.abs(base + sv.sum(2) - logits).max():.1e}.\n\n"
        "## Factor importance\n\n![factors](figures/phase4_shap_factors.png)\n\n" + md_table(fact_tab) +
        f"\n\nSpearman rank correlation, mean |SHAP| vs balanced-accuracy drop when occluded (13 factors): "
        f"rho = {rho:.2f} (p = {p_rho:.3f}).\n\n"
        "## Top 10 single columns\n\n" + md_table(top_feats.map(lambda v: f"{v:.3f}").reset_index()
                                                  .rename(columns={"index": "column", 0: "mean_abs_shap"})) +
        "\n\n## RQ2: modality ranking, SHAP vs real removal drop (test balanced accuracy)\n\n"
        "![modalities](figures/phase4_shap_modalities.png)\n\n" + md_table(rq2_out.reset_index().rename(columns={"index": "modality"})) +
        "\n\n## Faithfulness: deletion test\n\n"
        "Occlude the top-k factors (by SHAP) vs k random factors (mean of 50 draws); "
        "drop in the probability of the predicted class, share of predictions that change, "
        "and share of cases where top-k hurt more than random.\n\n" + md_table(deletion) +
        "\n\n## Example explanation\n\n![example](figures/phase4_shap_example_melanoma.png)\n", encoding="utf-8")

    mlflow.set_experiment("phase4-explain")
    with mlflow.start_run(run_name="shap_fusion"):
        mlflow.log_params({"explainer": "LinearExplainer", "background": "train", "units": "log-odds"})
        mlflow.log_metrics({"rq2_factor_spearman": rho, **{f"mean_abs_shap_{m}": v for m, v in mod_imp.items()}})
        mlflow.log_artifact(str(out))

    print(fact_tab.to_string(index=False))
    print(f"factor spearman rho={rho:.2f} p={p_rho:.3f}")
    print(rq2_out.to_string())
    print(deletion.to_string(index=False))
    print("wrote", out.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
