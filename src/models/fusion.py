"""Phase 3: fusion classifier (late fusion of NAMED features) vs unimodal baselines.

"Late fusion" = each input type first gets its own model (Phase 2); their outputs are
then put side by side in one table and a final model combines them. Every column has a
human-readable name (e.g. concept_streaks__IR = "probability of irregular streaks"),
so SHAP in Phase 4 can rank features we can turn into PubMed search terms.

Feature groups (= modalities):
  dermoscopic : 19 predicted concept probabilities + 5 diagnosis probabilities
  clinical    : 5 diagnosis probabilities + clinic_missing flag (1 = no clinical photo)
  metadata    : elevation, location, sex as 0/1 columns (14)
Train rows use OOF predictions, valid/test rows the final Phase 2 models.

Model: standardise -> logistic regression (class_weight balanced). C chosen by 5-fold CV
on TRAIN. Design choices (frozen vs fine-tuned dermoscopic inputs) are also chosen by
train CV, BEFORE anything is scored. Then valid and test are scored once.

Modality removal (for RQ2), two kinds:
  retrain : train a new fusion model without that group  ("how much does the system need it?")
  occlude : keep the final model, replace the group by its train mean  ("how much does THIS
            model rely on it?"; after standardising, mean = 0 = no contribution)

Run from the repo root:
    conda run -n derm python -m src.models.fusion
"""
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.concepts import CONCEPTS
from src.eval.metrics import bootstrap_ci, paired_bootstrap_diff
from src.models.unimodal import CLASSES, C_GRID, METADATA, md_table

PRED_DIR = config.PROCESSED_DIR / "preds"
MODEL_PATH = config.ROOT / "models" / "fusion_lr.joblib"
CV = StratifiedKFold(5, shuffle=True, random_state=config.SEED)


def load_pred(name, strip=""):
    """Phase 2 probabilities; `strip` removes a name prefix (ft_) so both sources share column names."""
    df = pd.read_csv(PRED_DIR / f"{name}.csv")
    return df.rename(columns=lambda c: c.removeprefix(strip))


def feature_groups(cases, derm_source):
    """One DataFrame per modality. derm_source: "frozen" or "finetuned"."""
    pre = "ft_" if derm_source == "finetuned" else ""
    derm = pd.concat([load_pred(f"{pre}concept_{c}", pre) for c in CONCEPTS]
                     + [load_pred(f"{pre}diag_derm", pre)], axis=1)
    clinic = load_pred("diag_clinic").assign(clinic_missing=(~cases["has_clinic"]).astype(float))
    meta = pd.get_dummies(cases[METADATA], prefix_sep="=").astype(float)
    return {"dermoscopic": derm, "clinical": clinic, "metadata": meta}


def fit_fusion(X, y, tr):
    """Pick C by 5-fold CV on train, refit on all train. Returns (model, C, cv_bal_acc)."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
    search = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, cv=CV,
                          scoring="balanced_accuracy").fit(X[tr], y[tr])
    return search.best_estimator_, search.best_params_["logisticregression__C"], search.best_score_


def predict_all(model, X, y, tr, C):
    """Probabilities for every case: OOF for train, final model for valid/test."""
    proba = np.zeros((len(X), len(CLASSES)))
    order = [list(model.classes_).index(l) for l in CLASSES]
    proba[~tr] = model.predict_proba(X[~tr])[:, order]
    oof_model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    proba[tr] = cross_val_predict(oof_model, X[tr], y[tr], cv=CV, method="predict_proba")[:, order]
    return proba


def fmt(r, key):
    return f"{r[key]:.3f} [{r[key + '_lo']:.3f}, {r[key + '_hi']:.3f}]"


def main():
    cases = pd.read_csv(config.CASES_CSV)
    split = cases["split"].to_numpy()
    tr, va, te = split == "train", split == "valid", split == "test"
    y = cases["diagnosis_5"].to_numpy()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase3-fusion")
    probs, info = {}, {}       # model name -> probabilities for all cases / details

    # 1. design choice: frozen vs fine-tuned dermoscopic inputs, decided by TRAIN CV only
    for src in ["frozen", "finetuned"]:
        X = pd.concat(feature_groups(cases, src).values(), axis=1).to_numpy()
        model, C, cv = fit_fusion(X, y, tr)
        info[f"fusion_{src}"] = {"C": C, "cv_bal_acc": cv, "n_features": X.shape[1]}
        probs[f"fusion_{src}"] = predict_all(model, X, y, tr, C)
        print(f"fusion_{src}: C={C}  train-CV balanced accuracy {cv:.3f}")
    src = max(["frozen", "finetuned"], key=lambda s: info[f"fusion_{s}"]["cv_bal_acc"])
    print(f"-> chosen dermoscopic source: {src}")

    groups = feature_groups(cases, src)
    X_df = pd.concat(groups.values(), axis=1)
    X = X_df.to_numpy()
    final, C_final, _ = fit_fusion(X, y, tr)
    final_name = f"fusion_{src}"

    # 2. modality removal: retrain without each group
    for g in groups:
        Xg = pd.concat([d for k, d in groups.items() if k != g], axis=1).to_numpy()
        model, C, cv = fit_fusion(Xg, y, tr)
        name = f"retrain_without_{g}"
        info[name] = {"C": C, "cv_bal_acc": cv, "n_features": Xg.shape[1]}
        probs[name] = predict_all(model, Xg, y, tr, C)

    # 3. modality removal: occlude each group in the FINAL model (train mean)
    order = [list(final.classes_).index(l) for l in CLASSES]
    for g, d in groups.items():
        Xo = X_df.copy()
        Xo[d.columns] = X_df.loc[tr, d.columns].mean().to_numpy()
        name = f"occlude_{g}"
        p = np.zeros((len(X), len(CLASSES)))
        p[~tr] = final.predict_proba(Xo.to_numpy()[~tr])[:, order]
        probs[name], info[name] = p, {"C": C_final, "n_features": X.shape[1]}

    # 4. unimodal baselines (Phase 2 diagnosis probabilities as they are)
    for name, file, strip in [("unimodal_derm_frozen", "diag_derm", ""), ("unimodal_derm_finetuned", "ft_diag_derm", "ft_"),
                              ("unimodal_clinic", "diag_clinic", ""), ("unimodal_meta", "diag_meta", "")]:
        probs[name], info[name] = load_pred(file, strip).to_numpy(), {}

    # 5. score valid + test (once), log to MLflow
    rows, res = [], {}
    for name, p in probs.items():
        r_va, r_te = bootstrap_ci(y[va], p[va], CLASSES), bootstrap_ci(y[te], p[te], CLASSES)
        res[name] = (r_va, r_te)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"derm_source": src if name.startswith(("retrain", "occlude")) else "-", **info[name]})
            mlflow.log_metrics({f"valid_{k}": v for k, v in r_va.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in r_te.items()})
        rows.append({"model": name, "C": info[name].get("C", "-"),
                     "cv_bal_acc": f"{info[name]['cv_bal_acc']:.3f}" if "cv_bal_acc" in info[name] else "-",
                     "valid_bal_acc": fmt(r_va, "bal_acc"), "valid_auc": f"{r_va['auc']:.3f}",
                     "test_bal_acc": fmt(r_te, "bal_acc"), "test_auc": fmt(r_te, "auc")})
    table = pd.DataFrame(rows)

    # 6. is fusion better than the best single input? (best baseline picked on VALID)
    baselines = [n for n in probs if n.startswith("unimodal")]
    best_base = max(baselines, key=lambda n: res[n][0]["bal_acc"])
    diffs = []
    for name in [best_base] + [n for n in probs if n.startswith(("retrain", "occlude"))]:
        d = paired_bootstrap_diff(y[te], probs[final_name][te], probs[name][te], CLASSES)
        diffs.append({"comparison": f"{final_name} - {name}",
                      "test_bal_acc_diff": f"{d['diff']:+.3f} [{d['diff_lo']:+.3f}, {d['diff_hi']:+.3f}]",
                      "p_not_better": f"{d['p']:.3f}"})
    diffs = pd.DataFrame(diffs)

    recall = pd.DataFrame([{"model": n, **{c: f"{res[n][1]['recall_' + c.replace(' ', '_')]:.2f}" for c in CLASSES}}
                           for n in [final_name, best_base]])
    test_counts = pd.Series(y[te]).value_counts().reindex(CLASSES)

    # 7. save
    pd.DataFrame(probs[final_name], columns=[f"fusion__{c}" for c in CLASSES]).to_csv(PRED_DIR / "fusion.csv", index=False)
    X_df.assign(split=split, diagnosis_5=y).to_csv(config.PROCESSED_DIR / "fusion_features.csv", index=False)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": list(X_df.columns), "groups": {g: list(d.columns) for g, d in groups.items()},
                 "classes": CLASSES, "derm_source": src}, MODEL_PATH)

    out = config.REPORTS_DIR / "phase3_fusion.md"
    out.write_text(
        "# Phase 3: fusion (named-feature logistic regression) vs unimodal baselines\n\n"
        f"Dermoscopic source chosen by train CV: **{src}**. Final model: `{final_name}` "
        f"({X.shape[1]} features, C={C_final}).\n"
        "Balanced accuracy / macro AUC with 95% bootstrap CI (1000 resamples). Chance = 0.20.\n"
        "cv_bal_acc = 5-fold CV on train (used for choices). occlude_* rows: train rows not scored.\n\n"
        + md_table(table) + "\n\n## Paired test-set comparisons (bootstrap on the same cases)\n\n"
        f"Best unimodal baseline (chosen on valid): `{best_base}`. "
        "p_not_better = share of bootstrap samples where the fusion model was not better.\n\n"
        + md_table(diffs) + "\n\n## Test recall per class (sensitivity)\n\n"
        "Test cases per class: " + ", ".join(f"{c} {n}" for c, n in test_counts.items()) + "\n\n"
        + md_table(recall) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(diffs.to_string(index=False))
    print(recall.to_string(index=False))
    print("wrote", out.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
