"""Steps 2.4 + 2.5: simple models on one input type each (frozen embeddings / metadata).

Tasks
  concept_<name> : predict each of the 7 grouped concepts from the DERMOSCOPIC embedding
  diag_derm      : 5-class diagnosis from the dermoscopic embedding
  diag_clinic    : 5-class diagnosis from the clinical embedding (+ missing flag)
  diag_meta      : 5-class diagnosis from elevation, location, sex

Model: logistic regression (a linear classifier) with class_weight="balanced", so rare
classes like BCC count as much as nevus. The regularisation strength C (small C =
simpler model) is chosen by 5-fold cross-validation on TRAIN. Valid is kept for comparing
designs, test is scored once.

Out-of-fold (OOF) predictions for train: the fusion model (Phase 3) will use these
probabilities as inputs. If we predicted train cases with a model that had already
seen them, the probabilities would be over-confident and fusion would learn to trust
them too much. So each train case is predicted by a model trained on the other 4/5
of train (5-fold cross-validation). Valid/test use the model trained on all of train.

Run from the repo root:
    conda run -n derm python -m src.models.unimodal
"""
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data.concepts import CONCEPT_CLASSES, CONCEPTS
from src.eval.metrics import bootstrap_ci

EMB_DIR = config.PROCESSED_DIR / "embeddings"
PRED_DIR = config.PROCESSED_DIR / "preds"
CLASSES = ["nevus", "melanoma", "miscellaneous", "seborrheic keratosis", "basal cell carcinoma"]
C_GRID = [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
METADATA = ["elevation", "location", "sex"]


def make_model(kind, C):
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    if kind == "meta":   # categories -> 0/1 columns
        return make_pipeline(ColumnTransformer(
            [("oh", OneHotEncoder(handle_unknown="ignore"), METADATA)]), clf)
    return make_pipeline(StandardScaler(), clf)


def fit_task(name, kind, X, y, labels, split):
    """Pick C on valid, then produce probabilities for every case (OOF for train)."""
    tr, va, te = (split == "train"), (split == "valid"), (split == "test")
    sub = (lambda m: X[m]) if kind != "meta" else (lambda m: X.loc[m])

    # 1. choose C by 5-fold cross-validation on TRAIN (valid is too small/noisy to tune on)
    search = GridSearchCV(make_model(kind, 1.0), {"logisticregression__C": C_GRID},
                          cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED),
                          scoring="balanced_accuracy").fit(sub(tr), y[tr])
    best_C = search.best_params_["logisticregression__C"]

    # 2. final model on all train -> valid + test probabilities
    model = make_model(kind, best_C).fit(sub(tr), y[tr])
    order = [list(model.classes_).index(l) for l in labels]   # columns in `labels` order
    proba = np.zeros((len(y), len(labels)))
    proba[va | te] = model.predict_proba(sub(va | te))[:, order]

    # 3. out-of-fold probabilities for train
    tr_idx = np.where(tr)[0]
    for k_tr, k_te in StratifiedKFold(5, shuffle=True, random_state=config.SEED).split(tr_idx, y[tr]):
        m = make_model(kind, best_C).fit(sub(_mask(tr_idx[k_tr], len(y))), y[tr_idx[k_tr]])
        o = [list(m.classes_).index(l) for l in labels]
        proba[tr_idx[k_te]] = m.predict_proba(sub(_mask(tr_idx[k_te], len(y))))[:, o]

    res_va = bootstrap_ci(y[va], proba[va], labels)
    res_te = bootstrap_ci(y[te], proba[te], labels)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({"task": name, "input": kind, "model": "logreg_balanced",
                           "C": best_C, "encoder": "efficientnet_b0_frozen" if kind != "meta" else "none"})
        mlflow.log_metrics({f"valid_{k}": v for k, v in res_va.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in res_te.items()})

    pd.DataFrame(proba, columns=[f"{name}__{l}" for l in labels]).to_csv(PRED_DIR / f"{name}.csv", index=False)
    return {"task": name, "C": best_C,
            "valid_bal_acc": f"{res_va['bal_acc']:.3f} [{res_va['bal_acc_lo']:.3f}, {res_va['bal_acc_hi']:.3f}]",
            "test_bal_acc": f"{res_te['bal_acc']:.3f} [{res_te['bal_acc_lo']:.3f}, {res_te['bal_acc_hi']:.3f}]",
            "test_auc": f"{res_te['auc']:.3f} [{res_te['auc_lo']:.3f}, {res_te['auc_hi']:.3f}]"}


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in df.values])


def _mask(idx, n):
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def main():
    cases = pd.read_csv(config.CASES_CSV)
    split = cases["split"].to_numpy()
    derm = np.load(EMB_DIR / "effb0_derm.npy")
    clinic = np.load(EMB_DIR / "effb0_clinic.npy")
    clinic = np.hstack([clinic, (~cases["has_clinic"]).to_numpy()[:, None].astype(float)])  # + missing flag
    y_diag = cases["diagnosis_5"].to_numpy()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase2-unimodal")

    rows = []
    for c in CONCEPTS:
        rows.append(fit_task(f"concept_{c}", "derm", derm, cases[f"{c}_grp"].to_numpy(),
                             CONCEPT_CLASSES[c], split))
        print(rows[-1])
    for name, kind, X in [("diag_derm", "derm", derm), ("diag_clinic", "clinic", clinic),
                          ("diag_meta", "meta", cases[METADATA])]:
        rows.append(fit_task(name, kind, X, y_diag, CLASSES, split))
        print(rows[-1])

    table = pd.DataFrame(rows)
    out = config.REPORTS_DIR / "phase2_unimodal.md"
    out.write_text("# Phase 2: unimodal results (frozen EfficientNet-B0 + logistic regression)\n\n"
                   "Balanced accuracy and macro AUC with 95% bootstrap CI (1000 resamples).\n"
                   "Chance balanced accuracy = 1 / number of classes.\n\n"
                   + md_table(table) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print("wrote", out.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
