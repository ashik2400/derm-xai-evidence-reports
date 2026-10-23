"""Metrics with bootstrap confidence intervals.

Bootstrap = resample the test cases with replacement many times, recompute the metric
each time, and take the middle 95% of those values as the confidence interval (CI).
With ~400 test cases and tiny classes (BCC 16, SK 19 in test) the CI shows how much
a score could move just by luck of which cases were in the test set.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score


def macro_auc(y_true, proba, labels):
    """One-vs-rest ROC AUC averaged over classes.

    For each class: "this class vs all others" AUC from its probability column.
    Classes absent from y_true (can happen in a bootstrap sample) are skipped.
    For 2 classes we use just the positive class (labels[1]).
    """
    cols = [1] if len(labels) == 2 else range(len(labels))
    aucs = []
    for j in cols:
        is_j = np.asarray(y_true) == labels[j]
        if 0 < is_j.sum() < len(is_j):
            aucs.append(roc_auc_score(is_j, proba[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def bootstrap_ci(y_true, proba, labels, n_boot=1000, seed=42):
    """Point value and 95% CI for balanced accuracy and macro AUC.

    y_true: array of class names; proba: (n, n_classes) in the order of `labels`.
    """
    y_true = np.asarray(y_true)
    labels = list(labels)
    y_pred = np.asarray(labels)[proba.argmax(1)]
    rng = np.random.default_rng(seed)

    def scores(idx):
        return (balanced_accuracy_score(y_true[idx], y_pred[idx]),
                macro_auc(y_true[idx], proba[idx], labels))

    point = scores(np.arange(len(y_true)))
    boots = np.array([scores(rng.integers(0, len(y_true), len(y_true))) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    out = {"bal_acc": point[0], "bal_acc_lo": lo[0], "bal_acc_hi": hi[0],
           "auc": point[1], "auc_lo": lo[1], "auc_hi": hi[1]}
    # per-class recall (= sensitivity): how many of each true class we caught
    for lab, r in zip(labels, recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)):
        out[f"recall_{lab.replace(' ', '_')}"] = r
    return out
