# Phase 3: fusion (named-feature logistic regression) vs unimodal baselines

Dermoscopic source chosen by train CV: **finetuned**. Final model: `fusion_finetuned` (44 features, C=0.001).
Balanced accuracy / macro AUC with 95% bootstrap CI (1000 resamples). Chance = 0.20.
cv_bal_acc = 5-fold CV on train (used for choices). occlude_* rows: train rows not scored.

| model | C | cv_bal_acc | valid_bal_acc | valid_auc | test_bal_acc | test_auc |
|---|---|---|---|---|---|---|
| fusion_frozen | 0.1 | 0.552 | 0.650 [0.538, 0.747] | 0.869 | 0.577 [0.508, 0.650] | 0.819 [0.782, 0.856] |
| fusion_finetuned | 0.001 | 0.606 | 0.600 [0.484, 0.702] | 0.883 | 0.590 [0.526, 0.660] | 0.842 [0.808, 0.875] |
| retrain_without_dermoscopic | 1.0 | 0.536 | 0.554 [0.434, 0.671] | 0.806 | 0.497 [0.421, 0.570] | 0.762 [0.723, 0.801] |
| retrain_without_clinical | 0.01 | 0.571 | 0.627 [0.518, 0.721] | 0.877 | 0.614 [0.539, 0.679] | 0.836 [0.795, 0.873] |
| retrain_without_metadata | 0.003 | 0.498 | 0.609 [0.499, 0.700] | 0.874 | 0.564 [0.493, 0.632] | 0.839 [0.803, 0.873] |
| occlude_dermoscopic | 0.001 | - | 0.440 [0.332, 0.559] | 0.800 | 0.415 [0.348, 0.489] | 0.766 [0.727, 0.803] |
| occlude_clinical | 0.001 | - | 0.545 [0.425, 0.650] | 0.866 | 0.546 [0.476, 0.612] | 0.836 [0.798, 0.870] |
| occlude_metadata | 0.001 | - | 0.572 [0.456, 0.684] | 0.864 | 0.579 [0.507, 0.646] | 0.842 [0.808, 0.872] |
| unimodal_derm_frozen | - | - | 0.477 [0.370, 0.596] | 0.843 | 0.462 [0.396, 0.539] | 0.840 [0.810, 0.869] |
| unimodal_derm_finetuned | - | - | 0.548 [0.430, 0.664] | 0.882 | 0.570 [0.494, 0.651] | 0.884 [0.860, 0.908] |
| unimodal_clinic | - | - | 0.421 [0.328, 0.533] | 0.753 | 0.383 [0.320, 0.449] | 0.745 [0.696, 0.789] |
| unimodal_meta | - | - | 0.416 [0.307, 0.535] | 0.712 | 0.448 [0.372, 0.521] | 0.687 [0.640, 0.731] |

## Paired test-set comparisons (bootstrap on the same cases)

Best unimodal baseline (chosen on valid): `unimodal_derm_finetuned`. p_not_better = share of bootstrap samples where the fusion model was not better.

| comparison | test_bal_acc_diff | p_not_better |
|---|---|---|
| fusion_finetuned - unimodal_derm_finetuned | +0.019 [-0.059, +0.090] | 0.340 |
| fusion_finetuned - retrain_without_dermoscopic | +0.093 [+0.032, +0.162] | 0.001 |
| fusion_finetuned - retrain_without_clinical | -0.024 [-0.070, +0.019] | 0.846 |
| fusion_finetuned - retrain_without_metadata | +0.026 [-0.014, +0.076] | 0.123 |
| fusion_finetuned - occlude_dermoscopic | +0.175 [+0.102, +0.245] | 0.000 |
| fusion_finetuned - occlude_clinical | +0.044 [-0.003, +0.093] | 0.033 |
| fusion_finetuned - occlude_metadata | +0.011 [-0.042, +0.068] | 0.340 |

## Test recall per class (sensitivity)

Test cases per class: nevus 219, melanoma 101, miscellaneous 40, seborrheic keratosis 19, basal cell carcinoma 16

| model | nevus | melanoma | miscellaneous | seborrheic keratosis | basal cell carcinoma |
|---|---|---|---|---|---|
| fusion_finetuned | 0.70 | 0.46 | 0.60 | 0.32 | 0.88 |
| unimodal_derm_finetuned | 0.72 | 0.54 | 0.65 | 0.32 | 0.62 |
