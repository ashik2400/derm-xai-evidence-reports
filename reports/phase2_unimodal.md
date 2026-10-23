# Phase 2: unimodal results (frozen EfficientNet-B0 + logistic regression)

Balanced accuracy and macro AUC with 95% bootstrap CI (1000 resamples).
Chance balanced accuracy = 1 / number of classes.

| task | C | valid_bal_acc | test_bal_acc | test_auc |
|---|---|---|---|---|
| concept_pigment_network | 0.01 | 0.593 [0.524, 0.671] | 0.628 [0.577, 0.676] | 0.798 [0.761, 0.832] |
| concept_streaks | 0.0001 | 0.551 [0.462, 0.638] | 0.608 [0.546, 0.669] | 0.754 [0.715, 0.792] |
| concept_pigmentation | 0.0003 | 0.562 [0.480, 0.642] | 0.523 [0.467, 0.584] | 0.735 [0.696, 0.777] |
| concept_regression_structures | 0.001 | 0.669 [0.591, 0.744] | 0.692 [0.644, 0.743] | 0.759 [0.704, 0.812] |
| concept_dots_and_globules | 0.001 | 0.531 [0.459, 0.607] | 0.581 [0.533, 0.630] | 0.741 [0.700, 0.778] |
| concept_blue_whitish_veil | 0.0001 | 0.740 [0.659, 0.813] | 0.739 [0.677, 0.796] | 0.834 [0.780, 0.882] |
| concept_vascular_structures | 0.01 | 0.510 [0.416, 0.603] | 0.533 [0.466, 0.605] | 0.797 [0.751, 0.842] |
| diag_derm | 0.3 | 0.477 [0.370, 0.596] | 0.462 [0.396, 0.539] | 0.840 [0.810, 0.869] |
| diag_clinic | 0.003 | 0.421 [0.328, 0.533] | 0.383 [0.320, 0.449] | 0.745 [0.696, 0.789] |
| diag_meta | 0.1 | 0.416 [0.307, 0.535] | 0.448 [0.372, 0.521] | 0.687 [0.640, 0.731] |
