# Phase 4 summary: what the explanations show (written by hand, not generated)

Details and tables: `phase4_shap.md` (SHAP, fusion model) and `phase4_gradcam.md` (Grad-CAM, dermoscopic network).

## SHAP (fusion model, test set)
- Sanity: shap.LinearExplainer matches the hand formula weight x (value - train mean) exactly (error 0.0);
  base + contributions = model score (error 6e-16).
- The dermoscopic network's own diagnosis (mean |SHAP| 0.105) and the clinical network's diagnosis (0.096)
  are the biggest factors, then vascular structures, elevation, dots and globules, location.
  The 7 concepts are individually small, but a concept is in the top 3 supporting factors for 331/395 test cases,
  so concept-based PubMed queries are possible for most cases.
- Explanations look clinically sensible in spot checks (e.g. BCC: regular vessels, nodular, no pigment network).
- Faithfulness (deletion test): removing SHAP's top-k factors always hurts more than removing k random ones
  (top beats random in 100% of test cases for k = 1, 2, 3, 5; k = 1 flips 27% of predictions vs 5% random).
  This is expected for a linear model, so it is a sanity check rather than a strong result.
- **Low confidence:** the fusion model (C = 0.001, strong regularisation) gives flat probabilities:
  the median top probability on test is 0.31 (5 classes). Ranking is fine (AUC 0.84), but the numbers are
  not trustworthy as "confidence". Reports must not present them as such; calibration belongs in Phase 7.

## RQ2: SHAP modality ranking vs real removal drop
| modality | mean abs SHAP | drop if retrained without | drop if occluded |
|---|---|---|---|
| dermoscopic | 0.301 (1st) | 0.093 (1st) | 0.175 (1st) |
| clinical | 0.099 (2nd) | -0.024 (3rd) | 0.044 (2nd) |
| metadata | 0.083 (3rd) | 0.026 (2nd) | 0.011 (3rd) |

- SHAP agrees with occlusion (same fixed model), and at factor level: Spearman rho = 0.82 (p = 0.001, 13 factors).
- SHAP disagrees with retraining on clinical vs metadata: the model *uses* the clinical branch, but a model
  trained without it does just as well (slightly better). Likely reason: the clinical photo's information
  is largely redundant with the dermoscopic one. Reading for the write-up: SHAP measures how much this model
  relies on a modality, not how much the task needs it. Only 3 modalities, so this is one observation, not a test.

## Grad-CAM (fine-tuned dermoscopic network, test set)
- Heat mostly on the lesion: on average 20% of heat falls in the outer 10% frame (36% of the area).
- Wrong predictions put more heat on the frame than correct ones (0.27 vs 0.16, Mann-Whitney p = 1.5e-10).
  Caveat: confounded by class (nevus, mostly correct, 0.14; BCC 0.33, SK and misc 0.30), so this is a
  pattern, not proof that off-lesion attention causes errors.
- Dark circular borders: after the centre crop, only 9/395 test images keep a dark border touching the edge
  (> 5% of pixels), so vignetting is a small issue for this network. In 1 of 6 correct and 1 of 3 wrong of
  these, heat on the border exceeds its area share.
- Artefacts seen by eye in the sample figures: heat on a ruler (case 983, correctly predicted miscellaneous),
  near a measuring grid (case 944), and on an air bubble/glare (case 996). Not quantified - rulers/bubbles have
  no simple automatic detector; note as a limitation.
- Concept heatmaps for "vascular structures" often sit on the skin around the lesion. Could be real
  peripheral vessels or a weak concept head (vascular was among the weakest concepts in Phase 2).

## Consequences for Phases 5-6
- Use `data/processed/explanations/explanations.jsonl`: per case the top supporting/opposing factors,
  with a plain value ("streaks: irregular (p=0.93)") and a `driver` field naming the column that pushed.
- The image-model diagnosis factors are not useful PubMed terms; queries should be built from concepts + metadata.
- Reports should not quote fusion probabilities as confidence (see low-confidence note).
