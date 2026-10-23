# Project log

Newest session at the bottom. Each entry: what we did, decisions and why, what is next.

---

## Session 1 — 2026-10-23

### What we did
- Read CLAUDE.md and checked the repo state. The data is in `data/raw/release_v0/` (not `data/raw/derm7pt/`), so CLAUDE.md now says that.
- Fixed `.gitignore`: it contained PowerShell code instead of patterns, and was missing `.env`.
- Checked the `derm` conda env: Python 3.11, DVC, MLflow, pandas, scikit-learn already installed. PyTorch not yet (Phase 2). GPU: RTX 3050 4 GB detected.
- Started Phase 1. Created `src/config.py` (all paths in one place) and this log.

### Decisions
- Keep the raw folder name `release_v0` (no renaming of 2,000 files inside protected data/raw; it is the official release name).
- DVC remote: local folder outside the repo (`C:\Users\ashik\dvc-storage\swayam-xai-derm`). Simple, no accounts; can add Google Drive later.
- Data report: a Python script writes `reports/data_report.md` + PNG figures (re-runnable, easy to read).
- MLflow: local SQLite file `mlflow.db` (gitignored).

- 1.2 DVC: added local remote `localstore`, ran `dvc add data/raw` (2,020 files, 156 MB) and `dvc push`. Pointer file: `data/raw.dvc`. `dvc status -c` says cache and remote are in sync.
  - To commit (manually, outside Claude): `data/raw.dvc`, `.dvc/config`.

- 1.3 Wrote `src/data/build_cases.py` -> `data/processed/cases.csv` (1,011 rows).
  - Split files are 0-based row positions into meta.csv; 413/203/395, no overlap, all rows covered.
  - 5-class counts match CLAUDE.md: nevus 575, melanoma 252, misc 97, SK 45, BCC 42.
  - Small classes in valid: BCC 7, SK 10 -> valid scores for these will be noisy.
  - All dermoscopic images exist. 9 cases have NO real clinical image: the `clinic` column points
    to the same file as `derm` (checked by eye: dermoscopic photos). Cases 565, 800, 824, 827, 833,
    834, 839, 844, 851 (train 2, valid 5, test 2; all melanoma or miscellaneous).
    Flag columns: `clinic_is_derm_copy`, `has_clinic`.
- 1.4 Decisions (see below).

### Decisions (1.4)
- **Missing clinical image rule:** keep all 1,011 cases and the official split. For the 9 cases,
  the clinical input is a zero vector plus a `clinic_missing` flag (0/1). Why: test set stays 395
  (comparable with published Derm7pt work), no rare cases lost, and the same mechanism is used for
  the "remove clinical modality" experiment (RQ2). Never feed the dermoscopic image as "clinical".
- **5-class mapping:** confirmed as in CLAUDE.md (matches the Derm7pt paper). 20-class column kept.

- 1.5 MLflow: `src/utils/mlflow_check.py` logged a test run to experiment `setup-check`
  (store: `mlflow.db`, artifacts: `mlruns/`, both gitignored). View: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- 1.6 Data report: `src/data/data_report.py` -> `reports/data_report.md` + `reports/figures/`.
  - Always-nevus baseline = 55.4% plain accuracy on test -> use balanced accuracy / per-class metrics.
  - LEAK confirmed: every melanoma (252) and BCC (42) has management = excision; no "clinical follow up"
    case is melanoma. One-rule balanced accuracy hides this (20%) because nevus still dominates each
    value; RQ3 must test it with a real model. seven_point_score: melanoma mean 4.9 vs others ~0.8-1.6.
  - Images: many have a ruler/grid and black circular borders (vignetting) -> possible shortcut
    features; note for Grad-CAM checks in Phase 4.
- `dvc add data/processed` + push. To commit manually: `data/processed.dvc`.
- Working preference changed: no approval needed for small steps; ask at phase start and for
  important mid-phase decisions.

### Phase 2 decisions (approved)
- Encoder: EfficientNet-B0, ImageNet weights (timm `efficientnet_b0.ra_in1k`), for both image types.
- Concepts: official paper grouping (`src/data/concepts.py`, copied from derm7pt code,
  `Derm7PtDatasetGroupInfrequent`). Note: "within regression" vessels = REG in the official code.
  Raw concept values kept in cases.csv for finer PubMed terms later.
- Concept predictor input: dermoscopic only (checklist is a dermoscopy tool; keeps RQ2 clean).
- **Tuning protocol (fixed for the whole project):** hyperparameters by 5-fold CV on TRAIN ->
  VALID to compare designs -> TEST scored once per final model.
  Why: first try tuned C on valid (203 cases, BCC 7); valid went up while test went down = overfitting
  to valid noise. Switched before any design decision was based on test.

### Phase 2 progress
- 2.1 Installed torch 2.14.1+cu126, torchvision 0.29.1, timm 1.0.30. CUDA works on the RTX 3050.
- 2.2 `src/data/dataset.py`: CaseImages (eval/train transforms, missing image -> zeros + flag).
- 2.3 `src/features/extract_embeddings.py` -> `data/processed/embeddings/effb0_{derm,clinic}.npy`
  (1011 x 1280; 9 zero rows in clinic).
- 2.4/2.5 `src/models/unimodal.py`: logistic regression (class_weight balanced) on frozen embeddings /
  one-hot metadata. OOF (5-fold) probabilities for train, saved to `data/processed/preds/` for fusion.
  MLflow experiment `phase2-unimodal`. Table: `reports/phase2_unimodal.md`.
  Test balanced accuracy [95% CI]:
  - diagnosis: derm 0.462 [0.396, 0.539] (AUC 0.840), clinic 0.383 [0.320, 0.449] (AUC 0.745),
    metadata 0.448 [0.372, 0.521] (AUC 0.687). Chance = 0.20.
  - concepts: best blue_whitish_veil 0.739 (AUC 0.834), regression 0.692; weakest pigmentation 0.523,
    vascular 0.533 (only 23 train IR cases).
- Fixed bug: multi-class AUC was NaN (sklearn needs sorted labels); now own one-vs-rest average.
- `dvc add data/processed` + push again. To commit manually: `data/processed.dvc`.

### Next
- 2.6 partial fine-tuning of EfficientNet-B0 on dermoscopic images (needs approval: ~20-40 min run).

- 2.6 Trial fine-tune: `src/models/finetune_derm.py` (EfficientNet-B0, last 3 blocks + conv_head unfrozen,
  multi-task: diagnosis + 7 concept heads, class-weighted loss, AMP, batch 16). MLflow `phase2-finetune`.
  Weights: `models/effb0_derm_ft_trial.pt` (gitignored). Table: `reports/phase2_finetune_trial_valid.csv`.
  - Best valid loss 1.617 at epoch 11; early stop at 16. Train loss kept falling (2.57 -> 0.69) while
    valid flattened after ~epoch 9 -> starts overfitting quickly (413 train images).
  - VALID balanced accuracy, frozen LR -> fine-tuned: diagnosis 0.477 -> 0.590 (AUC 0.843 -> 0.895),
    streaks 0.551 -> 0.757, BWV 0.740 -> 0.801, regression 0.669 -> 0.702, pigment network 0.593 -> 0.640,
    vascular 0.510 -> 0.574, dots 0.531 -> 0.581, pigmentation unchanged 0.562. Every AUC went up.
  - Caveat: the epoch was chosen on VALID loss, so these valid numbers are slightly optimistic. CIs still
    overlap for most tasks (diagnosis CIs overlap a little); streaks is the clearest gain.
  - Test not touched.

- Decision: adopt fine-tuning for the dermoscopic branch (option A: 5-fold OOF fine-tuning).
- 2.7 `src/models/finetune_derm_kfold.py`: 5 folds on TRAIN (same folds as unimodal.py), each early-stops
  on its held-out fold -> OOF train preds; final model on all train for a FIXED epoch count = median fold
  best epoch (no valid-based stopping, so valid stays honest). Preds: `data/processed/preds/ft_*.csv`
  (frozen preds kept). Weights `models/effb0_derm_ft_final.pt`. Table `reports/phase2_finetune_kfold_valid.csv`.
  MLflow run `effb0_derm_ft_kfold`. Test predicted (for fusion) but not scored.
  - Fold best epochs [3, 5, 5, 6, 10] -> final trains 5 epochs (trial picked 11 using valid).
  - Problem: DataLoader worker processes could not start in Claude's sandbox ("Access is denied" on a
    pipe handle) and the run hung. Fix: `--workers 0` option (load images in main process; ~8 s/epoch).
    Fold 5 epoch 1 took 32 min once (laptop likely slept); no effect on results.
  - VALID balanced accuracy, frozen -> fine-tuned (AUC): diagnosis 0.477 -> 0.548 [0.430, 0.664] (0.843 -> 0.882),
    streaks 0.551 -> 0.679, BWV 0.740 -> 0.785, vascular 0.510 -> 0.613, dots 0.531 -> 0.578,
    pigmentation 0.562 -> 0.584, pigment network 0.593 -> 0.602, regression 0.669 -> 0.650. All valid AUCs up.
  - OOF TRAIN, frozen -> fine-tuned: mostly flat (diagnosis 0.427 -> 0.421); clear gains only for BWV
    (0.666 -> 0.772) and dots (0.551 -> 0.610); vascular worse (0.552 -> 0.517).
  - Reading: the gain is real but modest and not certain; CIs overlap. BWV is the most consistent win.
- `dvc add data/processed` + push. To commit manually: `data/processed.dvc`.

- Fixed `.gitignore`: `models/` also hid `src/models/` (code never committed) -> now `/models/`.
  Note: mlflow_check.py actually lives in `src/models/utils/`, not `src/utils/`.

---

## Phase 3 — fusion (2026-10-23)

### Decisions (approved)
- Fusion = late fusion, logistic regression (class_weight balanced, standardised) on 44 NAMED features:
  dermoscopic 24 (19 concept probs + 5 diagnosis probs), clinical 6 (5 diag probs + clinic_missing),
  metadata 14 (elevation/location/sex one-hot). Why: small data, and SHAP needs readable features for
  PubMed queries (RQ1). No embedding fusion, no boosted trees.
- Frozen vs fine-tuned dermoscopic inputs chosen by TRAIN CV (not valid). Test scored ONCE at the end.
- Modality removal for RQ2, two kinds: `retrain_without_<g>` (new model without the group) and
  `occlude_<g>` (final model, group set to train mean = zero contribution after standardising).

### What we did
- `src/eval/metrics.py`: added `paired_bootstrap_diff` (A - B on the same resampled cases).
- `src/models/fusion.py` -> `reports/phase3_fusion.md`, `data/processed/preds/fusion.csv`,
  `data/processed/fusion_features.csv` (input table for SHAP), `models/fusion_lr.joblib`
  (model + feature names + groups). MLflow experiment `phase3-fusion`.

### Results (TEST, balanced accuracy [95% CI], chance 0.20)
- Train CV picked fine-tuned (0.606 vs 0.552). Final `fusion_finetuned`, C=0.001:
  test 0.590 [0.526, 0.660], AUC 0.842. (Valid had preferred frozen fusion 0.650 vs 0.600 - we
  followed the protocol; valid is noisy.)
- Best unimodal (picked on valid) = fine-tuned dermoscopic: test 0.570 [0.494, 0.651], AUC 0.884.
  Fusion - unimodal = +0.019 [-0.059, +0.090], p=0.34 -> NO significant gain from fusion.
  Fusion AUC is lower than dermoscopic alone (0.842 vs 0.884).
- Modality removal (fusion minus variant): dermoscopic retrain +0.093 [+0.032, +0.162], occlude +0.175
  -> dermoscopic carries the model. Clinical: retraining WITHOUT it is slightly better (-0.024) ->
  clinical adds noise; occlude +0.044. Metadata: +0.026 retrain, +0.011 occlude (small, CIs cross 0).
- Recall (test): fusion BCC 0.88 vs 0.62 for derm alone (16 cases), but melanoma 0.46 vs 0.54
  (safety concern: missed melanomas; discuss threshold / cost weighting before reports).
- `dvc add data/processed` + push. To commit manually: `data/processed.dvc`, `src/models/fusion.py`,
  `src/eval/metrics.py`, `reports/phase3_fusion.md`, `docs/PROJECT_LOG.md`.

### Next (after Phase 3)
- Melanoma recall: deferred to Phase 7 as a labelled post-hoc variant (decided at Phase 4 start).

---

## Phase 4 — explanations (2026-10-23)

### Decisions (approved)
- Installed `shap` 0.51 (+ numba, llvmlite). SHAP = shap.LinearExplainer (exact for LR), background = train
  rows, log-odds units, explained toward each case's PREDICTED class. Cross-checked with the hand formula.
- Grad-CAM: own code (no install), last feature map (7x7) of the fine-tuned dermoscopic network.
- Melanoma recall issue deferred to Phase 7.

### What we did
- `src/explain/factors.py`: groups the 44 columns into 13 readable factors (7 concepts, 2 image-model
  diagnoses, clinic_missing, elevation, location, sex); factor SHAP = sum of its columns.
- `src/explain/shap_fusion.py` -> `reports/phase4_shap.md`, figures `phase4_shap_*.png`,
  `data/processed/explanations/shap_values.npy` (cases x classes x features) and `explanations.jsonl`
  (valid + test: top 5 supporting / 3 opposing factors, plain value + `driver` column). MLflow `phase4-explain`.
  - Bug found and fixed: the image-model factors were labelled with their TOP class ("clinical model says nevus")
    while the push came from P(melanoma) -> misleading. Now labelled with P(explained class) + driver field.
- `src/explain/gradcam.py` -> `reports/phase4_gradcam.md`, figures `phase4_gradcam_{1,2}.png`,
  `data/processed/explanations/gradcam_shortcut.csv`.
  - Bug found and fixed: "dark pixels" also counted dark lesions (melanomas!) as border. Now only dark regions
    connected to the image edge count.
- `reports/phase4_summary.md`: hand-written interpretation (read this first).

### Results
- SHAP checks exact. Top factors: dermoscopic-model diagnosis, clinical-model diagnosis, vascular, elevation.
  A concept is in the top 3 supporting factors for 331/395 test cases.
- Deletion test: top-k SHAP factors beat random in 100% of cases (expected for a linear model).
- RQ2: SHAP modality ranking derm > clinical > metadata = occlusion ranking; retraining ranking is
  derm > metadata > clinical. Factor level Spearman rho 0.82 (p 0.001) vs occlusion drop.
  SHAP shows reliance of this model, not necessity of the modality (clinical is redundant).
- Fusion probabilities are flat (median top prob 0.31) -> not usable as confidence; calibrate in Phase 7.
- Grad-CAM: heat mostly on lesion (20% in outer frame vs 36% area); wrong cases more off-lesion (0.27 vs 0.16,
  p 1.5e-10, confounded by class). Dark border rare after centre crop (9/395). Artefacts by eye: ruler (983),
  grid (944), bubble (996).
- `dvc add data/processed` + push. To commit manually: `data/processed.dvc`, `src/explain/`,
  `reports/phase4_*.md`, `reports/figures/phase4_*.png`, `docs/PROJECT_LOG.md`.

### Next
- Phase 5 (needs plan + approval): PubMed retrieval (E-utilities esearch + efetch) from the top SHAP
  concept/metadata factors, plus the diagnosis-only baseline queries for RQ1. Decide query templates,
  number of abstracts, ranking, caching, and NCBI API key/rate limits.
