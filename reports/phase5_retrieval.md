# Phase 5: PubMed evidence retrieval (3 arms for RQ1)

PubMed E-utilities, top 20 per query (English, abstract, humans, not retracted), re-ranked with `ncbi/MedCPT-Cross-Encoder`. 6 abstracts per case and arm. 136 unique queries, 736 unique abstracts. Retrieved on 2026-10-23 (PubMed changes daily; raw answers are cached in data/processed/pubmed/cache).

## Checks on test cases (395)

| arm | abstracts/case | queries <3 hits (fallback) | queries 0 hits | diagnosis mentioned | SHAP-factor coverage | unique PMIDs (all cases) | reviews | median year |
|---|---|---|---|---|---|---|---|---|
| shap | 5.94 | 0.02 | 0.00 | 0.94 | 0.98 | 114 | 0.17 | 2015 |
| random | 5.86 | 0.07 | 0.01 | 0.97 | 0.47 | 144 | 0.18 | 2017 |
| diagnosis | 6.00 | 0.00 | 0.00 | 0.96 | 0.36 | 25 | 0.71 | 2020 |

SHAP-factor coverage = share of a case's SHAP factors (top 3) named in at least one of the arm's abstracts; it is high for the SHAP arm by construction, so it is a coverage check, not the RQ1 result. RQ1 is answered in Phase 6 (report faithfulness and relevance).

## Overlap between arms

| arms | mean Jaccard |
|---|---|
| shap vs diagnosis | 0.05 |
| shap vs random | 0.03 |
| random vs diagnosis | 0.04 |

## Example: test case 4 (true basal cell carcinoma, predicted melanoma), SHAP arm

- [pigmentation: irregular (p=0.92)] PMID 33314891 (2021): Dermatoscopy of melanoma according to type, anatomic site and stage. (MedCPT 15.88)
- [pigmentation: irregular (p=0.92)] PMID 15998328 (2005): Detection of asymmetric blotches (asymmetric structureless areas) in dermoscopy images of malignant melanoma using relative color. (MedCPT 15.80)
- [streaks: irregular (p=0.93)] PMID 34347005 (2021): Assessment of Diagnostic Accuracy of Dermoscopic Structures and Patterns Used in Melanoma Detection: A Systematic Review and Meta-analysis. (MedCPT 15.88)
- [streaks: irregular (p=0.93)] PMID 40633957 (2025): Cutaneous non-volar melanoma dermoscopy and histopathology correlation: a systematic review protocol. (MedCPT 15.73)
- [dots and globules: irregular (p=0.90)] PMID 26898083 (2015): Dermoscopy and Early Melanoma. (MedCPT 15.99)
- [dots and globules: irregular (p=0.90)] PMID 29466542 (2018): Accuracy of Dermoscopic Criteria for the Diagnosis of Melanoma In Situ. (MedCPT 15.96)
