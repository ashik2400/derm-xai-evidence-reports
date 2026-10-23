"""The 7 checklist concepts and how their raw values are grouped.

Grouping copied from the official derm7pt code (Kawahara et al. 2019,
class Derm7PtDatasetGroupInfrequent in derm7pt/dataset.py):
  ABS = absent, TYP/ATP = typical/atypical, REG/IR = regular/irregular, PRS = present.
Concepts not listed in a group keep their own values (they already have 2-3 values).
"""

CONCEPT_GROUPS = {
    "pigment_network": {"absent": "ABS", "typical": "TYP", "atypical": "ATP"},
    "streaks": {"absent": "ABS", "regular": "REG", "irregular": "IR"},
    "pigmentation": {"absent": "ABS",
                     "diffuse regular": "REG", "localized regular": "REG",
                     "diffuse irregular": "IR", "localized irregular": "IR"},
    "regression_structures": {"absent": "ABS",
                              "blue areas": "PRS", "white areas": "PRS", "combinations": "PRS"},
    "dots_and_globules": {"absent": "ABS", "regular": "REG", "irregular": "IR"},
    "blue_whitish_veil": {"absent": "ABS", "present": "PRS"},
    "vascular_structures": {"absent": "ABS",
                            "arborizing": "REG", "comma": "REG", "hairpin": "REG",
                            "within regression": "REG", "wreath": "REG",
                            "dotted": "IR", "linear irregular": "IR"},
}

CONCEPTS = list(CONCEPT_GROUPS)

# Fixed class order per concept (so label 0,1,2 always means the same thing)
CONCEPT_CLASSES = {c: sorted(set(m.values()), key=["ABS", "TYP", "REG", "ATP", "IR", "PRS"].index)
                   for c, m in CONCEPT_GROUPS.items()}
