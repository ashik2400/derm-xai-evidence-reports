"""Turn the 44 fusion columns into 13 readable "factors".

Several columns belong together: the 3 probabilities of pigment network
(absent/typical/atypical) are one clinical sign, the 14 metadata columns are just 3
questions (elevation, location, sex). A factor = one such group. Because the fusion
model is linear, a factor's SHAP value is simply the SUM of its columns' SHAP values.

Factors are what Phase 5 turns into PubMed search terms, so each gets a plain name.
"""
from src.data.concepts import CONCEPTS

VALUE_WORDS = {"ABS": "absent", "TYP": "typical", "ATP": "atypical",
               "REG": "regular", "IR": "irregular", "PRS": "present"}
CONCEPT_WORDS = {c: c.replace("_", " ") for c in CONCEPTS}

MODALITY_OF_FACTOR = {**{f"concept_{c}": "dermoscopic" for c in CONCEPTS},
                      "diag_derm": "dermoscopic", "diag_clinic": "clinical", "clinic_missing": "clinical",
                      "elevation": "metadata", "location": "metadata", "sex": "metadata"}


def factor_of(column):
    """concept_streaks__IR -> concept_streaks, location=back -> location."""
    return column.split("__")[0].split("=")[0]


def factor_columns(columns):
    """{factor: [column positions]} in a fixed order."""
    out = {}
    for j, col in enumerate(columns):
        out.setdefault(factor_of(col), []).append(j)
    return out


def column_words(col):
    """concept_streaks__IR -> 'P(streaks irregular)', diag_clinic__melanoma -> 'clinical model P(melanoma)'."""
    if "=" in col:
        return col.replace("=", " ")
    factor, label = (col.split("__") + [""])[:2]
    if factor.startswith("concept_"):
        return f"P({CONCEPT_WORDS[factor[8:]]} {VALUE_WORDS[label]})"
    if factor.startswith("diag_"):
        return f"{'dermoscopic' if factor == 'diag_derm' else 'clinical'} model P({label})"
    return col.replace("_", " ")


def describe(factor, columns, row, target=None):
    """Plain-English value of a factor for one case, e.g. 'streaks: irregular (p=0.71)'.

    row: the case's feature values (unscaled), aligned with `columns`.
    target: the class being explained. For the image-model diagnosis blocks we show
    P(target), because that is what pushes the score (their top class can be different).
    """
    vals = {col: row[j] for j, col in enumerate(columns) if factor_of(col) == factor}
    if factor == "clinic_missing":
        return "clinical photo: missing" if next(iter(vals.values())) > 0.5 else "clinical photo: available"
    if factor in ("elevation", "location", "sex"):
        active = max(vals, key=vals.get).split("=")[1]
        return f"{factor}: {active}"
    best = max(vals, key=vals.get)                     # most likely value among the probabilities
    label = best.split("__")[1]
    if factor.startswith("concept_"):
        return f"{CONCEPT_WORDS[factor[8:]]}: {VALUE_WORDS[label]} (p={vals[best]:.2f})"
    who = "dermoscopic" if factor == "diag_derm" else "clinical"
    if target is not None:
        return f"{who} image model: P({target})={vals[f'{factor}__{target}']:.2f}"
    return f"{who} image model says: {label} (p={vals[best]:.2f})"
