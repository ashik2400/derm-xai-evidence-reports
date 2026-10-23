"""Step 5.2: turn a case's explanation into PubMed queries, for the 3 RQ1 arms.

  shap      : the top 3 SEARCHABLE supporting factors (Phase 4 SHAP), one query each
  random    : 3 searchable factors picked at random from the ones SHAP did NOT pick
              (seeded per case), one query each -> "does SHAP's choice matter?"
  diagnosis : one query with the predicted diagnosis only

Factor query = <diagnosis> AND <factor term> AND (dermoscopy OR dermatoscopy).
Only PREDICTED values are used (from explanations.jsonl), never the true checklist columns.

The term dictionary below is written by hand. "Absent" values search the sign in general
(abstracts about pigment network in BCC usually discuss its absence); the exact value goes
to the LLM in Phase 6. Factors with no useful search term (image-model diagnoses, sex,
flat/palpable, clinical photo missing) are skipped.
"""
import numpy as np

from src import config

DIAGNOSIS_TERMS = {
    "nevus": "(nevus OR naevus)",
    "melanoma": "melanoma",
    "miscellaneous": '(lentigo OR dermatofibroma OR "vascular lesion" OR angioma OR melanosis)',
    "seborrheic keratosis": '"seborrheic keratosis"',
    "basal cell carcinoma": '"basal cell carcinoma"',
}
DERMOSCOPY = "(dermoscopy OR dermatoscopy OR dermoscopic)"

CONCEPT_TERMS = {   # (concept, predicted value word) -> PubMed term
    ("pigment network", "typical"): '"typical pigment network"',
    ("pigment network", "atypical"): '"atypical pigment network"',
    ("pigment network", "absent"): '"pigment network"',
    ("streaks", "regular"): '("regular streaks" OR starburst)',
    ("streaks", "irregular"): '("irregular streaks" OR pseudopods OR "radial streaming")',
    ("streaks", "absent"): "(streaks OR pseudopods)",
    ("pigmentation", "regular"): '("regular pigmentation" OR "homogeneous pigmentation" OR "regular blotches")',
    ("pigmentation", "irregular"): '("irregular pigmentation" OR "irregular blotches" OR blotch)',
    ("pigmentation", "absent"): "(blotch OR blotches)",
    ("regression structures", "present"): '("regression structures" OR "scar-like" OR peppering)',
    ("regression structures", "absent"): '("regression structures" OR regression)',
    ("dots and globules", "regular"): '("regular dots" OR "regular globules")',
    ("dots and globules", "irregular"): '("irregular dots" OR "irregular globules")',
    ("dots and globules", "absent"): "(dots OR globules)",
    ("blue whitish veil", "present"): '("blue-whitish veil" OR "blue-white veil" OR "blue-white structures")',
    ("blue whitish veil", "absent"): '("blue-whitish veil" OR "blue-white veil")',
    ("vascular structures", "regular"): '("arborizing vessels" OR "comma vessels" OR "hairpin vessels" OR "wreath vessels")',
    ("vascular structures", "irregular"): '("dotted vessels" OR "linear irregular vessels" OR "polymorphous vessels")',
    ("vascular structures", "absent"): '("vascular structures" OR vessels)',
}
METADATA_TERMS = {
    "elevation: nodular": "nodular",
    "location: acral": "acral",
    "location: head neck": '(face OR "head and neck")',
    "location: back": "(trunk OR back)",
    "location: chest": "(trunk OR chest)",
    "location: abdomen": "(trunk OR abdomen)",
    "location: lower limbs": '(legs OR "lower extremities" OR "lower limb")',
    "location: upper limbs": '(arms OR "upper extremities" OR "upper limb")',
    "location: genital areas": "genital",
    "location: buttocks": "buttock",
}
N_FACTORS = 3


def factor_term(item):
    """PubMed term for one explanation item, or None if not searchable.

    item: {"factor": ..., "value": "streaks: irregular (p=0.93)", ...} from explanations.jsonl
    """
    f, value = item["factor"], item["value"]
    if f.startswith("concept_"):
        concept, rest = value.split(": ")
        return CONCEPT_TERMS[(concept, rest.split(" (")[0])]
    return METADATA_TERMS.get(value)


def plain_words(term):
    """PubMed syntax -> plain text for the re-ranker: '("a b" OR c)' -> 'a b c'."""
    return " ".join(term.replace('"', "").replace("(", "").replace(")", "").replace(" OR ", " ").split())


def build_queries(expl):
    """All queries for one case. Returns {arm: [ {factor, value, term, query, rerank_text} ]}."""
    diag = expl["pred"]
    dterm = DIAGNOSIS_TERMS[diag]
    rank_diag = diag if diag != "miscellaneous" else plain_words(dterm)

    def factor_query(item):
        term = factor_term(item)
        return {"factor": item["factor"], "value": item["value"], "term": term,
                "query": f"{dterm} AND {term} AND {DERMOSCOPY}",
                "rerank_text": f"{rank_diag} dermoscopy {item['value'].split(' (')[0].replace(':', '')}; {plain_words(term)}"}

    # shap arm: top searchable factors that push TOWARD the prediction (positive SHAP)
    shap_items = [s for s in expl["all"] if s["shap"] > 0 and factor_term(s)][:N_FACTORS]

    # random arm: searchable factors SHAP did not pick (any sign), seeded per case
    picked = {s["factor"] for s in shap_items}
    pool = [s for s in expl["all"] if factor_term(s) and s["factor"] not in picked]
    rng = np.random.default_rng(config.SEED + expl["row"])
    rand_items = [pool[j] for j in sorted(rng.choice(len(pool), min(N_FACTORS, len(pool)), replace=False))] if pool else []

    return {
        "shap": [factor_query(s) for s in shap_items],
        "random": [factor_query(s) for s in rand_items],
        "diagnosis": [{"factor": None, "value": diag, "term": None, "query": f"{dterm} AND {DERMOSCOPY}",
                       "rerank_text": f"{rank_diag} dermoscopy"}],
    }
