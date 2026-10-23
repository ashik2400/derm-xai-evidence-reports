"""Step 1.3: build one clean table of all Derm7pt cases.

Reads data/raw (never writes there) and saves data/processed/cases.csv with:
  - split          : train / valid / test from the official index files
  - diagnosis_5    : 5-class label (original 20-class `diagnosis` is kept)
  - derm_path, clinic_path : real image paths found case-insensitively
  - has_derm, has_clinic   : False if the image file is missing on disk

Run from the repo root:
    conda run -n derm python -m src.data.build_cases
"""
import pandas as pd

from src import config
from src.data.concepts import CONCEPT_GROUPS

# --- 20 diagnosis labels -> 5 classes (from CLAUDE.md, checked against counts) ---
NEVUS = ["clark nevus", "reed or spitz nevus", "dermal nevus", "blue nevus",
         "congenital nevus", "combined nevus", "recurrent nevus"]
MELANOMA = ["melanoma (less than 0.76 mm)", "melanoma (in situ)",
            "melanoma (0.76 to 1.5 mm)", "melanoma (more than 1.5 mm)",
            "melanoma metastasis", "melanoma"]
MISC = ["vascular lesion", "lentigo", "dermatofibroma", "melanosis", "miscellaneous"]

DIAGNOSIS_5 = {d: "nevus" for d in NEVUS}
DIAGNOSIS_5.update({d: "melanoma" for d in MELANOMA})
DIAGNOSIS_5.update({d: "miscellaneous" for d in MISC})
DIAGNOSIS_5["seborrheic keratosis"] = "seborrheic keratosis"
DIAGNOSIS_5["basal cell carcinoma"] = "basal cell carcinoma"


def load_splits(n_rows):
    """Return a list: split name for each row of meta.csv.

    Checks that the index files are row positions, cover every row once, and do not overlap.
    """
    split_of_row = [None] * n_rows
    for name, path in config.SPLIT_FILES.items():
        idx = pd.read_csv(path)["indexes"].tolist()
        assert min(idx) >= 0 and max(idx) < n_rows, f"{name}: index out of range"
        for i in idx:
            assert split_of_row[i] is None, f"row {i} is in two splits"
            split_of_row[i] = name
    assert None not in split_of_row, "some rows are in no split"
    return split_of_row


def build_image_lookup():
    """Map lower-case relative path -> real path, so 'NEL/Nel026.jpg' finds 'NEL/NEL026.JPG'."""
    lookup = {}
    for f in config.IMAGES_DIR.rglob("*"):
        if f.is_file():
            rel = f.relative_to(config.IMAGES_DIR).as_posix()
            lookup[rel.lower()] = rel
    return lookup


def main():
    meta = pd.read_csv(config.META_CSV)
    print(f"meta.csv: {meta.shape[0]} rows, {meta.shape[1]} columns")

    # 1. Official split
    meta["split"] = load_splits(len(meta))

    # 2. 5-class label (fail loudly if a label is not in the mapping)
    unknown = set(meta["diagnosis"]) - set(DIAGNOSIS_5)
    assert not unknown, f"unmapped diagnosis labels: {unknown}"
    meta["diagnosis_5"] = meta["diagnosis"].map(DIAGNOSIS_5)

    # 2b. Grouped concept labels (paper grouping). Raw concept columns are kept for detail.
    for c, groups in CONCEPT_GROUPS.items():
        unknown = set(meta[c]) - set(groups)
        assert not unknown, f"{c}: ungrouped values {unknown}"
        meta[f"{c}_grp"] = meta[c].map(groups)

    # 3. Find images on disk, ignoring upper/lower case
    lookup = build_image_lookup()
    print(f"image files on disk: {len(lookup)}")
    for col in ["derm", "clinic"]:
        meta[f"{col}_path"] = meta[col].str.lower().map(lookup)  # NaN if not found
        meta[f"has_{col}"] = meta[f"{col}_path"].notna()

    # Some cases reuse the dermoscopic file as their "clinical" image (checked by eye:
    # these are dermoscopic photos). Treat them as having NO real clinical image.
    meta["clinic_is_derm_copy"] = meta["clinic_path"] == meta["derm_path"]
    meta.loc[meta["clinic_is_derm_copy"], "has_clinic"] = False

    # --- Summary ---
    print("\nCases per split:")
    print(meta["split"].value_counts().reindex(["train", "valid", "test"]).to_string())

    print("\n5-class counts per split:")
    print(pd.crosstab(meta["diagnosis_5"], meta["split"], margins=True)
          [["train", "valid", "test", "All"]].to_string())

    print("\nGrouped concept counts (train):")
    train = meta[meta["split"] == "train"]
    for c in CONCEPT_GROUPS:
        print(f"  {c:22s} {train[f'{c}_grp'].value_counts().to_dict()}")

    print("\nMissing images:")
    print(f"  dermoscopic missing: {(~meta['has_derm']).sum()}")
    print(f"  clinical missing:    {(~meta['has_clinic']).sum()}"
          f"  (of which clinic = copy of derm: {meta['clinic_is_derm_copy'].sum()})")
    missing = meta[~meta["has_derm"] | ~meta["has_clinic"]]
    if len(missing):
        print(missing[["case_num", "diagnosis", "diagnosis_5", "split",
                       "clinic", "derm"]].to_string(index=False))

    # Images on disk that no case refers to (useful sanity check)
    used = set(meta["derm"].str.lower()) | set(meta["clinic"].str.lower())
    unused = sorted(lookup[k] for k in lookup if k not in used)
    print(f"\nImage files not referenced by any case: {len(unused)}")
    for u in unused[:20]:
        print("  ", u)

    # --- Save ---
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meta.to_csv(config.CASES_CSV, index=False)
    print(f"\nSaved {config.CASES_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
