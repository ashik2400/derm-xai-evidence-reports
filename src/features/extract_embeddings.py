"""Step 2.3: turn every image into an embedding with a frozen pretrained encoder.

An embedding is a vector of 1280 numbers that summarises what the encoder "sees".
We compute them once and save them, so later models train in seconds.

Output (rows in the same order as data/processed/cases.csv):
    data/processed/embeddings/effb0_derm.npy     shape (1011, 1280)
    data/processed/embeddings/effb0_clinic.npy   shape (1011, 1280), zeros where no clinical image
    data/processed/embeddings/effb0_meta.json    encoder name, settings, missing counts

Run from the repo root:
    conda run -n derm python -m src.features.extract_embeddings
"""
import json

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import CaseImages, eval_transform

ENCODER = "efficientnet_b0"   # timm name; ImageNet-pretrained weights
BATCH_SIZE = 16
OUT_DIR = config.PROCESSED_DIR / "embeddings"


def load_encoder(device):
    # num_classes=0 removes the ImageNet classifier, leaving the 1280-d pooled features
    model = timm.create_model(ENCODER, pretrained=True, num_classes=0)
    return model.eval().to(device)


@torch.no_grad()  # no training here, so skip gradient bookkeeping (faster, less memory)
def embed(model, cases, modality, device):
    loader = DataLoader(CaseImages(cases, modality, eval_transform()),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    feats = []
    for images, missing in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            f = model(images.to(device)).float()
        f[missing.bool().to(device)] = 0.0   # missing image -> zero vector (our rule)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    cases = pd.read_csv(config.CASES_CSV)
    model = load_encoder(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    info = {"encoder": ENCODER, "pretrained": "imagenet (timm default)", "img_size": 224,
            "transform": "resize 224 + center crop 224 + ImageNet normalise",
            "rows": "same order as cases.csv", "n_cases": len(cases)}
    for modality in ["derm", "clinic"]:
        x = embed(model, cases, modality, device)
        np.save(OUT_DIR / f"effb0_{modality}.npy", x)
        n_zero = int((np.abs(x).sum(1) == 0).sum())
        info[f"{modality}_shape"] = list(x.shape)
        info[f"{modality}_zero_rows"] = n_zero
        print(f"{modality}: {x.shape}, zero rows (missing) = {n_zero}")

    (OUT_DIR / "effb0_meta.json").write_text(json.dumps(info, indent=2))
    print("saved to", OUT_DIR.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
