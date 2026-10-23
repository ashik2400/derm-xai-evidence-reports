"""Step 2.6: partially fine-tune EfficientNet-B0 on DERMOSCOPIC images (multi-task).

What "partial fine-tuning" means: the early layers (edges, colours, textures) stay frozen;
the last few blocks are unfrozen and adjusted to skin images. On top sit 8 small "heads":
1 for the 5-class diagnosis and 1 per checklist concept. Training them together
(multi-task) lets the concepts teach the shared features what to look at.

Trial run (this step): train on TRAIN, stop early on VALID loss, report VALID only.
Test is not touched here.

Run from the repo root:
    conda run -n derm python -m src.models.finetune_derm
"""
import time

import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data.concepts import CONCEPT_CLASSES, CONCEPTS
from src.data.dataset import CaseImages, eval_transform, train_transform
from src.eval.metrics import bootstrap_ci

CLASSES = ["nevus", "melanoma", "miscellaneous", "seborrheic keratosis", "basal cell carcinoma"]
TASKS = {"diagnosis": ("diagnosis_5", CLASSES)}
TASKS.update({c: (f"{c}_grp", CONCEPT_CLASSES[c]) for c in CONCEPTS})

HP = {  # hyperparameters (logged to MLflow)
    "encoder": "efficientnet_b0", "unfrozen_blocks": 3, "epochs_max": 30, "patience": 5,
    "batch_size": 16, "lr_backbone": 1e-4, "lr_heads": 1e-3, "weight_decay": 1e-4,
    "dropout": 0.3, "concept_loss_weight": 1.0, "seed": config.SEED,
}
WEIGHTS_PATH = config.ROOT / "models" / "effb0_derm_ft_trial.pt"


class LabelledImages(Dataset):
    """Dermoscopic image + integer label for every task."""

    def __init__(self, cases, transform):
        self.images = CaseImages(cases, "derm", transform)
        self.labels = np.stack([cases[col].map({l: i for i, l in enumerate(labs)}).to_numpy()
                                for col, labs in TASKS.values()], axis=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img, _ = self.images[i]
        return img, torch.as_tensor(self.labels[i])


class MultiTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(HP["encoder"], pretrained=True, num_classes=0)
        for p in self.backbone.parameters():              # freeze everything ...
            p.requires_grad = False
        for m in [*self.backbone.blocks[-HP["unfrozen_blocks"]:], self.backbone.conv_head, self.backbone.bn2]:
            for p in m.parameters():                      # ... then unfreeze the last blocks
                p.requires_grad = True
        self.drop = nn.Dropout(HP["dropout"])
        self.heads = nn.ModuleDict({t: nn.Linear(1280, len(labs)) for t, (_, labs) in TASKS.items()})

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return {t: h(f) for t, h in self.heads.items()}


def class_weights(y, n):
    """Rare classes get bigger weights so the loss treats every class equally."""
    counts = np.bincount(y, minlength=n).astype(float)
    return torch.tensor(len(y) / (n * np.maximum(counts, 1)), dtype=torch.float32)


def run_epoch(model, loader, losses, device, optimizer=None, scaler=None):
    """One pass over the data. Trains if an optimizer is given. Returns mean loss and probabilities."""
    model.train(optimizer is not None)
    total, n, probs = 0.0, 0, {t: [] for t in TASKS}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(optimizer is not None), torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x)
            loss_d = losses["diagnosis"](out["diagnosis"].float(), y[:, 0])
            loss_c = torch.stack([losses[c](out[c].float(), y[:, j + 1]) for j, c in enumerate(CONCEPTS)]).mean()
            loss = loss_d + HP["concept_loss_weight"] * loss_c
        if optimizer is not None:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += loss.item() * len(x)
        n += len(x)
        for t in TASKS:
            probs[t].append(out[t].float().softmax(1).detach().cpu().numpy())
    return total / n, {t: np.concatenate(p) for t, p in probs.items()}


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda")
    cases = pd.read_csv(config.CASES_CSV)
    train, valid = cases[cases.split == "train"], cases[cases.split == "valid"]

    train_ds = LabelledImages(train, train_transform())
    valid_ds = LabelledImages(valid, eval_transform())
    train_dl = DataLoader(train_ds, HP["batch_size"], shuffle=True, num_workers=2, persistent_workers=True)
    valid_dl = DataLoader(valid_ds, HP["batch_size"], shuffle=False, num_workers=2, persistent_workers=True)

    model = MultiTaskNet().to(device)
    losses = {t: nn.CrossEntropyLoss(weight=class_weights(train_ds.labels[:, j], len(labs)).to(device))
              for j, (t, (_, labs)) in enumerate(TASKS.items())}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": HP["lr_backbone"]},
        {"params": model.heads.parameters(), "lr": HP["lr_heads"]},
    ], weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP["epochs_max"])
    scaler = torch.amp.GradScaler()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase2-finetune")
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    best_loss, bad_epochs, best_probs = float("inf"), 0, None

    with mlflow.start_run(run_name="effb0_derm_ft_trial"):
        mlflow.log_params(HP)
        for epoch in range(1, HP["epochs_max"] + 1):
            t0 = time.time()
            tr_loss, _ = run_epoch(model, train_dl, losses, device, optimizer, scaler)
            va_loss, va_probs = run_epoch(model, valid_dl, losses, device)
            scheduler.step()
            mlflow.log_metrics({"train_loss": tr_loss, "valid_loss": va_loss}, step=epoch)
            print(f"epoch {epoch:2d}  train {tr_loss:.3f}  valid {va_loss:.3f}  ({time.time() - t0:.0f}s)")
            if va_loss < best_loss:                       # early stopping on valid loss
                best_loss, bad_epochs, best_probs = va_loss, 0, va_probs
                torch.save(model.state_dict(), WEIGHTS_PATH)
            else:
                bad_epochs += 1
                if bad_epochs >= HP["patience"]:
                    print("early stop")
                    break
        mlflow.log_metric("best_valid_loss", best_loss)

        # Compare with the frozen baseline on VALID, same metrics
        rows = []
        for t, (col, labs) in TASKS.items():
            name = "diag_derm" if t == "diagnosis" else f"concept_{t}"
            fz = pd.read_csv(config.PROCESSED_DIR / "preds" / f"{name}.csv").to_numpy()[cases.split.to_numpy() == "valid"]
            r_ft = bootstrap_ci(valid[col].to_numpy(), best_probs[t], labs)
            r_fz = bootstrap_ci(valid[col].to_numpy(), fz, labs)
            mlflow.log_metrics({f"valid_{t}_bal_acc": r_ft["bal_acc"], f"valid_{t}_auc": r_ft["auc"]})
            rows.append({"task": t,
                         "frozen_bal_acc": f"{r_fz['bal_acc']:.3f} [{r_fz['bal_acc_lo']:.3f}, {r_fz['bal_acc_hi']:.3f}]",
                         "finetuned_bal_acc": f"{r_ft['bal_acc']:.3f} [{r_ft['bal_acc_lo']:.3f}, {r_ft['bal_acc_hi']:.3f}]",
                         "frozen_auc": f"{r_fz['auc']:.3f}", "finetuned_auc": f"{r_ft['auc']:.3f}"})
        table = pd.DataFrame(rows)
        print("\nVALID comparison (frozen logistic regression vs fine-tuned heads):")
        print(table.to_string(index=False))
        table.to_csv(config.REPORTS_DIR / "phase2_finetune_trial_valid.csv", index=False)


if __name__ == "__main__":
    main()
