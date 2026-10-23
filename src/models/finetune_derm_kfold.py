"""Step 2.7: fine-tuned dermoscopic model with out-of-fold (OOF) predictions for fusion.

Same network and settings as the trial (finetune_derm.py), but run the way our tuning
protocol requires (hyperparameters from TRAIN only, valid for comparing designs):

  1. 5-fold on TRAIN (same folds as unimodal.py). For each fold: train on 4/5, early-stop
     on the held-out 1/5, and predict that 1/5. After 5 folds every train case has a
     prediction from a model that never trained on it (= OOF).
  2. Final model on ALL of train for a FIXED number of epochs = median best epoch of the
     5 folds. No valid-based early stopping, so valid stays an honest comparison set.
     It predicts valid and test.

Output: data/processed/preds/ft_diag_derm.csv and ft_concept_<name>.csv (same format as
the frozen preds), models/effb0_derm_ft_final.pt, MLflow experiment "phase2-finetune".
Test is predicted and saved (fusion needs it) but NOT scored here.

Run from the repo root:
    conda run -n derm python -m src.models.finetune_derm_kfold           # full run (~40 min)
    conda run -n derm python -m src.models.finetune_derm_kfold --smoke   # 1 epoch each, saves nothing
    add --workers 0 if image loading hangs (worker processes cannot start)
"""
import argparse
import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src import config
from src.data.concepts import CONCEPTS
from src.data.dataset import eval_transform, train_transform
from src.eval.metrics import bootstrap_ci
from src.models.finetune_derm import HP, TASKS, LabelledImages, MultiTaskNet, class_weights, run_epoch

PRED_DIR = config.PROCESSED_DIR / "preds"
FINAL_WEIGHTS = config.ROOT / "models" / "effb0_derm_ft_final.pt"
N_FOLDS = 5
NUM_WORKERS = 2   # image-loading helper processes; set by --workers (0 = load in main process)


def pred_name(task):
    return "ft_diag_derm" if task == "diagnosis" else f"ft_concept_{task}"


def loader(ds, shuffle):
    return DataLoader(ds, HP["batch_size"], shuffle=shuffle, num_workers=NUM_WORKERS,
                      persistent_workers=NUM_WORKERS > 0)


def train_model(fit_cases, stop_cases, epochs, device):
    """Train one network on fit_cases.

    If stop_cases is given: early-stop on their loss and return their probabilities at the
    best epoch. If not: train exactly `epochs` epochs. Returns (model, best_epoch, probs).
    """
    torch.manual_seed(config.SEED)
    fit_ds = LabelledImages(fit_cases, train_transform())
    fit_dl = loader(fit_ds, shuffle=True)
    stop_dl = loader(LabelledImages(stop_cases, eval_transform()), shuffle=False) if stop_cases is not None else None

    model = MultiTaskNet().to(device)
    losses = {t: nn.CrossEntropyLoss(weight=class_weights(fit_ds.labels[:, j], len(labs)).to(device))
              for j, (t, (_, labs)) in enumerate(TASKS.items())}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": HP["lr_backbone"]},
        {"params": model.heads.parameters(), "lr": HP["lr_heads"]},
    ], weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HP["epochs_max"])
    scaler = torch.amp.GradScaler()

    best_loss, best_epoch, bad_epochs, best_probs, best_state = float("inf"), epochs, 0, None, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, fit_dl, losses, device, optimizer, scaler)
        scheduler.step()
        msg = f"  epoch {epoch:2d}  train {tr_loss:.3f}"
        if stop_dl is not None:
            st_loss, st_probs = run_epoch(model, stop_dl, losses, device)
            msg += f"  held-out {st_loss:.3f}"
            if st_loss < best_loss:                       # early stopping on the held-out fold
                best_loss, best_epoch, bad_epochs, best_probs = st_loss, epoch, 0, st_probs
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
        print(msg + f"  ({time.time() - t0:.0f}s)")
        if stop_dl is not None and bad_epochs >= HP["patience"]:
            print("  early stop")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_probs


def predict(model, cases, device):
    """Probabilities for every task on `cases` (no augmentation)."""
    dl = loader(LabelledImages(cases, eval_transform()), shuffle=False)
    dummy = {t: nn.CrossEntropyLoss() for t in TASKS}     # run_epoch needs losses; value unused here
    _, probs = run_epoch(model, dl, dummy, device)
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 epoch per model, save nothing")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    args = parser.parse_args()
    global NUM_WORKERS
    NUM_WORKERS = args.workers
    epochs_max = 1 if args.smoke else HP["epochs_max"]

    np.random.seed(config.SEED)
    device = torch.device("cuda")
    cases = pd.read_csv(config.CASES_CSV)
    split = cases["split"].to_numpy()
    tr_idx = np.where(split == "train")[0]
    vt_idx = np.where(split != "train")[0]                # valid + test
    probs = {t: np.zeros((len(cases), len(labs))) for t, (_, labs) in TASKS.items()}

    # 1. 5-fold OOF on train (same folds as unimodal.py: stratified on diagnosis, same seed)
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=config.SEED)
    best_epochs = []
    for k, (k_fit, k_out) in enumerate(folds.split(tr_idx, cases["diagnosis_5"].to_numpy()[tr_idx]), 1):
        print(f"fold {k}/{N_FOLDS}")
        out_idx = tr_idx[k_out]
        _, best_epoch, out_probs = train_model(cases.iloc[tr_idx[k_fit]], cases.iloc[out_idx], epochs_max, device)
        best_epochs.append(best_epoch)
        for t in TASKS:
            probs[t][out_idx] = out_probs[t]
        torch.cuda.empty_cache()

    # 2. final model on all train, fixed epoch count chosen from the folds
    final_epochs = int(np.median(best_epochs))
    print(f"best epochs per fold: {best_epochs} -> final model trains {final_epochs} epochs")
    model, _, _ = train_model(cases.iloc[tr_idx], None, final_epochs, device)
    vt_probs = predict(model, cases.iloc[vt_idx], device)
    for t in TASKS:
        probs[t][vt_idx] = vt_probs[t]

    # 3. score OOF-train and valid (not test)
    va = split == "valid"
    rows, metrics = [], {}
    for t, (col, labs) in TASKS.items():
        y = cases[col].to_numpy()
        r_oof = bootstrap_ci(y[tr_idx], probs[t][tr_idx], labs)
        r_va = bootstrap_ci(y[va], probs[t][va], labs)
        metrics.update({f"oof_{t}_bal_acc": r_oof["bal_acc"], f"valid_{t}_bal_acc": r_va["bal_acc"],
                        f"valid_{t}_auc": r_va["auc"]})
        rows.append({"task": t, "oof_train_bal_acc": f"{r_oof['bal_acc']:.3f}",
                     "valid_bal_acc": f"{r_va['bal_acc']:.3f} [{r_va['bal_acc_lo']:.3f}, {r_va['bal_acc_hi']:.3f}]",
                     "valid_auc": f"{r_va['auc']:.3f}"})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    if args.smoke:
        print("smoke run: nothing saved")
        return

    # 4. save predictions, weights, table, MLflow run
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for t, (_, labs) in TASKS.items():
        name = pred_name(t)
        pd.DataFrame(probs[t], columns=[f"{name}__{l}" for l in labs]).to_csv(PRED_DIR / f"{name}.csv", index=False)
    FINAL_WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), FINAL_WEIGHTS)
    table.to_csv(config.REPORTS_DIR / "phase2_finetune_kfold_valid.csv", index=False)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase2-finetune")
    with mlflow.start_run(run_name="effb0_derm_ft_kfold"):
        mlflow.log_params({**HP, "n_folds": N_FOLDS, "fold_best_epochs": str(best_epochs),
                           "final_epochs": final_epochs})
        mlflow.log_metrics(metrics)
    print("saved preds to", PRED_DIR.relative_to(config.ROOT), "and weights to", FINAL_WEIGHTS.relative_to(config.ROOT))


if __name__ == "__main__":
    main()
