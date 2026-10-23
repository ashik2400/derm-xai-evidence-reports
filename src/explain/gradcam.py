"""Phase 4.2: Grad-CAM heatmaps for the fine-tuned dermoscopic network + shortcut check.

Grad-CAM answers "WHERE in the image did this head look?":
  1. take the last feature map of the network (1280 channels x 7 x 7 grid cells)
  2. ask how much the head's score would change if each channel got stronger (= gradient)
  3. weight each channel by its average gradient, add them up, keep only positive values
  -> a 7x7 heatmap, enlarged to 224x224 and laid over the image.

Two heads per case:
  diagnosis : the network's own predicted diagnosis
  concept   : the concept that SHAP ranked highest for the fusion prediction (Phase 4.1),
              at its predicted value - links "irregular streaks mattered" to WHERE they were seen

Shortcut check (every test case): many images have a dark circular border (vignetting)
or a ruler. If the heat sits on these instead of the lesion, the network learned a shortcut.
  heat_on_dark : share of heat on the dark border, compared with its share of the image.
                 Dark border = dark pixels CONNECTED TO THE IMAGE EDGE (a dark lesion in
                 the middle is not border; this matters because melanomas are often dark)
  heat_on_ring : share of heat in the outer 10% frame (36% of the area)

Run from the repo root:
    conda run -n derm python -m src.explain.gradcam
"""
import json

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms as T

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config
from src.data.concepts import CONCEPT_CLASSES
from src.data.dataset import IMG_SIZE, eval_transform
from src.explain.factors import VALUE_WORDS
from src.models.finetune_derm import CLASSES, MultiTaskNet
from src.models.unimodal import md_table

WEIGHTS = config.ROOT / "models" / "effb0_derm_ft_final.pt"
OUT_DIR = config.PROCESSED_DIR / "explanations"
DARK = 0.12           # pixel brightness (0-1) below this counts as dark
RING = 22             # outer frame width in pixels (10% of 224)
PER_CLASS = 2         # cases per (class, correct/wrong) in the figures


def grad_cam(model, x, targets):
    """Heatmaps (224x224, 0-1) for several (head, class index) targets on one image."""
    fm = model.backbone.forward_features(x)                   # (1, 1280, 7, 7)
    pooled = model.backbone.forward_head(fm, pre_logits=True)
    out = {t: model.heads[t](pooled) for t in model.heads}
    cams = []
    for head, k in targets:
        grad = torch.autograd.grad(out[head][0, k], fm, retain_graph=True)[0]
        cam = F.relu((grad.mean((2, 3), keepdim=True) * fm).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cams.append((cam / (cam.max() + 1e-8)).detach().cpu().numpy())
    return cams, {t: o.softmax(1)[0].detach().cpu().numpy() for t, o in out.items()}


def border_dark(dark):
    """Keep only the dark regions that touch the image edge (vignetting, black corners)."""
    labels, _ = ndimage.label(dark)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return np.isin(labels, edge[edge > 0])


def main():
    device = torch.device("cuda")
    model = MultiTaskNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    cases = pd.read_csv(config.CASES_CSV)
    expl = {r["row"]: r for r in map(json.loads, open(OUT_DIR / "explanations.jsonl", encoding="utf-8"))}
    test_rows = np.where(cases.split == "test")[0]
    norm, show = eval_transform(), T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor()])
    ring = np.ones((IMG_SIZE, IMG_SIZE), bool)
    ring[RING:-RING, RING:-RING] = False

    records, keep = [], {}
    for i in test_rows:
        img = Image.open(config.IMAGES_DIR / cases.derm_path[i]).convert("RGB")
        x = norm(img)[None].to(device)
        # concept target: first concept among SHAP's supporting factors (fallback: none)
        concept = next((s["factor"][8:] for s in expl[i]["supporting"] if s["factor"].startswith("concept_")), None)
        with torch.no_grad():
            p = model(x)
        k_diag = int(p["diagnosis"].argmax())
        targets = [("diagnosis", k_diag)]
        if concept:
            k_con = int(p[concept].argmax())
            targets.append((concept, k_con))
        cams, probs = grad_cam(model, x, targets)

        pix = show(img).numpy()                               # (3, 224, 224), 0-1, what the network saw
        dark = border_dark(pix.mean(0) < DARK)
        cam = cams[0]
        records.append({"row": i, "case_num": cases.case_num[i], "true": cases.diagnosis_5[i],
                        "pred": CLASSES[k_diag], "dark_area": dark.mean(),
                        "heat_on_dark": (cam * dark).sum() / cam.sum(),
                        "heat_on_ring": (cam * ring).sum() / cam.sum()})
        keep[i] = (pix.transpose(1, 2, 0), cams, targets, probs)

    rec = pd.DataFrame(records)
    rec["correct"] = rec["true"] == rec["pred"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT_DIR / "gradcam_shortcut.csv", index=False)

    # shortcut summary
    vig = rec[rec.dark_area > 0.05]                           # images with a visible dark border
    summary = pd.DataFrame([
        {"check": "heat in outer 10% frame (uniform heat = 0.36)", "all test": f"{rec.heat_on_ring.mean():.3f}",
         "correct": f"{rec[rec.correct].heat_on_ring.mean():.3f}", "wrong": f"{rec[~rec.correct].heat_on_ring.mean():.3f}"},
        {"check": f"images with dark border (>5% dark pixels): n={len(vig)}", "all test": f"{len(vig) / len(rec):.2f} of images",
         "correct": f"{vig.correct.sum()}", "wrong": f"{(~vig.correct).sum()}"},
        {"check": "  dark share of image (mean)", "all test": f"{vig.dark_area.mean():.3f}",
         "correct": f"{vig[vig.correct].dark_area.mean():.3f}", "wrong": f"{vig[~vig.correct].dark_area.mean():.3f}"},
        {"check": "  heat on dark pixels (mean)", "all test": f"{vig.heat_on_dark.mean():.3f}",
         "correct": f"{vig[vig.correct].heat_on_dark.mean():.3f}", "wrong": f"{vig[~vig.correct].heat_on_dark.mean():.3f}"},
        {"check": "  share of images: heat on dark > dark share", "all test": f"{(vig.heat_on_dark > vig.dark_area).mean():.2f}",
         "correct": f"{(vig[vig.correct].heat_on_dark > vig[vig.correct].dark_area).mean():.2f}",
         "wrong": f"{(vig[~vig.correct].heat_on_dark > vig[~vig.correct].dark_area).mean():.2f}"},
    ])
    worst = rec.sort_values("heat_on_dark", ascending=False).head(5)

    # figures: per true class, PER_CLASS correct + PER_CLASS wrong, 10 cases per figure
    rng = np.random.default_rng(config.SEED)
    chosen = []
    for c in CLASSES:
        for ok in [True, False]:
            pool = rec[(rec.true == c) & (rec.correct == ok)].row.to_numpy()
            chosen += list(rng.choice(pool, min(PER_CLASS, len(pool)), replace=False))
    chosen += list(worst.row.head(2))                         # plus the 2 most border-focused cases
    figs = []
    for n_fig, start in enumerate(range(0, len(chosen), 11), 1):
        part = chosen[start:start + 11]
        fig, axes = plt.subplots(len(part), 3, figsize=(7.5, 2.5 * len(part)), facecolor="#fcfcfb")
        for r, i in enumerate(part):
            pix, cams, targets, probs = keep[i]
            row = rec[rec.row == i].iloc[0]
            axes[r, 0].imshow(pix)
            axes[r, 0].set_title(f"case {row.case_num}: true {row.true}", fontsize=7, loc="left")
            for col, (cam, (head, k)) in enumerate(zip(cams, targets), 1):
                labs = CLASSES if head == "diagnosis" else [VALUE_WORDS[v] for v in CONCEPT_CLASSES[head]]
                axes[r, col].imshow(pix)
                axes[r, col].imshow(cam, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
                name = "diagnosis" if head == "diagnosis" else head.replace("_", " ")
                axes[r, col].set_title(f"{name}: {labs[k]} (p={probs[head][k]:.2f})", fontsize=7, loc="left")
            for ax in axes[r]:
                ax.axis("off")
        fig.tight_layout()
        path = config.FIGURES_DIR / f"phase4_gradcam_{n_fig}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figs.append(path.name)

    out = config.REPORTS_DIR / "phase4_gradcam.md"
    out.write_text(
        "# Phase 4.2: Grad-CAM on the fine-tuned dermoscopic network\n\n"
        "Heatmap of the last feature map (7x7, enlarged). Middle column: the network's predicted diagnosis. "
        "Right column: the concept SHAP ranked highest for the fusion prediction, at its predicted value. "
        "Correct/wrong = the dermoscopic network's own diagnosis head (not the fusion model).\n\n"
        f"## Shortcut check (all {len(rec)} test images)\n\n"
        f"Dark border = pixels with brightness < {DARK} connected to the image edge. If heat ignored the border, 'heat on dark' would be below "
        "the dark share; if the network used the border as a shortcut, it would be above.\n\n"
        + md_table(summary) + "\n\nMost border-focused cases:\n\n"
        + md_table(worst[["case_num", "true", "pred", "dark_area", "heat_on_dark", "heat_on_ring"]].round(3)) + "\n\n"
        + "\n\n".join(f"![gradcam]({'figures/' + f})" for f in figs) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("phase4-explain")
    with mlflow.start_run(run_name="gradcam_derm_ft"):
        mlflow.log_params({"target_layer": "conv_head+bn2 (7x7)", "dark_threshold": DARK, "ring_px": RING})
        mlflow.log_metrics({"heat_on_ring_mean": rec.heat_on_ring.mean(), "vignette_share": len(vig) / len(rec),
                            "heat_on_dark_mean_vignette": vig.heat_on_dark.mean(), "dark_area_mean_vignette": vig.dark_area.mean()})
        mlflow.log_artifact(str(out))
    print(summary.to_string(index=False))
    print(worst.to_string(index=False))
    print("wrote", out.relative_to(config.ROOT), figs)


if __name__ == "__main__":
    main()
