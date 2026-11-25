"""Generate the README figures from a trained run.

    python scripts/make_figures.py --config configs/voc_bird_cat_dog.yaml \
        --checkpoint runs/voc_bird_cat_dog/best.pt --history runs/voc_bird_cat_dog/last.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from detr.config import Config  # noqa: E402
from detr.data import build_datasets, build_loader  # noqa: E402
from detr.engine import autocast, evaluate, postprocess  # noqa: E402
from detr.models import build_model  # noqa: E402
from detr.utils import get_device, load_model_weights  # noqa: E402
from detr.visualize import attention_maps, denormalize, draw_boxes, plot_history, plot_pr_curves  # noqa: E402


def load_history(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return torch.load(path, map_location="cpu", weights_only=False)["history"]


@torch.no_grad()
def run_single(model, x, device, return_attn=False):
    mask = torch.zeros(1, *x.shape[1:], dtype=torch.bool, device=device)
    with autocast(device, True):
        out = model(x[None].to(device), mask, return_attn=return_attn)
    return out, postprocess(out)[0]


def prediction_grid(model, ds, indices, classes, thresh, device):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, idx in zip(axes.flat, indices):
        x, t = ds[idx]
        h, w = x.shape[1:]
        _, p = run_single(model, x, device)
        keep = p["scores"] > thresh
        scale = torch.tensor([w, h, w, h])
        gt = t["boxes"].clone()
        gt_xyxy = torch.stack([gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2,
                               gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2], -1) * scale
        ax.imshow(denormalize(x))
        draw_boxes(ax, gt_xyxy, t["labels"], classes, dashed=True, lw=1.5)
        draw_boxes(ax, p["boxes"][keep].float().cpu() * scale, p["labels"][keep].cpu(), classes,
                   scores=p["scores"][keep].tolist())
        ax.axis("off")
    fig.suptitle(f"VOC07 test images | solid: predictions (score > {thresh}), dashed: ground truth | no NMS",
                 fontsize=12)
    fig.tight_layout()
    return fig


def attention_figure(model, ds, idx, classes, thresh, device, max_queries=3):
    x, _ = ds[idx]
    h, w = x.shape[1:]
    out, p = run_single(model, x, device, return_attn=True)
    queries = [q for q in p["scores"].argsort(descending=True).tolist() if p["scores"][q] > thresh][:max_queries]
    maps = attention_maps(out, queries, (h, w))
    img = denormalize(x)
    fig, axes = plt.subplots(1, len(queries) + 1, figsize=(4.6 * (len(queries) + 1), 4))
    axes[0].imshow(img)
    scale = torch.tensor([w, h, w, h])
    draw_boxes(axes[0], p["boxes"][queries].float().cpu() * scale, p["labels"][queries].cpu(), classes,
               scores=p["scores"][queries].tolist())
    axes[0].set_title("detections")
    for ax, q, a in zip(axes[1:], queries, maps):
        ax.imshow(img)
        ax.imshow(a, cmap="inferno", alpha=0.6)
        ax.set_title(f"query {q} cross-attention ({classes[p['labels'][q]]})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--history", help="history.json or last.pt containing the training history")
    p.add_argument("--out-dir", default="assets")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.model.pretrained_backbone = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    model = build_model(cfg.model, len(cfg.data.classes))
    load_model_weights(model, args.checkpoint)
    model.to(device).eval()
    _, _, test_ds = build_datasets(cfg)
    classes = cfg.data.classes

    rng = random.Random(args.seed)
    grid_idx = rng.sample(range(len(test_ds)), 8)
    prediction_grid(model, test_ds, grid_idx, classes, args.threshold, device).savefig(
        out_dir / "predictions.jpg", dpi=110)

    # attention: a test image whose confident detections match its 2-3 ground-truth objects
    for idx in rng.sample(range(len(test_ds)), len(test_ds)):
        x, t = test_ds[idx]
        _, pr = run_single(model, x, device)
        if 2 <= len(t["labels"]) <= 3 and int((pr["scores"] > args.threshold).sum()) == len(t["labels"]):
            break
    attention_figure(model, test_ds, idx, classes, args.threshold, device).savefig(
        out_dir / "attention.jpg", dpi=110)

    loader = build_loader(test_ds, cfg.train.batch_size, False, 0)
    res = evaluate(model, loader, device, len(classes))
    plot_pr_curves(res, classes).savefig(out_dir / "pr_curves.png", dpi=120)
    print(f"test mAP@0.5 {res['mAP50']:.3f} | mAP@[.5:.95] {res['mAP']:.3f}")

    if args.history:
        plot_history(load_history(Path(args.history)), cfg.train.lr_drop_epoch).savefig(
            out_dir / "training_curves.png", dpi=120)
    plt.close("all")
    print(f"figures written to {out_dir}/")


if __name__ == "__main__":
    np.random.seed(0)
    main()
