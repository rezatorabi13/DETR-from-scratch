"""Plotting helpers for detections, attention maps, training curves and PR curves."""

from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from detr.data.transforms import IMAGENET_MEAN, IMAGENET_STD

PALETTE = ["#E8590C", "#7048E8", "#2B8A3E", "#1971C2", "#C2255C", "#0B7285"]


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu().float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def draw_boxes(ax, boxes_xyxy, labels, class_names: Sequence[str], scores=None,
               dashed: bool = False, lw: float = 2.0) -> None:
    """Draw pixel-space xyxy boxes; solid with a label tag, or dashed (ground truth)."""
    for k, (box, lbl) in enumerate(zip(np.asarray(boxes_xyxy).tolist(), np.asarray(labels).tolist())):
        x1, y1, x2, y2 = box
        color = PALETTE[lbl % len(PALETTE)]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=lw, ec=color, fc="none",
                                       ls="--" if dashed else "-"))
        if not dashed:
            text = class_names[lbl] + (f" {scores[k]:.2f}" if scores is not None else "")
            ax.text(x1, y1 - 3, text, color="white", fontsize=8, fontweight="bold",
                    bbox=dict(fc=color, ec="none", pad=1.5, alpha=0.9))


def plot_history(history: dict, lr_drop_epoch: int | None = None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for key, label in [("train_loss_ce", "classification (CE)"), ("train_loss_bbox", "box L1"),
                       ("train_loss_giou", "GIoU")]:
        axes[0].plot(epochs, history[key], lw=1.8, label=label)
    axes[0].set_title("Training loss components (final decoder layer)")

    axes[1].plot(epochs, history["train_match_acc"], lw=1.8, color="#7048E8", label="matched queries")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Classification accuracy of matched queries")

    axes[2].plot(history["val_epoch"], history["val_mAP50"], "o-", lw=1.8, label="mAP@0.5")
    axes[2].plot(history["val_epoch"], history["val_mAP"], "s-", lw=1.8, label="mAP@[.5:.95]")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Validation mAP")

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
        if lr_drop_epoch:
            ax.axvline(lr_drop_epoch, color="gray", ls=":", lw=1)
    fig.tight_layout()
    return fig


def plot_pr_curves(results: dict, class_names: Sequence[str]):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c, name in enumerate(class_names):
        recall, precision, _ = results["pr_curves"][c]
        ax.plot(recall, precision, lw=2, color=PALETTE[c % len(PALETTE)],
                label=f"{name} (AP@0.5 = {results['AP50_per_class'][c]:.2f})")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision-recall on VOC07 test (IoU 0.5)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def attention_maps(outputs: dict, query_ids: Sequence[int], image_hw: tuple[int, int]) -> list[np.ndarray]:
    """Per-query cross-attention of the last decoder layer, averaged over heads, upsampled to the image."""
    Hf, Wf = outputs["feat_hw"]
    h, w = image_hw
    attn = outputs["attn"][0].float().mean(0)  # [Q, Hf*Wf]
    maps = []
    for q in query_ids:
        a = attn[q].reshape(1, 1, Hf, Wf)
        a = F.interpolate(a, size=(math.ceil(h / 32) * 32, math.ceil(w / 32) * 32), mode="bilinear")
        a = a[0, 0, :h, :w]
        maps.append(((a - a.min()) / (a.max() - a.min() + 1e-8)).cpu().numpy())
    return maps
