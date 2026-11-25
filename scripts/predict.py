"""Run a trained detector on images and save annotated copies.

    python scripts/predict.py --config configs/voc_bird_cat_dog.yaml --checkpoint best.pt \
        --input path/to/images_or_file --output-dir predictions --threshold 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from detr.config import Config  # noqa: E402
from detr.data.transforms import build_transforms  # noqa: E402
from detr.engine import autocast, postprocess, scale_boxes  # noqa: E402
from detr.models import build_model  # noqa: E402
from detr.utils import get_device, load_model_weights  # noqa: E402
from detr.visualize import draw_boxes  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@torch.no_grad()
def predict_image(model, image: Image.Image, transform, device, amp: bool = True) -> dict:
    """Returns pixel-space xyxy ``boxes``, ``scores`` and ``labels`` for every query."""
    empty = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long),
             "difficult": torch.zeros(0, dtype=torch.bool)}
    x, _ = transform(image, empty)
    mask = torch.zeros(1, *x.shape[1:], dtype=torch.bool, device=device)
    with autocast(device, amp):
        out = model(x[None].to(device), mask)
    pred = {k: v.cpu() for k, v in postprocess(out)[0].items()}
    pred["boxes"] = scale_boxes(pred["boxes"], torch.tensor([image.height, image.width]))
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="an image file or a directory of images")
    p.add_argument("--output-dir", default="predictions")
    p.add_argument("--threshold", type=float, default=0.9)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.model.pretrained_backbone = False
    device = get_device()
    model = build_model(cfg.model, len(cfg.data.classes))
    load_model_weights(model, args.checkpoint)
    model.to(device).eval()
    transform = build_transforms(False, cfg.data.train_scales, cfg.data.max_size, cfg.data.eval_size)

    src = Path(args.input)
    paths = sorted(q for q in src.iterdir() if q.suffix.lower() in IMAGE_EXTS) if src.is_dir() else [src]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        image = Image.open(path).convert("RGB")
        pred = predict_image(model, image, transform, device, cfg.train.amp)
        keep = pred["scores"] > args.threshold
        fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
        ax.imshow(image)
        draw_boxes(ax, pred["boxes"][keep], pred["labels"][keep], cfg.data.classes,
                   scores=pred["scores"][keep].tolist())
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(out_dir / f"{path.stem}_pred.jpg")
        plt.close(fig)
        found = ", ".join(f"{cfg.data.classes[lb]} {s:.2f}" for lb, s in
                          zip(pred["labels"][keep].tolist(), pred["scores"][keep].tolist()))
        print(f"{path.name}: {found or 'no detections'}")


if __name__ == "__main__":
    main()
