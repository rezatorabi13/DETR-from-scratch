"""Evaluate a checkpoint on the VOC07 test split (or val) and report mAP.

    python scripts/evaluate.py --config configs/voc_bird_cat_dog.yaml --checkpoint runs/.../best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detr.config import Config
from detr.data import build_datasets, build_loader
from detr.engine import evaluate
from detr.models import build_model
from detr.utils import get_device, load_model_weights


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output", help="optional JSON file for the metrics")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config, args.overrides)
    cfg.model.pretrained_backbone = False  # weights come from the checkpoint
    device = get_device()

    _, val_ds, test_ds = build_datasets(cfg)
    ds = test_ds if args.split == "test" else val_ds
    loader = build_loader(ds, cfg.train.batch_size, False, cfg.data.num_workers)

    model = build_model(cfg.model, len(cfg.data.classes))
    load_model_weights(model, args.checkpoint)
    model.to(device)

    res = evaluate(model, loader, device, len(cfg.data.classes), cfg.train.amp)
    print(f"\n{args.split} split: {len(ds)} images")
    print(f"{'class':<10}{'AP@0.5':>10}{'AP@[.5:.95]':>14}")
    for name, ap50, ap in zip(cfg.data.classes, res["AP50_per_class"], res["AP_per_class"]):
        print(f"{name:<10}{ap50:>10.3f}{ap:>14.3f}")
    print(f"{'mean':<10}{res['mAP50']:>10.3f}{res['mAP']:>14.3f}")

    if args.output:
        metrics = {k: v for k, v in res.items() if k != "pr_curves"}
        Path(args.output).write_text(json.dumps({"split": args.split, "classes": cfg.data.classes, **metrics},
                                                indent=2))


if __name__ == "__main__":
    main()
