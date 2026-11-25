"""Train DETR.

    python scripts/train.py --config configs/voc_bird_cat_dog.yaml
    python scripts/train.py --config configs/voc_bird_cat_dog.yaml train.epochs=50 data.num_workers=0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from detr.config import Config
from detr.data import build_datasets, build_loader
from detr.engine import evaluate, train_one_epoch
from detr.losses import SetCriterion
from detr.matcher import HungarianMatcher
from detr.models import build_model
from detr.utils import get_device, seed_everything, setup_logging, warmup_step_schedule

log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt")
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.epochs=50")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config, args.overrides)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "train.log")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    seed_everything(cfg.train.seed)
    device = get_device()
    num_classes = len(cfg.data.classes)

    train_ds, val_ds, _ = build_datasets(cfg)
    train_loader = build_loader(train_ds, cfg.train.batch_size, True, cfg.data.num_workers)
    val_loader = build_loader(val_ds, cfg.train.batch_size, False, cfg.data.num_workers)
    log.info("classes %s | train %d | val %d images | device %s",
             cfg.data.classes, len(train_ds), len(val_ds), device)

    model = build_model(cfg.model, num_classes).to(device)
    matcher = HungarianMatcher(cfg.loss.cost_class, cfg.loss.cost_bbox, cfg.loss.cost_giou)
    criterion = SetCriterion(num_classes, matcher, cfg.loss.cost_class, cfg.loss.cost_bbox,
                             cfg.loss.cost_giou, cfg.loss.eos_coef, cfg.loss.aux_loss).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(cfg.train.lr, cfg.train.lr_backbone),
                                  weight_decay=cfg.train.weight_decay)
    steps_per_epoch = len(train_loader) // cfg.train.accum_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_step_schedule(cfg.train.warmup_iters, cfg.train.lr_drop_epoch * steps_per_epoch))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("model: %.1fM trainable parameters", n_params / 1e6)

    history, start_epoch, best_map = defaultdict(list), 0, -1.0
    last_ckpt, best_ckpt = out_dir / "last.pt", out_dir / "best.pt"
    if last_ckpt.exists() and not args.no_resume:
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_map = ckpt["epoch"] + 1, ckpt["best_map"]
        history = defaultdict(list, ckpt["history"])
        log.info("resumed from %s at epoch %d", last_ckpt, start_epoch)

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        stats = train_one_epoch(model, criterion, train_loader, optimizer, scheduler, device,
                                cfg.train.amp, cfg.train.clip_norm, cfg.train.accum_steps)
        for k, v in stats.items():
            history[f"train_{k}"].append(v)
        history["epoch_time"].append(time.time() - t0)
        msg = (f"epoch {epoch + 1:3d}/{cfg.train.epochs} | loss {stats['loss']:.3f} "
               f"(ce {stats['loss_ce']:.3f}, L1 {stats['loss_bbox']:.3f}, giou {stats['loss_giou']:.3f}) "
               f"| {history['epoch_time'][-1]:.0f}s")

        if (epoch + 1) % cfg.train.eval_every == 0 or epoch + 1 == cfg.train.epochs:
            res = evaluate(model, val_loader, device, num_classes, cfg.train.amp)
            history["val_epoch"].append(epoch + 1)
            history["val_mAP50"].append(res["mAP50"])
            history["val_mAP"].append(res["mAP"])
            msg += f" | val mAP@.5 {res['mAP50']:.3f} mAP {res['mAP']:.3f}"
            if res["mAP50"] > best_map:
                best_map = res["mAP50"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_mAP50": best_map,
                            "config": cfg.to_dict()}, best_ckpt)
                msg += " *"

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "best_map": best_map,
                    "history": dict(history), "config": cfg.to_dict()}, last_ckpt)
        (out_dir / "history.json").write_text(json.dumps(dict(history)))
        log.info(msg)

    log.info("done | best val mAP@0.5 = %.3f | checkpoint: %s", best_map, best_ckpt)


if __name__ == "__main__":
    main()
