"""Pascal VOC 2007: download, annotation parsing, class-subset filtering and splits."""

from __future__ import annotations

import logging
import random
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
)
VOC_FILES = ("VOCtrainval_06-Nov-2007.tar", "VOCtest_06-Nov-2007.tar")
VOC_MIRRORS = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/",
    "http://pjreddie.com/media/files/",
)


def download_voc2007(root: Path) -> Path:
    """Download and extract VOC 2007 trainval + test (~900 MB) into ``root`` if missing."""
    voc_dir = root / "VOCdevkit" / "VOC2007"
    if (voc_dir / "ImageSets" / "Main" / "test.txt").exists():
        return voc_dir
    root.mkdir(parents=True, exist_ok=True)
    for name in VOC_FILES:
        tar_path = root / name
        if not tar_path.exists():
            for mirror in VOC_MIRRORS:
                try:
                    log.info("Downloading %s", mirror + name)
                    urllib.request.urlretrieve(mirror + name, tar_path)
                    break
                except Exception as e:  # noqa: BLE001 - try the next mirror
                    log.warning("Mirror failed (%s): %s", mirror, e)
                    tar_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"Could not download {name}; place it in {root} manually.")
        log.info("Extracting %s", name)
        with tarfile.open(tar_path) as tar:
            try:
                tar.extractall(root, filter="data")
            except TypeError:  # Python without extraction filters
                tar.extractall(root)
    return voc_dir


def parse_voc_xml(xml_path: Path) -> dict:
    """Parse one VOC annotation. Boxes are converted to 0-based ``[x1, y1, x2, y2]`` pixels."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    record = {"id": xml_path.stem, "width": int(size.find("width").text),
              "height": int(size.find("height").text), "objects": []}
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        difficult = obj.find("difficult")
        record["objects"].append({
            "name": obj.find("name").text.strip(),
            "difficult": int(difficult.text) if difficult is not None else 0,
            "box": [float(bb.find("xmin").text) - 1, float(bb.find("ymin").text) - 1,
                    float(bb.find("xmax").text), float(bb.find("ymax").text)],
        })
    return record


def load_split(voc_dir: Path, split: str) -> list[dict]:
    ids = (voc_dir / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
    return [parse_voc_xml(voc_dir / "Annotations" / f"{i}.xml") for i in ids]


def is_clean(record: dict, classes: Sequence[str]) -> bool:
    """True if the image has objects and *all* of them belong to ``classes``.

    Images that also contain other classes are dropped rather than relabelled, so the
    model is never told that an unlabelled object (e.g. a person) is background.
    """
    names = {o["name"] for o in record["objects"]}
    return bool(names) and names <= set(classes)


def build_splits(voc_dir: Path, classes: Sequence[str], val_size: int, seed: int):
    """Train / val from VOC07 trainval (held out by image id), test = official VOC07 test."""
    trainval = [r for r in load_split(voc_dir, "trainval") if is_clean(r, classes)]
    test = [r for r in load_split(voc_dir, "test") if is_clean(r, classes)]
    rng = random.Random(seed)
    rng.shuffle(trainval)
    val, train = trainval[:val_size], trainval[val_size:]
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    return train, val, test


class VOCDetection(Dataset):
    """VOC images restricted to a class subset. Labels are indices into ``classes``."""

    def __init__(self, records: list[dict], voc_dir: Path, classes: Sequence[str], transforms):
        self.records, self.voc_dir, self.transforms = records, voc_dir, transforms
        self.classes = list(classes)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(self.voc_dir / "JPEGImages" / f"{rec['id']}.jpg").convert("RGB")
        objs = rec["objects"]
        target = {
            "boxes": torch.tensor([o["box"] for o in objs], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([self.cls_to_idx[o["name"]] for o in objs], dtype=torch.long),
            "difficult": torch.tensor([o["difficult"] for o in objs], dtype=torch.bool),
            "image_id": torch.tensor(idx),
            "orig_size": torch.tensor([rec["height"], rec["width"]]),
        }
        return self.transforms(img, target)
