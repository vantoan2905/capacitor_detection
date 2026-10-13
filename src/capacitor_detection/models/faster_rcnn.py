"""Faster R-CNN (torchvision) baseline on Pascal-VOC style data.

Expected layout of a dataset split::

    <split>/annotations/*.xml
    <split>/images/<same stem>.png
"""

from __future__ import annotations

import logging
import random
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

log = logging.getLogger(__name__)

SPLITS = {"train": 0.5, "val": 0.3, "test": 0.2}


def split_dataset(src_dir: Path, dst_dir: Path, image_ext: str = ".png", seed: int = 42) -> dict[str, int]:
    """Randomly copy ``src_dir/{annotations,images}`` into ``dst_dir/{train,val,test}/...``."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    stems = sorted(p.stem for p in (src_dir / "annotations").glob("*.xml"))
    random.Random(seed).shuffle(stems)
    counts, start = {}, 0
    for i, (split, fraction) in enumerate(SPLITS.items()):
        end = len(stems) if i == len(SPLITS) - 1 else start + round(len(stems) * fraction)
        for sub, ext in (("annotations", ".xml"), ("images", image_ext)):
            (dst_dir / split / sub).mkdir(parents=True, exist_ok=True)
            for stem in stems[start:end]:
                shutil.copyfile(src_dir / sub / f"{stem}{ext}", dst_dir / split / sub / f"{stem}{ext}")
        counts[split], start = end - start, end
    return counts


def parse_voc(xml_path: Path) -> dict[str, torch.Tensor]:
    """Parse a VOC annotation. Every object is mapped to class 1 (capacitor)."""
    root = ET.parse(xml_path).getroot()
    boxes = [
        [int(obj.find(f"bndbox/{k}").text) for k in ("xmin", "ymin", "xmax", "ymax")] for obj in root.findall("object")
    ]
    boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return {
        "boxes": boxes_t,
        "labels": torch.ones(len(boxes), dtype=torch.int64),
        "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
        "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
    }


class VOCDataset(Dataset):
    def __init__(self, split_dir: Path, image_ext: str = ".png"):
        self.split_dir = Path(split_dir)
        self.image_ext = image_ext
        self.annotations = sorted((self.split_dir / "annotations").glob("*.xml"))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        xml_path = self.annotations[idx]
        image = Image.open(self.split_dir / "images" / f"{xml_path.stem}{self.image_ext}").convert("RGB")
        target = parse_voc(xml_path)
        target["image_id"] = torch.tensor([idx])
        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int = 2) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=400, max_size=700)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    data_dir: Path,
    out_path: Path,
    num_classes: int = 2,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 0.005,
    image_ext: str = ".png",
) -> list[float]:
    """Train on ``data_dir/train`` and save the state dict to ``out_path``. Returns per-epoch loss."""
    device = _device()
    loader = DataLoader(
        VOCDataset(Path(data_dir) / "train", image_ext),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    model = build_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        start, batch_losses = time.time(), []
        for iteration, (images, targets) in enumerate(loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            if iteration % 100 == 0:
                log.info("iteration %d loss %.4f", iteration, loss.item())
        scheduler.step()
        losses.append(sum(batch_losses) / len(batch_losses))
        log.info("epoch %d/%d loss %.4f (%.1fs)", epoch, epochs, losses[-1], time.time() - start)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    torch.save(losses, out_path.with_name(f"{out_path.stem}_loss.pth"))
    return losses


@torch.no_grad()
def predict(weights: Path, image_dir: Path, out_dir: Path, num_classes: int = 2, threshold: float = 0.5) -> None:
    """Draw detections above ``threshold`` for every image in ``image_dir`` into ``out_dir``."""
    device = _device()
    model = build_model(num_classes)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(p for p in Path(image_dir).iterdir() if p.is_file()):
        image = Image.open(image_path).convert("RGB")
        prediction = model([F.to_tensor(image).to(device)])[0]
        draw = ImageDraw.Draw(image)
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            if score < threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)
            draw.text((x1, y1 - 10), f"{label.item()}: {score:.2f}", fill="red")
        image.save(out_dir / f"{image_path.stem}.png")
