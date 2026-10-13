"""Ultralytics YOLO training/prediction driven by the YAML files in ``configs/``."""

from __future__ import annotations

from pathlib import Path

import yaml


def train(config: Path, **overrides):
    """Train with an Ultralytics config file; keyword ``overrides`` win over the file."""
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(config).read_text())
    weights = overrides.pop("model", None) or cfg["model"]
    return YOLO(weights).train(cfg=str(config), **overrides)


def predict(weights: Path, source: Path, conf: float = 0.25, iou: float = 0.7, save_txt: bool = False, **kwargs):
    from ultralytics import YOLO

    return YOLO(str(weights)).predict(source=str(source), conf=conf, iou=iou, save=True, save_txt=save_txt, **kwargs)
