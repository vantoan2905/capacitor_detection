"""Combine per-tile detections into full-image detections."""

from __future__ import annotations

import numpy as np

from capacitor_detection.tiling import Tile


def to_image_coords(boxes: np.ndarray, tile: Tile, width: int, height: int) -> np.ndarray:
    """Shift ``(N, 5)`` ``x1, y1, x2, y2, score`` boxes from tile to image coordinates."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 5).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] + tile.x0, 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] + tile.y0, 0, height)
    return boxes


def non_max_suppression(boxes: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Greedy NMS over ``(N, 5)`` ``x1, y1, x2, y2, score`` boxes; returns the kept boxes."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]) + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return boxes[keep]


def merge_tile_detections(
    detections: list[tuple[Tile, np.ndarray]], width: int, height: int, iou_threshold: float = 0.5
) -> np.ndarray:
    """Map each tile's boxes to image coordinates and suppress duplicates from overlapping tiles."""
    shifted = [to_image_coords(boxes, tile, width, height) for tile, boxes in detections]
    return non_max_suppression(np.concatenate(shifted) if shifted else np.empty((0, 5)), iou_threshold)
