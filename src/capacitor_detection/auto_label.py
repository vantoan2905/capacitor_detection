"""Heuristic pre-labelling with classic image processing.

Finds the tray (largest bright blob), isolates dark rectangular components of roughly equal
size inside it, and writes them as labelme rectangles so they only need to be corrected by hand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from capacitor_detection.annotations import labelme_document

log = logging.getLogger(__name__)


def _candidate_mask(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, np.ones((15, 15)))
    mask = cv2.dilate(mask, np.ones((51, 51)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tray = max(contours, key=cv2.contourArea)
    mask = cv2.fillPoly(mask, [tray], 1)
    mask = gray * mask

    mask = cv2.GaussianBlur(mask, (27, 27), 0)
    mask = np.where(mask > np.mean(mask), 0, mask)
    mask = np.where(mask > np.mean(mask[mask > 0]), 0, mask)

    mask = cv2.GaussianBlur(mask, (27, 27), 0)
    mask = np.where(mask < np.mean(mask[mask > 0]), 0, mask)
    return cv2.dilate(mask, np.ones((3, 17)))


def auto_label(image_path: Path, label: str = "resitor", template_path: Path | None = None) -> int:
    """Write ``<image>.json`` (labelme) plus debug images next to ``image_path``.

    ``template_path`` optionally points to an existing labelme JSON whose top-level fields are
    reused. Returns the number of boxes found.
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), 1)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = _candidate_mask(gray)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours])
    lower, upper = areas.mean() * 2 / 3, areas.mean() * 4 / 3

    height, width = gray.shape[:2]
    if template_path:
        document = json.loads(Path(template_path).read_text())
        document.update(imagePath=image_path.name, imageHeight=height, imageWidth=width, shapes=[])
    else:
        document = labelme_document(image_path.name, height, width, [])

    for contour, area in zip(contours, areas):
        if not lower <= area <= upper:
            continue
        approx = cv2.approxPolyDP(contour, 0.08 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        document["shapes"].append(
            {
                "label": label,
                "points": [[x, y], [x + w, y + h]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )

    image_path.with_suffix(".json").write_text(json.dumps(document, indent=4))
    cv2.imwrite(f"{image_path}.jpg", mask / np.max(mask) * 255)
    cv2.imwrite(f"{image_path}_box.jpg", image)
    log.info("%s: %d boxes", image_path, len(document["shapes"]))
    return len(document["shapes"])
