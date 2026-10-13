"""Reading/writing labelme JSON and the per-image cell CSV files.

Cell CSV convention
-------------------
Every annotated image ``NNN.bmp`` has a sibling ``NNN.csv`` with one row per capacitor cell.
Data columns are stored in the order TL, BL, BR, TR, but for historical reasons the
headers of the last two corners are swapped::

    Top_Left_X/Y, Bottom_Left_X/Y, Top_Right_X/Y (= bottom-right), Bottom_Right_X/Y (= top-right)

Existing CSVs on disk use this layout, so it is kept as is. Always go through
:func:`read_quads` / :data:`CELL_COLUMNS` instead of indexing the columns by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELME_VERSION = "5.5.0"

# Column header -> geometric corner it actually holds.
CELL_COLUMNS = {
    "Top_Left": "tl",
    "Bottom_Left": "bl",
    "Top_Right": "br",
    "Bottom_Right": "tr",
}
CELL_HEADER = [f"{name}_{axis}" for name in CELL_COLUMNS for axis in ("X", "Y")]


def read_quads(df: pd.DataFrame) -> np.ndarray:
    """Return an ``(N, 4, 2)`` array of cell corners ordered TL, TR, BR, BL."""
    corner = {geo: df[[f"{name}_X", f"{name}_Y"]].to_numpy(float) for name, geo in CELL_COLUMNS.items()}
    return np.stack([corner["tl"], corner["tr"], corner["br"], corner["bl"]], axis=1)


def quads_to_frame(quads: np.ndarray) -> pd.DataFrame:
    """Inverse of :func:`read_quads`: ``(N, 4, 2)`` TL, TR, BR, BL -> cell CSV columns."""
    by_geo = {"tl": quads[:, 0], "tr": quads[:, 1], "br": quads[:, 2], "bl": quads[:, 3]}
    data = {}
    for name, geo in CELL_COLUMNS.items():
        data[f"{name}_X"] = by_geo[geo][:, 0]
        data[f"{name}_Y"] = by_geo[geo][:, 1]
    return pd.DataFrame(data, columns=CELL_HEADER)


def polygon_shape(label: str, points) -> dict:
    return {
        "label": label,
        "points": [[float(x), float(y)] for x, y in points],
        "group_id": None,
        "description": "",
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def labelme_document(image_name: str, height: int, width: int, shapes: list[dict]) -> dict:
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def write_labelme(path: Path, document: dict) -> None:
    Path(path).write_text(json.dumps(document, indent=4))


def load_shapes(json_path: Path) -> list[dict]:
    return json.loads(Path(json_path).read_text()).get("shapes", [])
