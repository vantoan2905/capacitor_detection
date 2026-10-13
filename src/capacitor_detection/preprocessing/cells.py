"""Operations on cell CSV files: row/column indices, defect labels, number boxes, labelme export."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from capacitor_detection.annotations import (
    labelme_document,
    polygon_shape,
    quads_to_frame,
    read_quads,
    write_labelme,
)
from capacitor_detection.constants import IMAGE_HEIGHT, IMAGE_WIDTH, NOT_AVAILABLE, NUMB_BOX

log = logging.getLogger(__name__)


def add_row_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-based ``Col``/``Row`` columns to a row-major cell CSV.

    A new row starts when a cell's top-left x equals the bottom-left x of the first cell of the
    current row, i.e. the cell sits directly below it.
    """
    df = df.copy()
    cols, rows = [], []
    row, col = 1, 0
    row_start_bl_x = df["Bottom_Left_X"].iloc[0]
    for i, (tl_x, bl_x) in enumerate(zip(df["Top_Left_X"], df["Bottom_Left_X"])):
        col += 1
        if i > 0 and tl_x == row_start_bl_x:
            row, col = row + 1, 1
            row_start_bl_x = bl_x
        cols.append(col)
        rows.append(row)
    df["Col"] = cols
    df["Row"] = rows
    return df


def merge_defects(df: pd.DataFrame, image_id: int, defects: pd.DataFrame) -> pd.DataFrame:
    """Add a ``Defect`` column from a defect table with columns ``image_id, col, row, label``."""
    df = df.copy()
    df["Defect"] = NOT_AVAILABLE
    for _, (_, col, row, label) in defects[defects.iloc[:, 0] == image_id].iloc[:, :4].iterrows():
        df.loc[(df["Col"] == col) & (df["Row"] == row), "Defect"] = label
    return df


def annotate_cells(csv_dir: Path, out_dir: Path, defects_xlsx: Path | None = None) -> int:
    """Add Row/Col (and Defect, if a defect sheet is given) to every cell CSV under ``csv_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    defects = pd.read_excel(defects_xlsx) if defects_xlsx else None
    count = 0
    for csv_path in sorted(Path(csv_dir).rglob("*.csv")):
        df = add_row_col(pd.read_csv(csv_path))
        if defects is not None:
            df = merge_defects(df, int(csv_path.stem), defects)
        df.to_csv(out_dir / csv_path.name, index=False)
        count += 1
    return count


def _inner_segment(a: np.ndarray, b: np.ndarray, extra: float, chained: bool = False):
    """End points of the middle ``1/3 + extra`` px of segment a-b.

    With ``chained`` the second point is measured from the first one instead of from ``a``
    (reproduces how the offsets below were originally tuned).
    """
    length = np.linalg.norm(b - a)
    center = length / 3 + extra
    t1 = (length - center) / 2 / length
    t2 = t1 + center / length
    p = a + t1 * (b - a)
    origin = p if chained else a
    return p, origin + t2 * (b - origin)


# Pixel offsets hand-tuned on C05 images.
_DIAGONAL_EXTRA = 20
_SIDE_EXTRA = 17
_X_SHIFT = np.array([2.0, 0.0])
_Y_SHIFT_LEFT = np.array([0.0, -4.0])
_Y_SHIFT_RIGHT = np.array([0.0, -2.0])


def number_box(quad: np.ndarray) -> np.ndarray:
    """Shrink a capacitor cell (TL, TR, BR, BL) to the region holding its printed number."""
    tl, tr, br, bl = quad
    top_a, top_b = (p + _X_SHIFT for p in _inner_segment(tl, tr, _DIAGONAL_EXTRA))
    bottom_a, bottom_b = (p + _X_SHIFT for p in _inner_segment(br, bl, _DIAGONAL_EXTRA))
    new_tl, new_bl = _inner_segment(top_a, bottom_b, _SIDE_EXTRA, chained=True)
    new_br, new_tr = _inner_segment(bottom_a, top_b, _SIDE_EXTRA, chained=True)
    return np.stack([new_tl + _Y_SHIFT_LEFT, new_tr + _Y_SHIFT_RIGHT, new_br + _Y_SHIFT_RIGHT, new_bl + _Y_SHIFT_LEFT])


def make_number_boxes(src_dir: Path, dst_dir: Path, types: list[str] | None = None) -> int:
    """Write number-box CSVs (plus a copy of the image) for each cell CSV in ``src_dir/<type>/``."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    count = 0
    for csv_path in sorted(src_dir.rglob("*.csv")):
        rel = csv_path.relative_to(src_dir)
        if types and rel.parts[0] not in types:
            continue
        quads = read_quads(pd.read_csv(csv_path))
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        quads_to_frame(np.stack([number_box(q) for q in quads])).to_csv(out, index=False)
        image = csv_path.with_suffix(".bmp")
        if image.exists():
            shutil.copy(image, out.with_suffix(".bmp"))
        count += 1
    return count


def _labels(df: pd.DataFrame, default: str) -> list[str]:
    if "Defect" not in df:
        return [default] * len(df)
    return [NOT_AVAILABLE if pd.isna(v) or str(v) == "True" else str(v) for v in df["Defect"]]


def cells_to_labelme(
    csv_dir: Path,
    out_dir: Path,
    label: str = NUMB_BOX,
    image_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
) -> int:
    """Convert every cell CSV under ``csv_dir`` to a labelme JSON in ``out_dir`` (same layout).

    Labels come from the ``Defect`` column when present, otherwise ``label`` is used.
    """
    csv_dir, out_dir = Path(csv_dir), Path(out_dir)
    width, height = image_size
    count = 0
    for csv_path in sorted(csv_dir.rglob("*.csv")):
        df = pd.read_csv(csv_path)
        shapes = [polygon_shape(lbl, quad) for lbl, quad in zip(_labels(df, label), read_quads(df))]
        out = (out_dir / csv_path.relative_to(csv_dir)).with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_labelme(out, labelme_document(f"{csv_path.stem}.bmp", height, width, shapes))
        count += 1
    return count
