"""Generate per-capacitor cells from a ``master_box`` and one example cell.

Each labelme JSON is expected to contain one ``master_box`` (the whole tray) and at least one
``numb_box``/``black_box`` (a single capacitor). The number of rows/columns is estimated from the
ratio of their side lengths, then the master box is subdivided into a regular grid and written
as a cell CSV next to the JSON.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from capacitor_detection.annotations import load_shapes, quads_to_frame
from capacitor_detection.constants import BLACK_BOX, MASTER_BOX, NUMB_BOX

log = logging.getLogger(__name__)


def _round_half_down(value: float) -> int:
    return int(value) + 1 if value - int(value) > 0.5 else int(value)


def _side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    """Left, right, bottom, top side lengths of a quad drawn as TL, BL, BR, TR."""
    tl, bl, br, tr = quad
    dist = np.linalg.norm
    return dist(bl - tl), dist(tr - br), dist(br - bl), dist(tr - tl)


def count_cells(master: np.ndarray, cell: np.ndarray) -> tuple[int, int]:
    """Estimate ``(rows, cols)`` of ``cell``-sized boxes that fit in ``master``.

    Both quads are ``(4, 2)`` arrays in labelme drawing order TL, BL, BR, TR.
    """
    m_left, m_right, m_bottom, m_top = _side_lengths(master)
    c_left, c_right, c_bottom, c_top = _side_lengths(cell)
    rows = (m_left / c_left + m_right / c_right) / 2
    cols = (m_bottom / c_bottom + m_top / c_top) / 2
    return _round_half_down(rows), _round_half_down(cols)


def _interpolate(p1: np.ndarray, p2: np.ndarray, steps: int) -> np.ndarray:
    t = np.arange(steps + 1)[:, None] / steps
    return p1 + (p2 - p1) * t


def build_grid(master: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Subdivide ``master`` (TL, BL, BR, TR) into a ``(rows + 1, cols + 1, 2)`` grid of points."""
    tl, bl, br, tr = master
    left = _interpolate(tl, bl, rows)
    right = _interpolate(tr, br, rows)
    return np.stack([_interpolate(left[i], right[i], cols) for i in range(rows + 1)])


def grid_to_quads(grid: np.ndarray) -> np.ndarray:
    """Row-major ``(rows * cols, 4, 2)`` cells ordered TL, TR, BR, BL."""
    tl = grid[:-1, :-1]
    tr = grid[:-1, 1:]
    br = grid[1:, 1:]
    bl = grid[1:, :-1]
    return np.stack([tl, tr, br, bl], axis=2).reshape(-1, 4, 2)


def write_cell_csv(path: Path, quads: np.ndarray) -> None:
    df = quads_to_frame(quads)
    df.insert(0, "Square_ID", np.arange(1, len(df) + 1))
    df.to_csv(path, index=False)


def _first_quad(shapes: list[dict], labels: set[str]) -> np.ndarray | None:
    for shape in shapes:
        if shape.get("label") in labels:
            return np.asarray(shape["points"][:4], dtype=float)
    return None


def build_grids(ann_dir: Path, average_per_folder: bool = False) -> int:
    """Write a cell CSV next to every labelme JSON under ``ann_dir``. Returns the number written.

    With ``average_per_folder`` the row/column counts are averaged over all images in the same
    folder (one folder per capacitor type), which smooths out sloppy example cells.
    """
    found: dict[Path, tuple[np.ndarray, int, int]] = {}
    for json_path in sorted(Path(ann_dir).rglob("*.json")):
        shapes = load_shapes(json_path)
        master = _first_quad(shapes, {MASTER_BOX})
        cell = _first_quad(shapes, {BLACK_BOX, NUMB_BOX})
        if master is None or cell is None:
            log.warning("skip %s: needs a %s and a %s/%s shape", json_path, MASTER_BOX, NUMB_BOX, BLACK_BOX)
            continue
        found[json_path] = (master, *count_cells(master, cell))

    if average_per_folder:
        per_folder = defaultdict(list)
        for json_path, (_, rows, cols) in found.items():
            per_folder[json_path.parent].append((rows, cols))
        means = {folder: np.mean(counts, axis=0) for folder, counts in per_folder.items()}
        found = {p: (master, int(means[p.parent][0]), int(means[p.parent][1])) for p, (master, _, _) in found.items()}

    for json_path, (master, rows, cols) in found.items():
        log.info("%s: %d x %d cells", json_path, rows, cols)
        write_cell_csv(json_path.with_suffix(".csv"), grid_to_quads(build_grid(master, rows, cols)))
    return len(found)
