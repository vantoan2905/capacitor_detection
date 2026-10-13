"""Split full-resolution images into overlapping tiles (and back).

The camera frames are 5472x3648 px while capacitors are small, so models are trained on tiles.
Two layouts are supported:

* ``fixed``: ~1/3-size tiles with a fixed pixel overlap, named ``<stem>_w<i>_h<j>``.
  :func:`merge_tiles` can stitch these back together (e.g. YOLO prediction images).
* ``ratio``: a 3x3 grid where each tile is extended by a fraction of the tile size on every
  side, named ``<stem>_<k>``. Used for the capacitor detector training set.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from capacitor_detection.annotations import labelme_document, polygon_shape, read_quads, write_labelme
from capacitor_detection.constants import BLACK_BOX, BLACK_BOX_IMAGE_IDS, NUMB_BOX

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tile:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0 : self.y1, self.x0 : self.x1]

    def contains(self, quad: np.ndarray) -> bool:
        xs, ys = quad[:, 0], quad[:, 1]
        return bool(np.all((self.x0 <= xs) & (xs <= self.x1) & (self.y0 <= ys) & (ys <= self.y1)))


def fixed_overlap_tiles(width: int, height: int, overlap: int = 70, splits: int = 3) -> list[Tile]:
    step_x, step_y = int(width / splits), int(height / splits)
    n_w = (width + overlap) // (step_x - overlap)
    n_h = (height + overlap) // (step_y - overlap)
    tiles = []
    for i in range(n_w):
        for j in range(n_h):
            x0, y0 = i * (step_x - overlap), j * (step_y - overlap)
            tiles.append(Tile(f"w{i}_h{j}", x0, y0, min(x0 + step_x, width), min(y0 + step_y, height)))
    return tiles


def ratio_overlap_tiles(width: int, height: int, overlap_ratio: float = 0.1, splits: int = 3) -> list[Tile]:
    part_w, part_h = width // splits, height // splits
    over_x, over_y = int(part_w * overlap_ratio), int(part_h * overlap_ratio)
    tiles = []
    for i in range(splits):
        for j in range(splits):
            tiles.append(
                Tile(
                    str(i * splits + j),
                    max(0, j * part_w - over_x),
                    max(0, i * part_h - over_y),
                    min(width, (j + 1) * part_w + over_x),
                    min(height, (i + 1) * part_h + over_y),
                )
            )
    return tiles


def make_tiles(width: int, height: int, mode: str, overlap: int = 70, overlap_ratio: float = 0.1) -> list[Tile]:
    if mode == "fixed":
        return fixed_overlap_tiles(width, height, overlap)
    if mode == "ratio":
        return ratio_overlap_tiles(width, height, overlap_ratio)
    raise ValueError(f"unknown tiling mode: {mode!r}")


def tile_dataset(
    src_dir: Path,
    dst_dir: Path,
    mode: str = "ratio",
    overlap: int = 70,
    overlap_ratio: float = 0.1,
    black_box_ids: frozenset[str] = BLACK_BOX_IMAGE_IDS,
) -> int:
    """Tile every ``*.bmp`` with a sibling cell CSV under ``src_dir`` into ``dst_dir``.

    For each tile an image and a labelme JSON (cells fully inside the tile, shifted to tile
    coordinates) are written. Returns the number of source images processed.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(Path(src_dir).rglob("*.bmp")):
        csv_path = image_path.with_suffix(".csv")
        if not csv_path.exists():
            log.warning("skip %s: no cell CSV", image_path)
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            log.warning("skip %s: cannot read image", image_path)
            continue
        quads = read_quads(pd.read_csv(csv_path))
        label = BLACK_BOX if image_path.stem in black_box_ids else NUMB_BOX
        height, width = image.shape[:2]
        for tile in make_tiles(width, height, mode, overlap, overlap_ratio):
            name = f"{image_path.stem}_{tile.name}"
            cv2.imwrite(str(dst_dir / f"{name}.bmp"), tile.crop(image))
            offset = np.array([tile.x0, tile.y0])
            shapes = [polygon_shape(label, q - offset) for q in quads if tile.contains(q)]
            write_labelme(dst_dir / f"{name}.json", labelme_document(f"{name}.bmp", tile.height, tile.width, shapes))
        count += 1
    return count


_FIXED_TILE_NAME = re.compile(r"^(?P<prefix>[^_]+)_(?P<tile>w\d+_h\d+)$")


def merge_tiles(tile_dir: Path, out_dir: Path, image_size: tuple[int, int], overlap: int = 70) -> int:
    """Stitch ``fixed``-mode tiles (e.g. YOLO prediction images) back into full images."""
    width, height = image_size
    tiles = {t.name: t for t in fixed_overlap_tiles(width, height, overlap)}
    groups: dict[str, list[tuple[Tile, Path]]] = defaultdict(list)
    for path in sorted(Path(tile_dir).glob("*.bmp")):
        match = _FIXED_TILE_NAME.match(path.stem)
        if match and match["tile"] in tiles:
            groups[match["prefix"]].append((tiles[match["tile"]], path))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for prefix, parts in groups.items():
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for tile, path in parts:
            canvas[tile.y0 : tile.y1, tile.x0 : tile.x1] = cv2.imread(str(path))[: tile.height, : tile.width]
        cv2.imwrite(str(out_dir / f"{prefix}_reconstructed.bmp"), canvas)
    return len(groups)
