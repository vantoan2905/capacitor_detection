import json

import cv2
import numpy as np

from capacitor_detection.inference import merge_tile_detections, nms_indices, non_max_suppression
from capacitor_detection.preprocessing.grid import build_grid, grid_to_quads, write_cell_csv
from capacitor_detection.tiling import (
    fixed_overlap_tiles,
    merge_tiles,
    ratio_overlap_tiles,
    tile_dataset,
)

WIDTH, HEIGHT = 5472, 3648


def test_ratio_tiles_cover_image():
    covered = np.zeros((HEIGHT, WIDTH), bool)
    for t in ratio_overlap_tiles(WIDTH, HEIGHT):
        covered[t.y0 : t.y1, t.x0 : t.x1] = True
    assert covered.all()


def test_fixed_tiles_match_legacy_layout():
    tiles = {t.name: t for t in fixed_overlap_tiles(WIDTH, HEIGHT)}
    assert len(tiles) == 9
    assert (tiles["w1_h2"].x0, tiles["w1_h2"].y0) == (1754, 2292)
    assert {(t.width, t.height) for t in tiles.values()} == {(1824, 1216)}
    # Known limitation of this layout: the last 140 px on the right/bottom are never tiled.
    assert max(t.x1 for t in tiles.values()) == WIDTH - 140


def test_ratio_tiles_sizes():
    tiles = ratio_overlap_tiles(WIDTH, HEIGHT)
    assert (tiles[0].width, tiles[0].height) == (2006, 1337)
    assert (tiles[4].width, tiles[4].height) == (2188, 1458)


def test_merge_tiles_restores_image(tmp_path):
    width, height = 600, 300
    image = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    for t in fixed_overlap_tiles(width, height, overlap=10):
        cv2.imwrite(str(tmp_path / f"042_{t.name}.bmp"), t.crop(image))
    assert merge_tiles(tmp_path, tmp_path / "out", (width, height), overlap=10) == 1
    tiles = fixed_overlap_tiles(width, height, overlap=10)
    x_end, y_end = max(t.x1 for t in tiles), max(t.y1 for t in tiles)
    merged = cv2.imread(str(tmp_path / "out" / "042_reconstructed.bmp"))
    np.testing.assert_array_equal(merged[:y_end, :x_end], image[:y_end, :x_end])


def test_tile_dataset_shifts_boxes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    cv2.imwrite(str(src / "053.bmp"), np.zeros((300, 600, 3), np.uint8))
    master = np.array([[10, 10], [10, 70], [70, 70], [70, 10]], float)  # TL, BL, BR, TR
    write_cell_csv(src / "053.csv", grid_to_quads(build_grid(master, 2, 2)))

    assert tile_dataset(src, tmp_path / "dst", mode="ratio") == 1
    doc = json.loads((tmp_path / "dst" / "053_0.json").read_text())
    assert (doc["imageWidth"], doc["imageHeight"]) == (220, 110)
    assert [s["label"] for s in doc["shapes"]] == ["black_box"] * 4
    assert doc["shapes"][0]["points"] == [[10, 10], [40, 10], [40, 40], [10, 40]]
    # tile 1 starts at x=180, so no cell fits
    assert json.loads((tmp_path / "dst" / "053_1.json").read_text())["shapes"] == []


def test_nms_keeps_best_of_overlapping():
    boxes = [[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.7]]
    np.testing.assert_allclose(non_max_suppression(boxes)[:, 4], [0.9, 0.7])


def test_merge_tile_detections_dedupes_overlap():
    tiles = ratio_overlap_tiles(600, 300)
    # the same object seen by tiles 0 and 1 in their own coordinates
    detections = [(tiles[0], [[190, 10, 210, 30, 0.9]]), (tiles[1], [[10, 10, 30, 30, 0.6]])]
    merged = merge_tile_detections(detections, 600, 300)
    np.testing.assert_allclose(merged, [[190, 10, 210, 30, 0.9]])


def test_nms_indices_per_class():
    boxes = [[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [0, 0, 10, 10, 0.7]]
    assert nms_indices(boxes).tolist() == [0]
    assert nms_indices(boxes, classes=[0, 0, 1]).tolist() == [0, 2]
    assert nms_indices([], classes=[]).tolist() == []
