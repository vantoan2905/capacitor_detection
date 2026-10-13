import json

import numpy as np
import pandas as pd
import pytest

from capacitor_detection.annotations import read_quads
from capacitor_detection.constants import NOT_AVAILABLE
from capacitor_detection.preprocessing.cells import add_row_col, cells_to_labelme, merge_defects, number_box
from capacitor_detection.preprocessing.grid import build_grid, build_grids, count_cells, grid_to_quads, write_cell_csv

# labelme drawing order: TL, BL, BR, TR
MASTER = np.array([[100, 50], [100, 350], [500, 350], [500, 50]], float)
CELL = np.array([[100, 50], [100, 150], [200, 150], [200, 50]], float)


def test_count_cells():
    assert count_cells(MASTER, CELL) == (3, 4)


def test_grid_quads_are_row_major_tl_tr_br_bl():
    quads = grid_to_quads(build_grid(MASTER, 3, 4))
    assert quads.shape == (12, 4, 2)
    np.testing.assert_allclose(quads[0], [[100, 50], [200, 50], [200, 150], [100, 150]])
    np.testing.assert_allclose(quads[4][0], [100, 150])  # first cell of second row


def test_cell_csv_round_trip_keeps_legacy_layout(tmp_path):
    quads = grid_to_quads(build_grid(MASTER, 3, 4))
    path = tmp_path / "001.csv"
    write_cell_csv(path, quads)
    df = pd.read_csv(path)
    # Data columns are TL, BL, BR, TR under the historical headers.
    assert list(df.columns[:5]) == ["Square_ID", "Top_Left_X", "Top_Left_Y", "Bottom_Left_X", "Bottom_Left_Y"]
    assert df.loc[0, "Top_Right_X"] == 200 and df.loc[0, "Top_Right_Y"] == 150  # bottom-right corner
    np.testing.assert_allclose(read_quads(df), quads)


def test_build_grids_from_labelme(tmp_path):
    shapes = [
        {"label": "master_box", "points": MASTER.tolist()},
        {"label": "numb_box", "points": CELL.tolist()},
    ]
    (tmp_path / "C05").mkdir()
    (tmp_path / "C05" / "001.json").write_text(json.dumps({"shapes": shapes}))
    (tmp_path / "C05" / "002.json").write_text(json.dumps({"shapes": shapes[:1]}))  # no example cell
    assert build_grids(tmp_path) == 1
    assert len(pd.read_csv(tmp_path / "C05" / "001.csv")) == 12


def test_add_row_col_and_defects(tmp_path):
    # slightly rotated tray so no x coordinates coincide by accident
    master = MASTER + np.array([[0, 0], [3, 0], [3, 2], [0, 2]])
    path = tmp_path / "007.csv"
    write_cell_csv(path, grid_to_quads(build_grid(master, 3, 4)))
    df = add_row_col(pd.read_csv(path))
    assert df["Col"].tolist() == [1, 2, 3, 4] * 3
    assert df["Row"].tolist() == [1] * 4 + [2] * 4 + [3] * 4

    defects = pd.DataFrame({"id": [7, 7, 8], "col": [2, 4, 2], "row": [1, 3, 1], "label": ["crack", "chip", "x"]})
    labels = merge_defects(df, 7, defects)["Defect"]
    assert labels[1] == "crack" and labels[11] == "chip"
    assert (labels.drop([1, 11]) == NOT_AVAILABLE).all()


def test_number_box_is_inside_cell():
    cell = np.array([[0, 0], [90, 0], [90, 90], [0, 90]], float)  # TL, TR, BR, BL
    box = number_box(cell)
    assert box.shape == (4, 2)
    assert (box > 0).all() and (box < 90).all()
    tl, tr, br, bl = box
    assert tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]


def test_cells_to_labelme(tmp_path):
    src = tmp_path / "csv"
    src.mkdir()
    write_cell_csv(src / "001.csv", grid_to_quads(build_grid(MASTER, 3, 4)))
    assert cells_to_labelme(src, tmp_path / "out", label="numb_box", image_size=(640, 480)) == 1
    doc = json.loads((tmp_path / "out" / "001.json").read_text())
    assert (doc["imagePath"], doc["imageWidth"], doc["imageHeight"]) == ("001.bmp", 640, 480)
    assert len(doc["shapes"]) == 12
    assert doc["shapes"][0]["points"] == [[100, 50], [200, 50], [200, 150], [100, 150]]


@pytest.mark.parametrize("value", [True, "True", np.nan])
def test_cells_to_labelme_defect_placeholder(tmp_path, value):
    df = pd.read_csv(_one_cell_csv(tmp_path))
    df["Defect"] = value
    df.to_csv(tmp_path / "001.csv", index=False)
    cells_to_labelme(tmp_path, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "001.json").read_text())["shapes"][0]["label"] == NOT_AVAILABLE


def _one_cell_csv(tmp_path):
    path = tmp_path / "001.csv"
    write_cell_csv(path, grid_to_quads(build_grid(CELL, 1, 1)))
    return path
