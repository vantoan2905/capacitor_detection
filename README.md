# Capacitor Detection

Detect capacitors on tray images (5472x3648 px camera frames) with two YOLOv8 models:

1. **Region detector** (`configs/region_detector.yaml`): oriented box (OBB) around the whole tray (`master_box`).
2. **Capacitor detector** (`configs/capacitor_detector.yaml`): `numb_box` / `black_box` per capacitor, trained on
   overlapping image tiles because the capacitors are small.

A torchvision Faster R-CNN baseline is included as well.

## Project layout

```
.
├── configs/                  # Ultralytics training configs (+ models/ architecture YAMLs)
├── data/                     # local only, git-ignored (see data/README.md)
├── notebooks/legacy/         # original exploration notebooks, kept for reference
├── src/capacitor_detection/
│   ├── annotations.py        # labelme JSON + cell CSV I/O
│   ├── constants.py          # labels, capacitor types, black-box image ids
│   ├── preprocessing/
│   │   ├── sort.py           # group raw images by capacitor type
│   │   ├── grid.py           # master_box + example cell -> grid of cells
│   │   └── cells.py          # row/col, defect labels, number boxes, labelme export
│   ├── tiling.py             # split images into tiles / stitch them back
│   ├── inference.py          # tile detections -> image coordinates + NMS
│   ├── auto_label.py         # OpenCV heuristic pre-labelling
│   ├── models/yolo.py        # train / predict
│   ├── models/faster_rcnn.py # VOC dataset, train / predict
│   └── cli.py                # `capdet` command
└── tests/
```

## Installation

```sh
git clone https://github.com/vantoan2905/capacitor_detection.git
cd capacitor_detection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[yolo,dev]"      # add ",frcnn" for the Faster R-CNN baseline
```

## Pipeline

Run `capdet <command> --help` for all options.

### Capacitor detector data

```sh
# 1. one folder per capacitor type (C05, C10, W18, ...)
capdet sort data/raw data/interim/by_type

# 2. annotate one master_box + one numb_box/black_box per image in labelme, then generate all cells
capdet grid data/interim/by_type --average-per-folder

# 3. cut images into labelled tiles (labelme JSON per tile)
capdet tile data/interim/by_type data/interim/tiles --mode ratio

# 4. convert to a YOLO dataset
labelme2yolo --json_dir data/interim/tiles
mv data/interim/tiles/YOLODataset data/processed/capacitor_detector/
```

Optional steps on the cell CSVs:

```sh
capdet defects data/interim/by_type data/interim/cells --defects data/archives/processed_defect_data_updated.xlsx
capdet to-labelme data/interim/cells data/interim/cells_labelme   # labels from the Defect column
capdet number-boxes data/interim/by_type data/interim/number_boxes --types C05
```

### Region detector data

`data/archives/ann_defect_json.zip` holds the `master_box` labelme annotations:

```sh
python -m labelme2yolov8 --json_dir <images+json dir> --output_format obb --val_size 0.15 --test_size 0.15
# place the result under data/processed/region_detector/YOLOv8Dataset/
```

### Training and inference

```sh
capdet train configs/region_detector.yaml
capdet train configs/capacitor_detector.yaml --epochs 30 --device 0

capdet predict runs/capacitor_detector/train/weights/best.pt data/interim/test_tiles --conf 0.5 --iou 0.3 --save-txt
capdet merge-tiles runs/detect/predict data/interim/merged      # only for --mode fixed tiles
```

### Faster R-CNN baseline (Pascal VOC data)

```sh
capdet frcnn-split data/voc data/voc_split          # 50/30/20 train/val/test
capdet frcnn-train data/voc_split --epochs 10
capdet frcnn-predict runs/faster_rcnn/model.pth data/voc_split/test/images runs/faster_rcnn/predict
```

## Notes

- Cell CSVs keep a historical quirk: the `Top_Right_*` / `Bottom_Right_*` headers are swapped relative to the data.
  Existing files on disk use this layout; `annotations.read_quads` handles it.
- `fixed` tiling does not cover the last ~140 px on the right/bottom of a frame; `ratio` tiling covers everything.

## Development

```sh
pytest
ruff check src tests
```
