# Capacitor Detection

Detect capacitors on tray images taken by an industrial camera (5472×3648 px frames).

The capacitors are small compared with the frame, so detection runs in two stages:

| Stage | Model | Config | Output |
| --- | --- | --- | --- |
| 1. Region detector | YOLOv8s-OBB | [configs/region_detector.yaml](configs/region_detector.yaml) | oriented box around the whole tray (`master_box`) |
| 2. Capacitor detector | YOLOv8x | [configs/capacitor_detector.yaml](configs/capacitor_detector.yaml) | one box per capacitor: `numb_box` (printed number) or `black_box` (black capacitor) |

The capacitor detector is trained and run on a 3×3 grid of overlapping tiles. Detections from the tiles are mapped
back to frame coordinates and duplicates are removed with NMS. A torchvision Faster R-CNN baseline is included for
comparison, and the capacitor detector can be deployed with NVIDIA DeepStream / TensorRT.

```
raw frames ─► sort by type ─► labelme (1 master_box + 1 example cell) ─► grid of cells ─► tiles ─► YOLO dataset ─► train
                                                                                                                    │
full frame ─► 9 overlapping tiles ─► YOLOv8 ─► tile → frame coordinates ─► cross-tile NMS ─► detections  ◄──────────┘
```

## Quick start

Requires Python ≥ 3.10.

```sh
git clone https://github.com/vantoan2905/capacitor_detection.git
cd capacitor_detection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[yolo,dev]"      # add ",frcnn" for the Faster R-CNN baseline
                                  # or: pip install -r requirements.txt (all extras)
capdet --help
```

| Extra | Installs | Needed for |
| --- | --- | --- |
| `yolo` | `ultralytics`, `labelme2yolo` | `train`, `predict`, building YOLO datasets |
| `frcnn` | `torch`, `torchvision`, `pillow` | `frcnn-*` commands |
| `dev` | `pytest`, `ruff` | tests and linting |

Data is not in the repository. Put it under `data/` following [data/README.md](data/README.md).

## Pipeline

Run `capdet <command> --help` for all options of a command.

### 1. Prepare the capacitor detector dataset

```sh
# Copy raw images into one folder per capacitor type (C05, C10, W18, Y18, Z10, Z18, Z33)
capdet sort data/raw data/interim/by_type

# In labelme, draw one master_box and one example numb_box/black_box cell per image,
# then generate every cell of the grid (writes a cell CSV next to each JSON)
capdet grid data/interim/by_type --average-per-folder

# Cut images and cells into labelled tiles (one labelme JSON per tile)
capdet tile data/interim/by_type data/interim/tiles --mode ratio

# Convert the tiles to a YOLO dataset
labelme2yolo --json_dir data/interim/tiles
mv data/interim/tiles/YOLODataset data/processed/capacitor_detector/
```

Optional steps on the cell CSVs:

```sh
# Add Row/Col and defect labels from the defect spreadsheet
capdet defects data/interim/by_type data/interim/cells --defects data/archives/processed_defect_data_updated.xlsx

# Export cells to labelme JSON (label taken from the Defect column)
capdet to-labelme data/interim/cells data/interim/cells_labelme

# Shrink cells to the printed-number region (offsets were tuned on C05)
capdet number-boxes data/interim/by_type data/interim/number_boxes --types C05

# Heuristic OpenCV pre-labelling of a single image, to speed up manual annotation
capdet auto-label path/to/image.bmp --template path/to/existing.json
```

### 2. Prepare the region detector dataset

The `master_box` labelme annotations are in `data/archives/ann_defect_json.zip`:

```sh
python -m labelme2yolov8 --json_dir <images+json dir> --output_format obb --val_size 0.15 --test_size 0.15
# move the result to data/processed/region_detector/YOLOv8Dataset/
```

### 3. Train

```sh
capdet train configs/region_detector.yaml
capdet train configs/capacitor_detector.yaml --epochs 30 --device 0
```

`--model`, `--data`, `--epochs`, `--batch` and `--device` override the values in the config file.
Results are written to `runs/<detector>/train/`.

### 4. Predict

```sh
capdet predict runs/capacitor_detector/train/weights/best.pt data/interim/test_tiles --conf 0.5 --iou 0.3 --save-txt

# Stitch prediction tiles back into full frames (only for tiles made with --mode fixed)
capdet merge-tiles runs/detect/predict data/interim/merged
```

### 5. Deploy with DeepStream

[deploy/deepstream/](deploy/deepstream/README.md) runs the capacitor detector on full-resolution frames with
NVIDIA DeepStream / TensorRT: the 9 tiles are inferred as one batch and merged with cross-tile NMS.

```sh
CUDA_VER=12.6 deploy/deepstream/setup.sh runs/capacitor_detector/train/weights/best.pt 1344
python3 deploy/deepstream/app.py data/interim/test_images --out runs/deepstream/detections.jsonl
```

See [deploy/deepstream/README.md](deploy/deepstream/README.md) for requirements and what must stay in sync with
training.

### Faster R-CNN baseline

Uses a Pascal VOC dataset (`annotations/` + `images/`):

```sh
capdet frcnn-split data/voc data/voc_split          # 50/30/20 train/val/test
capdet frcnn-train data/voc_split --epochs 10
capdet frcnn-predict runs/faster_rcnn/model.pth data/voc_split/test/images runs/faster_rcnn/predict
```

## Commands

| Command | Description |
| --- | --- |
| `sort` | Copy raw images into one folder per capacitor type |
| `grid` | Build the grid of cells from a `master_box` and one example cell |
| `defects` | Add Row/Col (and defect labels) to cell CSVs |
| `number-boxes` | Shrink cells to the printed-number region |
| `to-labelme` | Convert cell CSVs to labelme JSON |
| `tile` | Cut images and cell CSVs into labelled tiles (`ratio` or `fixed`) |
| `merge-tiles` | Stitch `fixed` tiles back into full images |
| `auto-label` | Heuristic pre-labelling of one image |
| `train` | Train a YOLO model from a config in `configs/` |
| `predict` | Run a trained YOLO model on images |
| `frcnn-split` / `frcnn-train` / `frcnn-predict` | Faster R-CNN baseline |

## Project layout

```
.
├── configs/                  # Ultralytics training configs (+ models/ architecture YAMLs)
├── data/                     # local only, git-ignored (see data/README.md)
├── deploy/deepstream/        # DeepStream / TensorRT inference pipeline
├── notebooks/legacy/         # original exploration notebooks, kept for reference
├── src/capacitor_detection/
│   ├── cli.py                # `capdet` command
│   ├── constants.py          # labels, capacitor types, frame size, black-box image ids
│   ├── annotations.py        # labelme JSON + cell CSV I/O
│   ├── preprocessing/
│   │   ├── sort.py           # group raw images by capacitor type
│   │   ├── grid.py           # master_box + example cell -> grid of cells
│   │   └── cells.py          # row/col, defect labels, number boxes, labelme export
│   ├── tiling.py             # split images into tiles / stitch them back
│   ├── inference.py          # tile detections -> image coordinates + NMS
│   ├── auto_label.py         # OpenCV heuristic pre-labelling
│   └── models/
│       ├── yolo.py           # train / predict
│       └── faster_rcnn.py    # VOC dataset, train / predict
└── tests/
```

## Notes

- **Tiling modes**: `ratio` (default, used for training) is a 3×3 grid where each tile is extended by
  `--overlap-ratio` of the tile size and covers the whole frame. `fixed` uses a pixel overlap and leaves the last
  ~140 px on the right/bottom uncovered; only `fixed` tiles can be stitched back with `merge-tiles`.
- **Black capacitors**: which images are labelled `black_box` comes from `BLACK_BOX_IMAGE_IDS` in
  [constants.py](src/capacitor_detection/constants.py); override it with `capdet tile --black-box-ids`.
- **Cell CSV quirk**: the `Top_Right_*` / `Bottom_Right_*` headers are swapped relative to the data. Existing files
  use this layout, and `annotations.read_quads` handles it.

## Development

```sh
pytest
ruff check src tests
```
