# data/

Everything in this folder except this file is git-ignored. Suggested layout:

```
data/
├── archives/     # original archives: final_label_2.rar, duplicate_.rar,
│                 # ann_defect_json.zip (master_box labels), processed_defect_data_updated.xlsx
├── raw/          # camera images grouped by batch, e.g. raw/24.09/C05 VS-W/309.bmp
├── interim/      # sorted-by-type images, cell CSVs, tiles, labelme JSON
└── processed/    # YOLO datasets referenced by configs/*.yaml
    ├── region_detector/YOLOv8Dataset/dataset.yaml
    └── capacitor_detector/YOLODataset/dataset.yaml
```
