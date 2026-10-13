"""Command line entry point: ``capdet <command> ...``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from capacitor_detection.constants import BLACK_BOX_IMAGE_IDS, CAPACITOR_TYPES, IMAGE_HEIGHT, IMAGE_WIDTH, NUMB_BOX


def _size(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


def _sort(args):
    from capacitor_detection.preprocessing.sort import sort_by_type

    return f"copied {sort_by_type(args.src, args.dst, tuple(args.types))} files"


def _grid(args):
    from capacitor_detection.preprocessing.grid import build_grids

    return f"wrote {build_grids(args.ann_dir, args.average_per_folder)} cell CSVs"


def _defects(args):
    from capacitor_detection.preprocessing.cells import annotate_cells

    return f"wrote {annotate_cells(args.csv_dir, args.out_dir, args.defects)} CSVs"


def _number_boxes(args):
    from capacitor_detection.preprocessing.cells import make_number_boxes

    return f"wrote {make_number_boxes(args.src, args.dst, args.types)} CSVs"


def _to_labelme(args):
    from capacitor_detection.preprocessing.cells import cells_to_labelme

    return f"wrote {cells_to_labelme(args.csv_dir, args.out_dir, args.label, args.image_size)} JSON files"


def _tile(args):
    from capacitor_detection.tiling import tile_dataset

    ids = frozenset(args.black_box_ids) if args.black_box_ids is not None else BLACK_BOX_IMAGE_IDS
    n = tile_dataset(args.src, args.dst, args.mode, args.overlap, args.overlap_ratio, ids)
    return f"tiled {n} images"


def _merge(args):
    from capacitor_detection.tiling import merge_tiles

    return f"merged {merge_tiles(args.tile_dir, args.out_dir, args.image_size, args.overlap)} images"


def _auto_label(args):
    from capacitor_detection.auto_label import auto_label

    return f"found {auto_label(args.image, args.label, args.template)} boxes"


def _train(args):
    from capacitor_detection.models import yolo

    overrides = {k: v for k, v in vars(args).items() if k in ("model", "data", "epochs", "batch", "device") and v}
    yolo.train(args.config, **overrides)


def _predict(args):
    from capacitor_detection.models import yolo

    yolo.predict(args.weights, args.source, args.conf, args.iou, args.save_txt)


def _frcnn_split(args):
    from capacitor_detection.models import faster_rcnn

    return str(faster_rcnn.split_dataset(args.src, args.dst, args.image_ext))


def _frcnn_train(args):
    from capacitor_detection.models import faster_rcnn

    faster_rcnn.train(args.data_dir, args.out, args.num_classes, args.epochs, args.batch_size, image_ext=args.image_ext)


def _frcnn_predict(args):
    from capacitor_detection.models import faster_rcnn

    faster_rcnn.predict(args.weights, args.image_dir, args.out_dir, args.num_classes, args.threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capdet", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help):
        p = sub.add_parser(name, help=help, description=help)
        p.set_defaults(func=func)
        return p

    p = add("sort", _sort, "Copy raw images into one folder per capacitor type")
    p.add_argument("src", type=Path, help="folder of batches, each containing '<TYPE ...>' folders")
    p.add_argument("dst", type=Path)
    p.add_argument("--types", nargs="+", default=list(CAPACITOR_TYPES))

    p = add("grid", _grid, "Write a cell CSV next to each labelme JSON (master_box + one example cell)")
    p.add_argument("ann_dir", type=Path)
    p.add_argument("--average-per-folder", action="store_true", help="average row/col counts per folder")

    p = add("defects", _defects, "Add Row/Col (and Defect labels) to cell CSVs")
    p.add_argument("csv_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--defects", type=Path, help="xlsx with columns: image_id, col, row, label")

    p = add("number-boxes", _number_boxes, "Shrink cells to the printed-number region")
    p.add_argument("src", type=Path, help="output of 'sort' (one folder per type)")
    p.add_argument("dst", type=Path)
    p.add_argument("--types", nargs="+", help="only these type folders (offsets were tuned on C05)")

    p = add("to-labelme", _to_labelme, "Convert cell CSVs to labelme JSON")
    p.add_argument("csv_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--label", default=NUMB_BOX, help="label when the CSV has no Defect column")
    p.add_argument("--image-size", type=_size, default=(IMAGE_WIDTH, IMAGE_HEIGHT), metavar="WxH")

    p = add("tile", _tile, "Cut images + cell CSVs into labelled tiles")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--mode", choices=("ratio", "fixed"), default="ratio")
    p.add_argument("--overlap", type=int, default=70, help="pixels, fixed mode")
    p.add_argument("--overlap-ratio", type=float, default=0.1, help="fraction of tile size, ratio mode")
    p.add_argument("--black-box-ids", nargs="*", help="image ids labelled black_box (default: built-in list)")

    p = add("merge-tiles", _merge, "Stitch fixed-mode tiles back into full images")
    p.add_argument("tile_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--image-size", type=_size, default=(IMAGE_WIDTH, IMAGE_HEIGHT), metavar="WxH")
    p.add_argument("--overlap", type=int, default=70)

    p = add("auto-label", _auto_label, "Heuristically pre-label one image (labelme rectangles)")
    p.add_argument("image", type=Path)
    p.add_argument("--label", default="resitor")
    p.add_argument("--template", type=Path, help="labelme JSON to copy top-level fields from")

    p = add("train", _train, "Train a YOLO model from a config in configs/")
    p.add_argument("config", type=Path)
    p.add_argument("--model")
    p.add_argument("--data")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--device")

    p = add("predict", _predict, "Run a trained YOLO model on images")
    p.add_argument("weights", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--save-txt", action="store_true")

    p = add("frcnn-split", _frcnn_split, "Split a VOC dataset into train/val/test (50/30/20)")
    p.add_argument("src", type=Path, help="folder with annotations/ and images/")
    p.add_argument("dst", type=Path)
    p.add_argument("--image-ext", default=".png")

    p = add("frcnn-train", _frcnn_train, "Train the Faster R-CNN baseline")
    p.add_argument("data_dir", type=Path, help="output of frcnn-split")
    p.add_argument("--out", type=Path, default=Path("runs/faster_rcnn/model.pth"))
    p.add_argument("--num-classes", type=int, default=2, help="including background")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-ext", default=".png")

    p = add("frcnn-predict", _frcnn_predict, "Draw Faster R-CNN detections on images")
    p.add_argument("weights", type=Path)
    p.add_argument("image_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    message = args.func(args)
    if message:
        print(message)


if __name__ == "__main__":
    main()
