"""DeepStream pipeline: tiled capacitor detection on full-resolution tray images.

    appsrc (images) -> nvvideoconvert -> nvstreammux -> nvdspreprocess (9 tile ROIs)
        -> nvinfer (YOLOv8) -> probe: cross-tile NMS + JSON lines output -> fakesink

Usage (on the DeepStream machine, after setup.sh)::

    python3 deploy/deepstream/app.py data/interim/test_images --out runs/deepstream/detections.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import cv2
import gi
import numpy as np
import pyds

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from capacitor_detection.inference import nms_indices

CONFIGS = Path(__file__).parent / "configs"
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg"}


def build_pipeline(width: int, height: int) -> Gst.Pipeline:
    caps = f"video/x-raw,format=RGBA,width={width},height={height},framerate=1/1"
    return Gst.parse_launch(
        f"appsrc name=src format=time block=true max-bytes={2 * width * height * 4} caps={caps} "
        "! nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA "
        f"! mux.sink_0 nvstreammux name=mux batch-size=1 width={width} height={height} batched-push-timeout=4000000 "
        f"! nvdspreprocess config-file={CONFIGS / 'config_preprocess.txt'} "
        f"! nvinfer name=pgie config-file-path={CONFIGS / 'config_infer_capacitor.txt'} "
        "! fakesink sync=false"
    )


def feed(appsrc, images: list[Path], width: int, height: int) -> None:
    for i, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (height, width):
            print(f"skip {path}: unreadable or not {width}x{height}", file=sys.stderr)
            frame = np.zeros((height, width, 3), np.uint8)  # keep frame numbering aligned with `images`
        buf = Gst.Buffer.new_wrapped(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).tobytes())
        buf.pts = i * Gst.SECOND
        buf.duration = Gst.SECOND
        if appsrc.emit("push-buffer", buf) != Gst.FlowReturn.OK:
            break
    appsrc.emit("end-of-stream")


def _objects(frame_meta):
    item = frame_meta.obj_meta_list
    while item is not None:
        yield pyds.NvDsObjectMeta.cast(item.data)
        item = item.next


def make_probe(images: list[Path], labels: list[str], out, iou: float):
    """Remove duplicates from overlapping tiles and write one JSON line per image."""

    def probe(pad, info):
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        item = batch_meta.frame_meta_list
        while item is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(item.data)
            objs = list(_objects(frame_meta))
            boxes = [
                [o.rect_params.left, o.rect_params.top, o.rect_params.left + o.rect_params.width,
                 o.rect_params.top + o.rect_params.height, o.confidence]
                for o in objs
            ]  # fmt: skip
            keep = set(nms_indices(boxes, iou, classes=[o.class_id for o in objs]).tolist())
            detections = []
            for i, obj in enumerate(objs):
                if i not in keep:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj)
                    continue
                x1, y1, x2, y2, score = boxes[i]
                detections.append(
                    {
                        "label": labels[obj.class_id],
                        "score": round(score, 4),
                        "box": [round(v, 1) for v in (x1, y1, x2, y2)],
                    }
                )
            out.write(json.dumps({"image": str(images[frame_meta.frame_num]), "detections": detections}) + "\n")
            out.flush()
            item = item.next
        return Gst.PadProbeReturn.OK

    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/deepstream/detections.jsonl"))
    parser.add_argument("--image-size", default="5472x3648", metavar="WxH")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU for merging detections across tiles")
    args = parser.parse_args()

    width, height = map(int, args.image_size.lower().split("x"))
    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        sys.exit(f"no images in {args.image_dir}")
    labels = (Path(__file__).parent / "models" / "labels.txt").read_text().split()

    Gst.init(None)
    pipeline = build_pipeline(width, height)
    loop = GLib.MainLoop()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w") as out:
        pgie_src = pipeline.get_by_name("pgie").get_static_pad("src")
        pgie_src.add_probe(Gst.PadProbeType.BUFFER, make_probe(images, labels, out, args.iou))

        def on_message(bus, message):
            if message.type == Gst.MessageType.EOS:
                loop.quit()
            elif message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"error: {err}: {debug}", file=sys.stderr)
                loop.quit()

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_message)

        pipeline.set_state(Gst.State.PLAYING)
        feeder = threading.Thread(target=feed, args=(pipeline.get_by_name("src"), images, width, height), daemon=True)
        feeder.start()
        try:
            loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)
    print(f"{len(images)} images -> {args.out}")


if __name__ == "__main__":
    main()
