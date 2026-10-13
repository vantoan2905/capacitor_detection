"""Write the tiling ROIs into the DeepStream configs so they match training.

python deploy/deepstream/gen_rois.py --image-size 5472x3648 --overlap-ratio 0.1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from capacitor_detection.tiling import ratio_overlap_tiles

CONFIGS = Path(__file__).parent / "configs"


def _set(text: str, key: str, value: str) -> str:
    text, n = re.subn(rf"^{re.escape(key)}=.*$", f"{key}={value}", text, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"expected exactly one '{key}=' line, found {n}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-size", default="5472x3648", metavar="WxH")
    parser.add_argument("--overlap-ratio", type=float, default=0.1)
    args = parser.parse_args()

    width, height = map(int, args.image_size.lower().split("x"))
    tiles = ratio_overlap_tiles(width, height, args.overlap_ratio)
    rois = "".join(f"{t.x0};{t.y0};{t.width};{t.height};" for t in tiles)

    preprocess = CONFIGS / "config_preprocess.txt"
    text = _set(preprocess.read_text(), "roi-params-src-0", rois)
    shape = re.search(r"^network-input-shape=\d+;(.*)$", text, flags=re.MULTILINE)[1]
    preprocess.write_text(_set(text, "network-input-shape", f"{len(tiles)};{shape}"))

    infer = CONFIGS / "config_infer_capacitor.txt"
    infer.write_text(_set(infer.read_text(), "batch-size", str(len(tiles))))
    print(f"{len(tiles)} ROIs for {width}x{height}: {rois}")


if __name__ == "__main__":
    main()
