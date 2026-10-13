#!/usr/bin/env bash
# Build the DeepStream-Yolo bbox parser and export the trained YOLO weights to ONNX.
# Run on the DeepStream machine (inside the DeepStream container or on Jetson):
#
#   CUDA_VER=12.6 deploy/deepstream/setup.sh runs/capacitor_detector/train/weights/best.pt
#
# CUDA_VER must match the CUDA version shipped with your DeepStream release.
set -euo pipefail

WEIGHTS=$(realpath "${1:?usage: setup.sh <best.pt> [imgsz]}")
IMGSZ=${2:-1344}
HERE=$(cd "$(dirname "$0")" && pwd)
DS_YOLO="$HERE/DeepStream-Yolo"
MODELS="$HERE/models"
: "${CUDA_VER:?set CUDA_VER, e.g. CUDA_VER=12.6}"

if [ ! -d "$DS_YOLO" ]; then
    git clone https://github.com/marcoslucianops/DeepStream-Yolo.git "$DS_YOLO"
fi
make -C "$DS_YOLO/nvdsinfer_custom_impl_Yolo" CUDA_VER="$CUDA_VER"

# Export with the DeepStream-Yolo script: it adds the output layer NvDsInferParseYolo expects
# and writes labels.txt in the class order of the trained model.
mkdir -p "$MODELS"
WORK=$(mktemp -d)
cp "$WEIGHTS" "$WORK/capacitor_detector.pt"
(cd "$WORK" && python3 "$DS_YOLO/utils/export_yoloV8.py" -w capacitor_detector.pt -s "$IMGSZ" --dynamic --simplify)
mv "$(ls -t "$WORK"/*.onnx | head -1)" "$MODELS/capacitor_detector.onnx"
mv "$WORK/labels.txt" "$MODELS/labels.txt"
rm -rf "$WORK"

echo "Exported $MODELS/capacitor_detector.onnx ($(wc -l < "$MODELS/labels.txt") classes)"
echo "The TensorRT engine is built on the first run of app.py."
