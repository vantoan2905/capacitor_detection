# Capacitor Detection

## Overview

Capacitor Detection is a deep learning-based project designed to identify and classify capacitors in images. The project leverages computer vision techniques to enhance accuracy in detecting different types of capacitors.

## Features

- Automated Detection: Identifies capacitors in images with high accuracy.
- Classification: Differentiates between various types of capacitors.
- Visualization: Provides bounding box overlays for detected capacitors.
- Deployment: Can be integrated into real-time applications.

## Tech Stack

- **Programming Language:** Python
- **Frameworks/Libraries:** TensorFlow, PyTorch, OpenCV, Ultralytics
- **Dataset:** Custom-labeled capacitor images
- **Model:** CNN-based object detection (YOLO, Faster R-CNN)

## Installation

### Clone the repository
```sh
git clone https://github.com/your-repo/capacitor-detection.git
cd capacitor_detection
```

### Install dependencies
```sh
pip install -r requirements.txt
```

## Training Models

### YOLO Training
```sh
cd YOLO_Training
```

#### YOLO v8 Configuration
- **Config Capacitor:**
  ```sh
  cd children_bbox_detection/config
  ```
- **Config Area Capacitor:**
  ```sh
  cd father_bbox_detection/config
  ```

### Faster-RCNN Training
```sh
cd faster_RCNN_detection
```

