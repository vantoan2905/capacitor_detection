# preprocess_train_YOLO
preprocessdata train YOLO
"# preprocess_train_YOLO" 
# YOLO Preprocessing and Training

This repository contains scripts and configurations for preprocessing datasets and training YOLO (You Only Look Once) models, specifically optimized for object detection tasks. The setup is designed for flexibility and ease of use, allowing customization for various datasets and training scenarios.

## Features

- **Dataset Preprocessing**:
  - Normalizes bounding boxes to YOLO format: `[class_id center_x center_y width height]`.
  - Supports automatic dataset splitting into training, validation, and testing sets.
  - Includes visualization tools to inspect annotated data.

- **Model Training**:
  - Supports YOLOv5, YOLOv7, and YOLOv8 architectures.
  - Configurable hyperparameters for batch size, learning rate, and epochs.
  - Fine-tuning with pretrained weights for faster convergence.
  
- **Evaluation**:
  - Provides mAP (mean Average Precision) metrics for model performance.
  - Visualizes predictions on test datasets.



