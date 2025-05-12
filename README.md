# Object Detection Model Comparison: YOLOv8 vs Deformable DETR

This repository contains code and analysis for a comprehensive comparison between two state-of-the-art object detection models: YOLOv8 (CNN-based) and Deformable DETR (Transformer-based).

## Project Overview

This project evaluates and compares the performance of YOLOv8x and Deformable DETR on identical test images, analyzing metrics such as inference speed, memory usage, detection accuracy, and confidence scores.

## Key Findings

- **Speed & Efficiency**: YOLOv8 processes images 4× faster (0.11s vs 0.45s) and achieves 17.51 FPS vs DETR's 2.75 FPS
- **Memory Usage**: YOLOv8 uses 60% less memory (647MB vs 1656MB)
- **Detection Quality**: YOLOv8 detects 2.2× more objects per image with higher confidence scores
- **Consistency**: YOLOv8 shows more consistent performance across diverse images

## Repository Contents

- `Final_Analysis.py`: Main script for running comparative tests
- `requirements.txt`: Dependencies required to run the code
- `Analysis_WriteUp.pdf`: Detailed analysis of results
- `Results/`: Directory containing output images and metrics

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have a compatible GPU with CUDA support for optimal performance

## Usage

Run the analysis script:
```bash
python Final_Analysis.py
```

The script will:
- Process test images with both models
- Calculate performance metrics
- Generate visualizations with bounding boxes
- Save comparison results in tabular format

## Model Details

### Deformable DETR
- Transformer-based architecture with deformable attention
- Uses ResNet-50 backbone
- Trained on COCO 2017 dataset
- Citation: Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2021). Deformable DETR: Deformable Transformers for End-to-End Object Detection. In International Conference on Learning Representations (ICLR).

### YOLOv8
- CNN-based architecture with anchor-free detection
- Custom CSPDarknet53 backbone with C2f modules
- Trained on COCO dataset
- Citation: Jocher, G., Chaurasia, A. and Qiu, J. (2023) Ultralytics YOLOv8.

## Evaluation Methodology

- Consistent test dataset of 11 images
- Standardized hardware (NVIDIA GeForce RTX 4060 GPU)
- Uniform confidence threshold (0.6)
- Controlled image size (640px)
- Memory tracking and timing methodology

## Conclusion

While both models offer state-of-the-art approaches to object detection, YOLOv8's superior speed, accuracy, and resource efficiency make it the clear choice for practical applications, particularly those requiring real-time processing or deployment on devices with limited computational resources.

