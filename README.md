# elifnuryilmaz-realtime-object-detection-code-optimization-performed-in-comparative-algo-analysis
Official code repository for the paper "Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms".

# Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms

## Overview

This repository contains the source code, experimental implementations, and evaluation framework accompanying the research paper:

**"Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms"**

The study presents a comprehensive comparison of modern real-time object detection models, including multiple YOLO variants (YOLOv8, YOLOv9, YOLOv10, YOLO11), SSD300-VGG16, and EfficientDet (D0–D7), evaluated on the MS COCO 2017 dataset within a unified PyTorch framework.

A key contribution of this work is the investigation of software-level inference optimizations that improve real-time performance without modifying the underlying model architectures.

---

## Models Evaluated

### YOLO Family

* YOLOv8s
* YOLOv8m
* YOLOv8l
* YOLOv9s
* YOLOv9m
* YOLOv9l
* YOLOv10s
* YOLOv10m
* YOLOv10l
* YOLO11s
* YOLO11m
* YOLO11l

### SSD

* SSD300-VGG16

### EfficientDet

* EfficientDet-D0
* EfficientDet-D1
* EfficientDet-D2
* EfficientDet-D3
* EfficientDet-D4
* EfficientDet-D5
* EfficientDet-D6
* EfficientDet-D7

---

## Dataset

Experiments were conducted using the **MS COCO 2017** dataset.

The dataset contains:

* 123,287 annotated images
* 80 object categories
* Bounding box annotations in COCO JSON format

Dataset link:

https://cocodataset.org

---

## Performance Metrics

The following evaluation metrics are implemented:

* Frames Per Second (FPS)
* Inference Time
* Precision
* Recall
* F1-Score
* Intersection over Union (IoU)
* mAP@0.5
* mAP@0.5:0.95

---

## Code Optimization Techniques

The repository includes implementations of several software-level optimizations designed to improve real-time inference performance:

### Core Optimizations

* Single GPU model initialization
* Efficient CUDA memory management
* Optimized preprocessing pipeline
* Inference using `torch.no_grad()`
* OpenCV-based visualization
* Improved FPS computation
* Memory cleanup and garbage collection

### Advanced Optimizations

* Batch processing
* Tensor operation optimization
* GPU-accelerated Non-Maximum Suppression (NMS)
* Caching mechanisms
* Reduced CPU-GPU transfer overhead
* Dynamic confidence thresholding

These optimizations improved:

* FPS by approximately 35–60%
* Inference latency by 8–15 ms
* GPU memory utilization by 10–25%

---

## Experimental Environment

### Hardware

* CPU: Intel Core i7
* GPU: NVIDIA GeForce RTX 3050
* RAM: 16 GB

### Software

* Python 3.x
* PyTorch 2.0.1 + CUDA 11.7

---

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Example usage:

```bash
python evaluate.py --model yolov10m
```

```bash
python evaluate.py --model efficientdet_d1
```

```bash
python evaluate.py --model ssd300
```

---

## Repository Structure

```text
├── datasets/
├── models/
├── scripts/
├── results/
├── graphs/
├── videos/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Results

The experiments demonstrate that:

* YOLOv10 and YOLO11 provide the best balance between accuracy and speed.
* EfficientDet models achieve strong localization performance.
* SSD300-VGG16 remains computationally lightweight but achieves lower accuracy than modern architectures.
* Software-level optimization significantly improves real-time object detection performance without altering network architectures.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{yilmaz2025realtime,
  title={Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms},
  author={Yilmaz, Elif Nur and Navruz, Tugba Selcen},
  year={2025}
}
```

---

## Authors

Elif Nur Yılmaz

Tuğba Selcen Navruz

---

## License

This project is released under the MIT License.
