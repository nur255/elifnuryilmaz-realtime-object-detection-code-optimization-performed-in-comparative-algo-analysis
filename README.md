# elifnuryilmaz-realtime-object-detection-code-optimization-performed-in-comparative-algo-analysis
Official code repository for the paper "Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms".

# Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms

Official source code repository accompanying the manuscript:

**"Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms"**

Submitted to **The Visual Computer**.

---

# Overview

This repository contains the complete source code, benchmarking framework, optimization implementations, experimental results, and supporting materials associated with the research paper:

> **Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms**

The study presents a comprehensive comparison of modern object detection architectures, including multiple variants of YOLO, SSD300-VGG16, and EfficientDet, evaluated under identical experimental conditions using the MS COCO 2017 dataset within a unified PyTorch framework.

A primary contribution of this work is the investigation of software-level optimization techniques that significantly improve real-time inference performance without modifying the underlying neural network architectures.

---

# Models Evaluated

## YOLO Family

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

## SSD

* SSD300-VGG16

## EfficientDet

* EfficientDet-D0
* EfficientDet-D1
* EfficientDet-D2
* EfficientDet-D3
* EfficientDet-D4
* EfficientDet-D5
* EfficientDet-D6
* EfficientDet-D7

---

# Dataset

Experiments were conducted using the **MS COCO 2017** dataset.

Dataset characteristics:

* 123,287 annotated images
* 80 object categories
* Bounding-box annotations in COCO JSON format

The dataset is not included in this repository due to its large size.

Official download source:

https://cocodataset.org

---

# Dataset Preparation

After downloading the COCO 2017 dataset, place it inside the following directory:

```text
src/dataset/

├── train2017/
├── val2017/
├── test2017/
└── annotations/
```

The benchmarking framework automatically loads images and annotations from this structure.

Note:

* The detection scripts do not require the COCO dataset.
* The dataset is only required for performance evaluation and benchmarking using `benchmark_comparison.py`.

---

# Performance Metrics

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

# Software-Level Optimization Techniques

The repository contains implementations of multiple optimization strategies designed to improve real-time object detection performance.

## Core Optimizations

* Single GPU model initialization
* Efficient CUDA memory management
* Optimized preprocessing pipeline
* Inference using `torch.no_grad()`
* OpenCV-based visualization
* Improved FPS computation
* Memory cleanup and garbage collection

## Advanced Optimizations

* Batch processing
* Tensor operation optimization
* GPU-accelerated Non-Maximum Suppression (NMS)
* Caching mechanisms
* Reduced CPU-GPU transfer overhead
* Dynamic confidence thresholding

---

# Performance Improvements

The implemented optimizations resulted in:

* FPS improvements of approximately 35–60%
* Inference latency reductions of 8–15 ms
* GPU memory utilization reductions of 10–25%
* Improved visualization smoothness and detection stability

---

# Experimental Environment

## Hardware

* CPU: Intel Core i7
* GPU: NVIDIA GeForce RTX 3050
* RAM: 16 GB

## Software

* Python 3.x
* PyTorch 2.0.1
* CUDA 11.7

---

# Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

* torch
* torchvision
* ultralytics
* opencv-python
* numpy
* matplotlib
* pandas
* pycocotools
* effdet
* timm
* tqdm

---

# Installation

Clone the repository:

```bash
git clone https://github.com/nur255/elifnuryilmaz-realtime-object-detection-code-optimization-performed-in-comparative-algo-analysis.git
```

Navigate into the repository:

```bash
cd elifnuryilmaz-realtime-object-detection-code-optimization-performed-in-comparative-algo-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Real-Time Detection

Each detection script performs inference using a single selected model.

Before execution, specify the desired model inside the corresponding script.

## YOLO Detection

```bash
python yolo_detection.py
```

Supported models:

* yolov8s
* yolov8m
* yolov8l
* yolov9s
* yolov9m
* yolov9l
* yolov10s
* yolov10m
* yolov10l
* yolo11s
* yolo11m
* yolo11l

## SSD Detection

```bash
python ssd_detection.py
```

## EfficientDet Detection

```bash
python efficientdet_detection.py
```

Each model must be executed separately to generate the corresponding video output and performance measurements.

---

# Running Benchmark Evaluation

To reproduce the comparative evaluation reported in the paper:

```bash
python benchmark_comparison.py
```

The benchmarking script computes:

* FPS
* Inference Time
* Precision
* Recall
* F1-Score
* IoU
* mAP
* mAP@0.5:0.95

using the COCO 2017 dataset.

---

# Repository Structure

```text
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── requirements.txt
│
├── src/
│   ├── yolo_detection.py
│   ├── ssd_detection.py
│   ├── efficientdet_detection.py
│   ├── benchmark_comparison.py
│   ├── dataset/
│   └── model_weights/
│
├── sample_videos/
│
├── results/
│
└── figures/
```

---

# Reproducibility

To support transparent and reproducible research, this repository includes:

* Complete source code
* Model implementations
* Benchmarking scripts
* Optimization modules
* Experimental results
* Performance graphs
* Sample videos

All experiments reported in the manuscript were conducted using:

* NVIDIA RTX 3050 GPU
* Intel Core i7 CPU
* 16 GB RAM
* PyTorch 2.0.1
* CUDA 11.7
* MS COCO 2017 dataset

Researchers can reproduce the reported experiments using the provided code and documentation.

---

# Results

The experiments demonstrate that:

* YOLOv10 and YOLO11 achieve the best balance between speed and accuracy.
* EfficientDet models provide strong localization performance.
* SSD300-VGG16 remains lightweight but exhibits lower accuracy than more recent architectures.
* Software-level optimization significantly improves real-time object detection performance without modifying network architectures.

---

# DOI

This repository will be archived through Zenodo to ensure long-term accessibility and reproducibility.

DOI:
10.5281/zenodo.20574222

Permanent URL:
(https://doi.org/10.5281/zenodo.20574222)
---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{yilmaz2026,
title={Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms},
author={Yilmaz, Elif Nur and Navruz, Tugba Selcen},
journal={The Visual Computer},
note={Under Review},
year={2026}
}
```

---

# Authors

**Elif Nur Yılmaz**

**Tuğba Selcen Navruz**

---

# License

This project is distributed under the MIT License.
See the LICENSE file for details.
