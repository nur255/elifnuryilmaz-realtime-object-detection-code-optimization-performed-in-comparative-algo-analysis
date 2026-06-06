# elifnuryilmaz-realtime-object-detection-code-optimization-performed-in-comparative-algo-analysis
Official code repository for the paper "Real-Time Object Detection: Code Optimizations Performed in a Comparative Analysis of YOLO, SSD, and EfficientDet Algorithms".

## Running Real-Time Detection

Each detection script performs inference using a single selected model.

Before execution, specify the desired model name inside the corresponding script.

Example:

YOLO Models:

yolov8s
yolov8m
yolov8l
yolov9s
yolov9m
yolov9l
yolov10s
yolov10m
yolov10l
yolo11s
yolo11m
yolo11l

After selecting the model, run:

python yolo_detection.py

For EfficientDet:

python efficientdet_detection.py

For SSD:

python ssd_detection.py

Each model must be executed separately to generate its corresponding video output and performance measurements.

