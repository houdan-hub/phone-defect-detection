# Training Guide

This document describes how to train, fine-tune, and export the improved
YOLOv11 model for phone defect detection.

## Prerequisites

```bash
pip install ultralytics==8.1.0 torch==2.1.0 torchvision==0.16.0 PyYAML
```

A CUDA-capable GPU is strongly recommended (training on CPU is impractical).

## Dataset Preparation

### Directory Structure

Organize your dataset in YOLO format:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       ├── img_101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt
    │   └── ...
    └── val/
        ├── img_101.txt
        └── ...
```

### Label Format

Each `.txt` file contains one line per detection:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to [0, 1] relative to image dimensions.

### Class Mapping

| ID | Class | Chinese | Description |
|----|-------|---------|-------------|
| 0 | scratch | 划痕 | Surface scratches, fine to prominent |
| 1 | bump | 磕碰 | Dents and physical deformation |
| 2 | paint | 掉漆 | Paint loss and oxidation |
| 3 | crack | 裂纹 | Cracks and fractures |

### data.yaml

```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 4
names: ['scratch', 'bump', 'paint', 'crack']
```

## Data Augmentation

The best-performing configuration (Experiment 6) used these augmentation
parameters:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `mosaic` | 0.4 | Mosaic augmentation probability |
| `flipud` | 0.5 | Vertical flip probability |
| `fliplr` | 0.5 | Horizontal flip probability (default) |
| `hsv_h` | 0.015 | Hue augmentation |
| `hsv_s` | 0.7 | Saturation augmentation |
| `hsv_v` | 0.4 | Value/brightness augmentation |
| `degrees` | 0.0 | No rotation (phone orientation matters) |
| `translate` | 0.1 | Translation augmentation |
| `scale` | 0.5 | Scale augmentation |
| `shear` | 0.0 | No shear |
| `perspective` | 0.0 | No perspective warp |

**Key insight**: Mosaic at 0.4 (lower than the default 1.0) combined with
vertical flip (flipud=0.5) delivered the largest improvement. This makes
sense for phone images — aggressive mosaic can destroy the contextual
shape of a phone, while vertical flip simulates real-world orientation
variation.

## Baseline Training

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # load pretrained weights (base model used in thesis)

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.001,
    patience=20,
    project="runs/detect",
    name="baseline",
    device=0,
)
```

## Best Configuration (Experiment 6 — Reproducing 0.965 mAP)

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=0.0005,
    patience=25,
    # Fixed loss gains shared by ALL 10 experiments (from archived args.yaml)
    box=12.0,        # box loss gain (baseline default in this study)
    cls=3.0,         # classification loss gain
    # Augmentation
    mosaic=0.4,      # dynamic background balancing
    flipud=0.5,      # vertical flip (space isotropy for phone photos)
    # No Inner-IoU, no CBAM — pure physics-prior augmentation
    project="runs/detect",
    name="exp6_best",
    device=0,
)
```

## CBAM Attention (Experiments 8-10)

The repository includes a **zero-library-modification** approach to inject
CBAM attention modules into YOLOv11 via a Python hook:

```bash
cd training
python train_cbam_hook.py
```

See [`train_cbam_hook.py`](../training/train_cbam_hook.py) for the
implementation. The hook injects CBAM modules after specified backbone
layers (`model.3` and `model.5`) without modifying the Ultralytics source.

> **Finding**: CBAM did **not** improve performance on this dataset
> (single-layer: -3.3%, multi-layer: -4.3%). This is likely because the
> dataset is small and the model already has sufficient capacity; the
> attention module introduced overfitting. This negative result is
> documented as part of the ablation study.

## Inner-IoU Loss (Experiments 3-5, 7)

Inner-IoU is an improved bounding-box regression loss that can be enabled
by modifying the Ultralytics loss function. Refer to the Ultralytics
documentation for custom loss integration.

Results:
- Inner-IoU alone: -0.5% (not effective)
- Inner-IoU + flipud augmentation: +4.7% (second-best overall)

## ONNX Export

After training, export the best model to ONNX for deployment:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/exp6_best/weights/best.pt")
model.export(
    format="onnx",
    imgsz=640,
    simplify=True,
    opset=12,
)
# Output: runs/detect/exp6_best/weights/best.onnx
```

Copy the exported `best.onnx` to `backend/models/best.onnx`.

## Evaluation

```python
from ultralytics import YOLO

model = YOLO("runs/detect/exp6_best/weights/best.pt")
metrics = model.val(data="data.yaml", imgsz=640)

print(f"mAP@0.5:      {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")
```

## Reproducing the Full Ablation Study

The 10 experiments can be reproduced by varying these parameters:

| Exp | box | cls | mosaic | flipud | Inner-IoU | CBAM | imgsz |
|-----|-----|-----|--------|--------|-----------|------|-------|
| 1 | 7.5 | 0.5 | 1.0 | 0.0 | No | No | 640 |
| 2 | 5.0 | 0.5 | 1.0 | 0.0 | No | No | 640 |
| 3 | 7.5 | 0.5 | 1.0 | 0.0 | Yes | No | 640 |
| 4 | 5.0 | 0.5 | 1.0 | 0.0 | Yes | No | 640 |
| 5 | 12.0 | 4.0 | 0.5 | 0.0 | Yes | No | 800 |
| **6** | **12.0** | **3.0** | **0.4** | **0.5** | **No** | **No** | **640** |
| 7 | 7.5 | 0.5 | 0.4 | 0.5 | Yes | No | 640 |
| 8 | 7.5 | 0.5 | 1.0 | 0.0 | Yes | Yes | 640 |
| 9 | 7.5 | 0.5 | 1.0 | 0.0 | No | Single | 640 |
| 10 | 7.5 | 0.5 | 1.0 | 0.0 | No | Multi | 640 |

Raw results: [`../experiments/experiment_summary.csv`](../experiments/experiment_summary.csv)
