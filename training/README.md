# Training Scripts & Configuration

This directory contains the model training code and configuration files
for the improved YOLOv11**s** phone defect detector (4 classes: scratch,
bump, paint, crack).

## Files

| File | Description |
|------|-------------|
| `yolov11s-cbam.yaml` | YOLOv11**s** model architecture with CBAM attention module (used in Experiments 8–10) |
| `train_cbam_hook.py` | Training script that injects CBAM via a zero-modification hook |
| `reproduce_experiments.py` | Re-runs any of the 10 ablation experiments with the exact archived hyperparameters |

## Reproduce the Ablation Study

All 10 experiments share one base configuration (verified against the
archived `args.yaml` of every run):

| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| model | `yolo11s.pt` (CBAM: `yolov11s-cbam.yaml`) | epochs | 100 |
| batch | 8 | imgsz | 640 |
| optimizer | auto | lr0 | 0.0005 |
| box gain | 12.0 | cls gain | 3.0 |
| fliplr | 0.5 | translate / scale | 0.1 / 0.2 |

```bash
pip install ultralytics torch torchvision

# Single experiment (e.g. the best one, Exp 6)
python reproduce_experiments.py --data /path/to/dataset.yaml --exp 6

# All 10 experiments
python reproduce_experiments.py --data /path/to/dataset.yaml --exp all
```

The per-experiment overrides (mosaic, flipud, box gain, Inner-IoU ratio,
CBAM variant) are encoded in `reproduce_experiments.py` and mirror thesis
Table 4-2 exactly. Raw per-epoch training curves for every run are archived
in [`experiments/raw_results/`](../experiments/raw_results/).

## Quick Start

```bash
# Standard training (baseline or best config)
# See docs/training_guide.md for the exact hyperparameters

# CBAM injection training (Experiments 8-10)
python train_cbam_hook.py
```

## CBAM Hook Approach

The `train_cbam_hook.py` script demonstrates a **non-invasive** method
to add CBAM (Convolutional Block Attention Module) to YOLOv11:

```python
from ultralytics import YOLO
from my_modules.inject_hook import inject_cbam_into_model

model = YOLO("yolov11s.pt")
model = inject_cbam_into_model(model, target_layer_name="model.3")
model = inject_cbam_into_model(model, target_layer_name="model.5")
model.train(data="data.yaml", epochs=100, ...)
```

This approach:
- Does **not** modify the Ultralytics library source code
- Injects attention modules after specified backbone layers
- Is compatible with standard `model.train()` and `model.export()`

## Model Configuration

The `yolov11s-cbam.yaml` defines the YOLOv11**s** architecture:
- Backbone: C3k2 + C2PSA (standard YOLOv11, s-scale: ~9.4M params, ~21.7 GFLOPs)
- Head: PANet with Detect layer at P3/P4/P5
- `nc: 4` — scratch, bump, paint, crack
- The CBAM variant modifies specific backbone/neck layers

## Training Best Practices (from this project)

1. **Start from pretrained weights** (`yolo11s.pt`) — transfer learning
   is essential for small datasets.
2. **Use physics-prior augmentation** — flipud=0.5 (phones are photographed
   upside-down as often as not) and mosaic=0.4 (balances background
   complexity vs. tiny-target preservation) gave the largest gain: +5.8%
   mAP@0.5 with zero extra parameters.
3. **Fix loss gains and ablate one thing at a time** — box=12.0, cls=3.0
   were held constant across all 10 runs (except where box was itself the
   ablated variable).
4. **Validate negative results too** — CBAM and Inner-IoU *hurt* accuracy
   here; documenting why is more valuable than stacking modules.
5. **Always run a baseline first** — you need a reference point to
   measure improvements.

## Export to ONNX

```python
from ultralytics import YOLO
model = YOLO("runs/detect/exp6/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
```

Place the exported `best.onnx` in `backend/models/`.
