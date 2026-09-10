# Training Scripts & Configuration

This directory contains the model training code and configuration files
for the improved YOLOv11 phone defect detector.

## Files

| File | Description |
|------|-------------|
| `yolo11n-cbam.yaml` | YOLOv11n model architecture with CBAM attention module |
| `train_cbam_hook.py` | Training script that injects CBAM via a zero-modification hook |

## Quick Start

```bash
# Install training dependencies
pip install ultralytics torch torchvision PyYAML

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

model = YOLO("yolov11n.pt")
model = inject_cbam_into_model(model, target_layer_name="model.3")
model = inject_cbam_into_model(model, target_layer_name="model.5")
model.train(data="data.yaml", epochs=50, ...)
```

This approach:
- Does **not** modify the Ultralytics library source code
- Injects attention modules after specified backbone layers
- Is compatible with standard `model.train()` and `model.export()`

> **Note**: The `inject_cbam_into_model` helper is referenced in the
> script. For a complete implementation, refer to the Ultralytics
> documentation on custom modules, or implement CBAM as a custom
> `nn.Module` and insert it via the model's `model` attribute.

## Model Configuration

The `yolo11n-cbam.yaml` defines the YOLOv11n architecture:
- Backbone: C3k2 + C2PSA (standard YOLOv11)
- Head: PANet with Detect layer at P3/P4/P5
- Scales: n (nano), s (small), m, l, x
- The CBAM variant modifies specific backbone layers

## Training Best Practices (from this project)

1. **Start from pretrained weights** (`yolov11n.pt`) — transfer learning
   is essential for small datasets.
2. **Use moderate augmentation** — mosaic=0.4, flipud=0.5 worked best.
3. **Tune loss gains** — box=12.0, cls=3.0 improved localization.
4. **Early stopping** — patience=20 prevents overfitting.
5. **Always run a baseline first** — you need a reference point to
   measure improvements.

## Export to ONNX

```python
from ultralytics import YOLO
model = YOLO("runs/detect/exp/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
```

Place the exported `best.onnx` in `backend/models/`.
