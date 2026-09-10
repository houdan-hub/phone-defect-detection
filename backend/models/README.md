
# Model Weights

Place your trained model file here.

## Download

Download the pre-trained model from GitHub Releases:

**[Download best.onnx (v1.0.0)](https://github.com/houdan-hub/phone-defect-detection/releases/download/v1.0.0/best.onnx)** (36.2 MB)

After downloading, place it in this directory as `best.onnx`:

# From repository root
cd backend/models
# Move or copy the downloaded file here

## Model Specs
表格
| Property | Value |
| --- | --- |
| Format | ONNX |
| Input | 640×640 RGB, float32 |
| Output | [1, 84, 8400] (4 bbox + 80 class scores) |
| Classes | scratch, bump, paint, crack |
| mAP@0.5 | 0.965 |
| Opset | 12 |
