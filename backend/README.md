# Backend — FastAPI Inference Server

The backend serves the trained YOLOv11 ONNX model via a REST API and
implements the multi-view grading algorithm.

## Quick Start

```bash
cd backend
pip install -r ../requirements.txt

# Place your model file
# cp /path/to/best.onnx models/best.onnx

python app.py
# API docs: http://localhost:8000/docs
```

## Configuration

Environment variables (see `.env.example` in repo root):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best.onnx` | Path to ONNX model file |
| `SAVE_DIR` | `results` | Directory for annotated images |
| `MAX_FILE_AGE_HOURS` | `24` | Auto-cleanup threshold |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

## Project Structure

```
backend/
├── app.py                  # FastAPI routes & server entry
├── detector/
│   ├── __init__.py
│   └── detector.py         # ONNX inference + grading engine
├── models/                 # Place best.onnx here
│   └── README.md
├── results/                # Runtime output (auto-cleaned)
│   └── .gitkeep
└── test_images/            # Sample images for testing
    └── .gitkeep
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/predict` | Single-view defect detection |
| POST | `/api/v1/grade` | Multi-view grading & price estimation |
| GET | `/health` | Health check |
| GET | `/static/{file}` | Serve annotated images |
| GET | `/docs` | Swagger UI |
| GET | `/` | Redirect to `/docs` |

Full API docs: [`../docs/api_reference.md`](../docs/api_reference.md)

## Inference Pipeline

```
Image → letterbox → BGR2RGB → HWC2CHW → normalize → ONNX inference
      → transpose → conf filter → NMS → coord restore → visualize
```

See [`../docs/architecture.md`](../docs/architecture.md) for details.

## GPU Acceleration

Install the GPU version of ONNX Runtime:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu==1.16.3
```

The detector automatically tries CUDA first and falls back to CPU.
