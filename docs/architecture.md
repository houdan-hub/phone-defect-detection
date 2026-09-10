# System Architecture

## Overview

The system follows a **client-server architecture** with a WeChat Mini Program
as the mobile client and a FastAPI server hosting the ONNX inference engine.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              WeChat Mini Program (frontend/)             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │   │
│  │  │  Photo   │  │  Upload  │  │  Grading Report UI   │  │   │
│  │  │ Capture  │──►│  & API  │──►│  (score, grade,     │  │   │
│  │  │          │  │  Calls   │  │   price, defect cards)│  │   │
│  │  └──────────┘  └──────────┘  └──────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP/REST (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Server Layer (backend/)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              FastAPI Application (app.py)                │   │
│  │  ┌────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │  /predict      │  │  /grade                      │  │   │
│  │  │  (single view) │  │  (multi-view aggregation)    │  │   │
│  │  └───────┬────────┘  └──────────────┬───────────────┘  │   │
│  └──────────┼──────────────────────────┼──────────────────┘   │
│             │                          │                       │
│             ▼                          ▼                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         PhoneDefectDetector (detector/detector.py)       │   │
│  │  ┌────────────┐ ┌────────────┐ ┌─────────────────────┐ │   │
│  │  │ Preprocess │ │  Inference │ │  Postprocess (NMS)  │ │   │
│  │  │ letterbox  │ │  ONNX RT   │ │  coord restore      │ │   │
│  │  └────────────┘ └─────┬──────┘ └─────────────────────┘ │   │
│  │                        │                                │   │
│  │              ┌─────────▼──────────┐                     │   │
│  │              │  Grading Engine    │                     │   │
│  │              │  (weighted scoring)│                     │   │
│  │              └────────────────────┘                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│             │                                                   │
│             ▼                                                   │
│  ┌────────────────────┐  ┌────────────────────────────────┐   │
│  │  ONNX Model        │  │  Static File Server            │   │
│  │  (best.onnx)       │  │  (/static — annotated images)  │   │
│  └────────────────────┘  └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend (WeChat Mini Program)

**File**: `frontend/miniprogram/pages/index/index.js`

- **Photo capture**: Uses `wx.chooseMedia` to let users take or select photos.
- **Upload**: Sends each image via `wx.uploadFile` to `/api/v1/predict`.
- **State management**: Maintains `allResults` keyed by view name
  (`front`, `back`, `side_diagonal1`, `side_diagonal2`).
- **Grading**: After all views are captured, calls `/api/v1/grade` with
  aggregated results and a user-specified base price.
- **UI**: Displays annotated images, defect detail cards, final score,
  grade, and estimated price.

### 2. Backend API (FastAPI)

**File**: `backend/app.py`

- **Singleton model**: The ONNX model is loaded once at startup to avoid
  per-request overhead.
- **Auto-cleanup**: Result images older than 24 hours are deleted on each
  request to bound disk usage.
- **CORS**: Configured for cross-origin requests from the mini program.
- **Health check**: `/health` endpoint for monitoring.
- **Swagger UI**: Auto-generated at `/docs`.

### 3. Detection Engine

**File**: `backend/detector/detector.py`

#### Preprocessing Pipeline
1. **Letterbox**: Resize image preserving aspect ratio, pad to 640×640
   with gray (114,114,114) borders.
2. **Color space**: BGR → RGB (OpenCV loads BGR, model expects RGB).
3. **Layout**: HWC → CHW (channels-first for PyTorch-style models).
4. **Normalization**: Pixel values ÷ 255.0 → [0, 1].
5. **Batch dimension**: Add batch axis → (1, 3, 640, 640).

#### Inference
- ONNX Runtime with `CUDAExecutionProvider` preferred, `CPUExecutionProvider`
  as fallback.
- Input tensor fed via `session.run(None, {input_name: blob})`.

#### Postprocessing
1. **Transpose**: Raw output (1, 84, 8400) → (8400, 84) for row-wise access.
2. **Decode**: First 4 columns = bbox (cx, cy, w, h); remaining 80 = class
   scores (only 4 used for this model).
3. **Confidence filter**: Keep detections with score > 0.25.
4. **NMS**: OpenCV `dnn.NMSBoxes` with IoU threshold 0.45.
5. **Coordinate restore**: Map letterbox coordinates back to original image
   using scale ratio and padding offsets.
6. **Area ratio**: Compute `(w × h) / (640 × 640)` for grading.

### 4. Grading Engine

See [`grading_algorithm.md`](grading_algorithm.md) for full mathematical
derivation.

### 5. Static File Serving

- Annotated images and defect crops are saved to `backend/results/`.
- Served via `/static/{filename}` with `Cache-Control: no-store` headers
  to ensure the mini program always fetches fresh results.

## Data Flow Diagram

```
User captures photo
       │
       ▼
┌──────────────┐
│ Mini Program │─── POST /api/v1/predict (image + view_name)
└──────────────┘
       │
       ▼
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  FastAPI     │────►│  Detector   │────►│  ONNX Model  │
│  app.py      │     │  preprocess │     │  inference   │
│              │◄────│  postprocess│◄────│              │
└──────┬───────┘     └─────────────┘     └──────────────┘
       │
       ├─── Save annotated image + crops to results/
       │
       ▼
 Response: detections + result_image_url
       │
       ▼ (repeat for each view)
       │
┌──────────────┐
│ Mini Program │─── POST /api/v1/grade (all_results + base_price)
└──────────────┘
       │
       ▼
┌──────────────┐     ┌─────────────────┐
│  FastAPI     │────►│ Grading Engine  │
│  app.py      │     │ weighted scoring│
└──────┬───────┘     └────────┬────────┘
       │                      │
       ▼                      ▼
 Response: score, grade,    Defect close-up
 est_price, defect cards    thumbnails saved
```

## Deployment Considerations

- **Model size**: ~5.6 MB (YOLOv11n ONNX), suitable for container deployment.
- **CPU inference**: ~50-100ms per image on modern CPU.
- **GPU inference**: ~5-15ms per image with CUDA.
- **Concurrency**: FastAPI + Uvicorn handles concurrent requests; model
  inference is the bottleneck. For production, consider a task queue
  (Celery/RQ) with GPU workers.
- **Storage**: Result images are auto-cleaned; for persistent storage,
  integrate S3 or similar object storage.
