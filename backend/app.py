"""
Mobile Phone Defect Detection & Grading API
============================================
FastAPI backend serving an improved YOLOv11 model for second-hand phone
appearance defect detection and intelligent quality grading.

Endpoints:
  POST /api/v1/predict  - Single-view defect detection
  POST /api/v1/grade    - Multi-view grading & price estimation
  GET  /docs            - Interactive API documentation (Swagger UI)
"""

import os
import uuid
import time
import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, FileResponse

from detector.detector import PhoneDefectDetector

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.onnx")
SAVE_DIR = os.getenv("SAVE_DIR", "results")
MAX_FILE_AGE_HOURS = int(os.getenv("MAX_FILE_AGE_HOURS", "24"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI(title="MobileGrader API", version="1.0.0")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
    """Remove result images older than *max_age_hours* to bound disk usage."""
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if now - os.path.getmtime(filepath) > max_age_seconds:
                try:
                    os.remove(filepath)
                    cleaned += 1
                except OSError as exc:
                    print(f"[cleanup] failed to remove {filename}: {exc}")
    if cleaned:
        print(f"[cleanup] removed {cleaned} files older than {max_age_hours}h")
    return cleaned


def get_base_url(request: Request) -> str:
    """Derive the public base URL from the incoming request."""
    return str(request.base_url).rstrip("/")


# ---------------------------------------------------------------------------
# Middleware & static files
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=SAVE_DIR), name="static")

# ---------------------------------------------------------------------------
# Model loading (singleton, loaded once at startup)
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"[WARN] Model file not found at {MODEL_PATH}. "
          f"Place best.onnx in the models/ directory or set MODEL_PATH.")
detector = PhoneDefectDetector(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Redirect to Swagger UI for convenience."""
    return RedirectResponse(url="/docs")


@app.post("/api/v1/predict")
async def predict_view(
    request: Request,
    file: UploadFile = File(...),
    view_name: str = Form(...),
):
    """
    Run inference on a single phone view image.

    - **file**: JPEG/PNG image of one phone view (front / back / side).
    - **view_name**: Identifier for the view, e.g. ``front``, ``back``,
      ``side_diagonal1``, ``side_diagonal2``.

    Returns a list of detections (class, confidence, bbox, area ratio)
    and a URL to the annotated result image.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    cleanup_old_files(SAVE_DIR, MAX_FILE_AGE_HOURS)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        blob, ratio, dwdh = detector.preprocess(img)
        outputs = detector.session.run(None, {detector.input_name: blob})
        detections = detector.postprocess(outputs, ratio, dwdh)

        task_id = f"{view_name}_{uuid.uuid4().hex[:8]}"
        saved_filename = detector.visualize_and_save(img, detections, task_id)

        base_url = get_base_url(request)
        result_image_url = f"{base_url}/static/{saved_filename}"

        return jsonable_encoder({
            "status": "success",
            "view": view_name,
            "detections": detections,
            "result_image_url": result_image_url,
        })

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[predict] error: {exc}")
        raise HTTPException(status_code=500, detail="Internal inference error.")


@app.post("/api/v1/grade")
async def finalize_grading(payload: dict):
    """
    Aggregate detections from all views and produce a grading report.

    Expected payload::

        {
            "results": {
                "front": {"detections": [...], "image_name": "det_front_xxx.jpg"},
                "back":  {"detections": [...], "image_name": "det_back_xxx.jpg"}
            },
            "base_price": 3000
        }

    Returns a score (0-100), grade letter (S/A/B/C/D), estimated resale
    price, and per-defect detail cards.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    all_results = payload.get("results") or payload.get("all_results")
    base_price = float(payload.get("base_price", 3000))

    if not all_results:
        raise HTTPException(status_code=400, detail="No detection results provided.")

    try:
        report = detector.calculate_grading(all_results, base_price)
        return jsonable_encoder({"status": "success", "report": report})
    except Exception as exc:
        print(f"[grade] error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/static/{file_name}")
async def serve_static(file_name: str):
    """Serve annotated result images with no-cache headers."""
    file_path = os.path.join(SAVE_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok", "model_loaded": detector is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
