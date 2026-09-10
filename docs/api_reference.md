# API Reference

Base URL: `http://<host>:<port>` (default: `http://localhost:8000`)

Interactive Swagger UI: `http://localhost:8000/docs`

---

## POST /api/v1/predict

Run defect detection on a single phone-view image.

### Request

- **Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | JPEG or PNG image of the phone |
| `view_name` | string | Yes | View identifier: `front`, `back`, `side_diagonal1`, `side_diagonal2` |

### Example (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@phone_front.jpg" \
  -F "view_name=front"
```

### Example (Python)

```python
import requests

with open("phone_front.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/v1/predict",
        files={"file": f},
        data={"view_name": "front"},
    )
print(resp.json())
```

### Success Response (200 OK)

```json
{
  "status": "success",
  "view": "front",
  "detections": [
    {
      "class": "scratch",
      "score": 0.873,
      "bbox": [120, 200, 180, 210],
      "area_ratio": 0.001406,
      "crop_image_url": "crop_front_a1b2c3d4_0.jpg"
    },
    {
      "class": "bump",
      "score": 0.921,
      "bbox": [300, 450, 340, 490],
      "area_ratio": 0.003906,
      "crop_image_url": "crop_front_a1b2c3d4_1.jpg"
    }
  ],
  "result_image_url": "http://localhost:8000/static/det_front_a1b2c3d4.jpg"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` on success |
| `view` | string | Echoed `view_name` |
| `detections` | array | List of detected defects (empty if none) |
| `detections[].class` | string | Defect class: `scratch`, `bump`, `paint`, `crack` |
| `detections[].score` | float | Confidence score (0-1), rounded to 3 decimals |
| `detections[].bbox` | array | Bounding box `[x1, y1, x2, y2]` in original image pixels |
| `detections[].area_ratio` | float | Bbox area / (640×640), rounded to 6 decimals |
| `detections[].crop_image_url` | string | Filename of the cropped defect close-up |
| `result_image_url` | string | Full URL to the annotated result image |

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| 400 | `Invalid image format.` | Uploaded file is not an image |
| 400 | `Could not decode image.` | Image file is corrupted |
| 500 | `Internal inference error.` | Model inference failed |
| 503 | `Model not loaded.` | ONNX model file not found |

---

## POST /api/v1/grade

Aggregate detections from all views and compute a grading report.

### Request

- **Content-Type**: `application/json`

```json
{
  "results": {
    "front": {
      "detections": [
        {"class": "scratch", "score": 0.87, "bbox": [120,200,180,210], "area_ratio": 0.0014}
      ],
      "image_name": "det_front_a1b2c3d4.jpg"
    },
    "back": {
      "detections": [],
      "image_name": "det_back_e5f6g7h8.jpg"
    }
  },
  "base_price": 3000
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `results` | object | Yes | — | Map of view name → detection result object |
| `results[view].detections` | array | Yes | — | Detections from `/predict` for that view |
| `results[view].image_name` | string | Yes | — | Filename of the annotated image (for cropping) |
| `base_price` | number | No | 3000 | Reference price in CNY for estimation |

> **Note**: `all_results` is also accepted as an alias for `results` for
> backward compatibility.

### Success Response (200 OK)

```json
{
  "status": "success",
  "report": {
    "score": 94.35,
    "grade": "A (Excellent)",
    "cards": [
      {
        "pos": "front",
        "class_name": "scratch",
        "desc": "Fine scratch, visible under strong light",
        "deduction": 1.15,
        "thumb_url": "/static/thumb_front_0_a1b2c3.jpg",
        "full_image_name": "det_front_a1b2c3d4.jpg"
      }
    ],
    "est_price": 2550,
    "discount_rate": 85
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `report.score` | float | Final quality score (0-100), higher is better |
| `report.grade` | string | Grade label: `S (Like-New)`, `A (Excellent)`, `B (Good)`, `C (Fair)`, `D (For Parts)` |
| `report.cards` | array | Per-defect detail cards with description and deduction |
| `report.est_price` | int | Estimated resale price = `base_price × discount_rate / 100` |
| `report.discount_rate` | int | Resale value as percentage of base price (e.g. 85 = 85%) |

### Error Responses

| Status | Description |
|--------|-------------|
| 400 | `No detection results provided.` |
| 500 | Internal grading error (details in server logs) |
| 503 | `Model not loaded.` |

---

## GET /health

Health check endpoint for monitoring.

### Response

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## GET /static/{filename}

Serve annotated result images and defect thumbnails.

### Headers

- `Cache-Control: no-store, no-cache, must-revalidate` — ensures clients
  always fetch fresh images (important for WeChat Mini Program caching).

---

## GET /

Redirects to `/docs` (Swagger UI).

---

## GET /docs

Interactive Swagger UI for exploring and testing the API.

---

## GET /openapi.json

OpenAPI 3.0 specification in JSON format.
