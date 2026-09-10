# Frontend — WeChat Mini Program

Mobile client for capturing phone photos, running AI defect detection,
and viewing the intelligent grading report.

## Prerequisites

- [WeChat DevTools](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)
- A WeChat Mini Program AppID (or use test account)
- Backend server running (see [`../backend/README.md`](../backend/README.md))

## Setup

1. Open WeChat DevTools → **Import Project**
2. Select this `frontend/` directory
3. Enter your AppID (or choose "Test Account")
4. Configure the API endpoint:

```bash
cd miniprogram/utils
cp config.example.js config.js
# Edit config.js and set API_BASE to your backend URL
# e.g., API_BASE: "http://192.168.1.100:8000"
```

> **Note**: WeChat Mini Programs require HTTPS for production. For local
> development, enable "不校验合法域名" (skip domain validation) in
> DevTools → Details → Local Settings.

## Project Structure

```
frontend/
├── miniprogram/
│   ├── app.js / app.json / app.wxss   # App entry & global config
│   ├── pages/
│   │   └── index/                     # Main page (detection + grading)
│   │       ├── index.js               # Page logic
│   │       ├── index.json             # Page config
│   │       ├── index.wxml             # Page template
│   │       └── index.wxss             # Page styles
│   └── utils/
│       ├── config.example.js          # API config template
│       ├── config.js                  # Your API config (gitignored)
│       └── util.ts                    # Utility functions
├── project.config.json                # DevTools project config
├── package.json
└── tsconfig.json
```

## Usage Flow

1. **Enter base price** — the reference price of the phone model in
   like-new condition.
2. **Capture photos** — tap the view buttons (Front / Back / Side) to
   take or select photos. Each photo is sent to the API for detection.
3. **View detections** — annotated images with bounding boxes appear
   after each upload.
4. **Generate report** — tap "Grade" to aggregate all views and compute
   the final score, grade, and estimated resale price.
5. **Inspect defects** — tap any defect card to preview the close-up
   crop and full annotated image.

## API Integration

The frontend communicates with the backend via two endpoints:

- `POST /api/v1/predict` — upload image, receive detections
- `POST /api/v1/grade` — send aggregated results, receive grading report

See [`../docs/api_reference.md`](../docs/api_reference.md) for the
full API specification.

## Customization

- **UI text**: Edit `index.wxml` for labels and button text.
- **Styles**: Edit `index.wxss` for colors, spacing, layout.
- **View names**: The four views (`front`, `back`, `side_diagonal1`,
  `side_diagonal2`) are defined in `index.wxml` and matched to
  position weights in the backend `detector.py`.
