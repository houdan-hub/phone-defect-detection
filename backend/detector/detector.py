"""
Phone Defect Detector
=====================
ONNX-runtime inference engine for the improved YOLOv11 model, plus a
rule-based multi-view grading algorithm that maps detected defects to a
quality score and estimated resale price.

Model: YOLOv11n (improved via data-augmentation fine-tuning)
Input: 640 x 640 RGB image
Classes: scratch, bump, paint, crack
"""

import os
import math
import uuid

import cv2
import numpy as np
import onnxruntime as ort


class PhoneDefectDetector:
    """Wraps an ONNX YOLOv11 model for phone-defect inference and grading."""

    # ------------------------------------------------------------------
    # Class-level defect taxonomy & grading weights
    # ------------------------------------------------------------------
    #: Penalty per defect type (higher = more severe deduction).
    PENALTY_MAP = {
        "crack": 50,    # structural / safety risk
        "bump": 15,     # physical deformation
        "paint": 10,    # cosmetic oxidation
        "scratch": 5,   # normal wear
    }

    #: View-position weighting (front face matters most to buyers).
    POSITION_WEIGHTS = {
        "front": 1.0,
        "back": 0.6,
        "side_diagonal1": 0.8,
        "side_diagonal2": 0.8,
    }

    def __init__(self, onnx_path: str):
        """
        Load the ONNX model.  Prefers CUDA, falls back to CPU.

        Args:
            onnx_path: Path to the exported ``.onnx`` model file.
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as exc:
            print(f"[detector] CUDA unavailable, using CPU: {exc}")
            self.session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )

        self.input_name = self.session.get_inputs()[0].name
        self.img_size = 640  # must match training input size
        self.classes = {0: "scratch", 1: "bump", 2: "paint", 3: "crack"}

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def letterbox(self, im, color=(114, 114, 114)):
        """Resize image with aspect-ratio preservation and pad to 640x640."""
        shape = im.shape[:2]
        r = min(self.img_size / shape[0], self.img_size / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (self.img_size - new_unpad[0]) / 2, (self.img_size - new_unpad[1]) / 2

        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def preprocess(self, img_bgr):
        """
        Convert BGR image to model-ready tensor.

        Returns:
            blob: (1, 3, 640, 640) float32 normalized tensor
            ratio: letterbox scale factor
            dwdh: (pad_w, pad_h) for coordinate restoration
        """
        img, ratio, (dw, dh) = self.letterbox(img_bgr)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img, ratio, (dw, dh)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(self, outputs, ratio, dwdh, conf_thres=0.25, iou_thres=0.45):
        """
        Decode raw model output, apply confidence filtering + NMS, and
        restore bounding-box coordinates to the original image space.

        Returns:
            List of dicts with keys: class, score, bbox, area_ratio
        """
        preds = np.squeeze(outputs[0]).T
        boxes = preds[:, :4]
        scores = np.max(preds[:, 4:], axis=1)
        class_ids = np.argmax(preds[:, 4:], axis=1)

        mask = scores > conf_thres
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert [cx, cy, w, h] -> [x1, y1, w, h] for OpenCV NMS
        nms_boxes = [[x - w / 2, y - h / 2, w, h] for x, y, w, h in boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_thres, iou_thres)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # Restore to original image coordinates
                x1 = (x - w / 2 - dwdh[0]) / ratio
                y1 = (y - h / 2 - dwdh[1]) / ratio
                x2 = (x + w / 2 - dwdh[0]) / ratio
                y2 = (y + h / 2 - dwdh[1]) / ratio

                results.append({
                    "class": str(self.classes[class_ids[i]]),
                    "score": float(round(float(scores[i]), 3)),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "area_ratio": float(round(float((w * h) / (self.img_size ** 2)), 6)),
                })
        return results

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_and_save(self, image_bgr, detections, task_id, save_dir="results"):
        """
        Draw bounding boxes + labels on the image, crop each defect region,
        and save everything to *save_dir*.

        Returns:
            Filename of the annotated full image.
        """
        os.makedirs(save_dir, exist_ok=True)
        canvas = image_bgr.copy()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['score']}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop defect close-up
            crop_img = image_bgr[y1:y2, x1:x2]
            if crop_img.size > 0:
                crop_filename = f"crop_{task_id}_{idx}.jpg"
                cv2.imwrite(os.path.join(save_dir, crop_filename), crop_img)
                det["crop_image_url"] = crop_filename

        file_name = f"det_{task_id}.jpg"
        cv2.imwrite(os.path.join(save_dir, file_name), canvas)
        return file_name

    # ------------------------------------------------------------------
    # Grading algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def _defect_description(cls: str, area: float) -> str:
        """Human-readable description of a defect based on type and size."""
        if cls == "scratch":
            return "Fine scratch, visible under strong light" if area < 0.005 \
                else "Prominent scratch, easily noticeable"
        if cls == "bump":
            return "Minor corner dent, concealable with a case" if area < 0.01 \
                else "Noticeable dent, affects body integrity"
        if cls == "paint":
            return "Edge paint wear, normal usage aging" if area < 0.01 \
                else "Oxidized paint loss, case recommended"
        if cls == "crack":
            return "Crack present — water-ingress and cut risk"
        return "Appearance anomaly, inspect close-up"

    @staticmethod
    def _grade_logic(score: float):
        """
        Map a 0-100 score to a grade label and resale discount rate.

        Returns:
            (grade_label, discount_rate)  where discount_rate is a fraction
            of the original price (e.g. 0.85 = 85% of base price).
        """
        if score >= 98:
            return "S (Like-New)", 0.92
        if score >= 90:
            return "A (Excellent)", 0.85
        if score >= 75:
            return "B (Good)", 0.70
        if score >= 60:
            return "C (Fair)", 0.50
        return "D (For Parts)", 0.30

    def calculate_grading(self, all_view_results, base_price=3000, save_dir="results"):
        """
        Aggregate defects across all phone views and compute a final grade.

        Deduction formula for each defect::

            deduction = penalty[type] * log10(1 + area_ratio * 1000) * position_weight

        Args:
            all_view_results: dict mapping view name ->
                ``{"detections": [...], "image_name": "..."}``
            base_price: original / reference price in CNY.
            save_dir: directory containing annotated images for cropping.

        Returns:
            dict with score, grade, per-defect cards, estimated price.
        """
        total_score = 100.0
        defect_cards = []

        for pos, data in all_view_results.items():
            if isinstance(data, list):
                detections = data
                image_name = f"det_{pos}.jpg"
            else:
                detections = data.get("detections", [])
                image_name = data.get("image_name", "")

            pos_weight = self.POSITION_WEIGHTS.get(pos, 0.8)

            img_path = os.path.join(save_dir, image_name)
            raw_img = cv2.imread(img_path) if os.path.exists(img_path) else None

            for i, det in enumerate(detections):
                penalty = self.PENALTY_MAP.get(det["class"], 5)
                area = det["area_ratio"]

                deduction = penalty * math.log10(1 + area * 1000) * pos_weight
                total_score -= deduction

                # Generate defect close-up thumbnail
                thumb_name = f"thumb_{pos}_{i}_{uuid.uuid4().hex[:6]}.jpg"
                if raw_img is not None:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    h, w, _ = raw_img.shape
                    pad = 20
                    cy1, cy2 = max(0, y1 - pad), min(h, y2 + pad)
                    cx1, cx2 = max(0, x1 - pad), min(w, x2 + pad)
                    if cy2 > cy1 and cx2 > cx1:
                        cv2.imwrite(
                            os.path.join(save_dir, thumb_name),
                            raw_img[cy1:cy2, cx1:cx2],
                        )

                defect_cards.append({
                    "pos": pos,
                    "class_name": det["class"],
                    "desc": self._defect_description(det["class"], area),
                    "deduction": round(float(deduction), 2),
                    "thumb_url": f"/static/{thumb_name}",
                    "full_image_name": image_name,
                })

        final_score = float(max(0, round(total_score, 2)))
        grade, discount = self._grade_logic(final_score)

        return {
            "score": final_score,
            "grade": grade,
            "cards": defect_cards,
            "est_price": int(base_price * discount),
            "discount_rate": int(discount * 100),
        }
