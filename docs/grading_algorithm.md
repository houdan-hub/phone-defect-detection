# Grading Algorithm

## Motivation

The grading system converts raw defect detections into a **single, interpretable
quality score** that maps to a grade letter and resale price. Unlike a neural
network, this rule-based approach is:

- **Transparent**: every deduction can be traced to a specific defect.
- **Auditable**: the formula and weights are explicit and adjustable.
- **Data-efficient**: no labeled grade data required (only detection labels).

## Scoring Formula

For each detected defect *i*, the score deduction is:

```
deduction_i = penalty[type_i] × log₁₀(1 + area_ratio_i × 1000) × position_weight[view_i]
```

The final score starts at 100 and subtracts all deductions:

```
final_score = max(0, 100 - Σ deduction_i)
```

### Component Breakdown

#### 1. Penalty Map (`penalty[type]`)

Severity weighting per defect type:

| Defect | Penalty | Rationale |
|--------|---------|-----------|
| **crack** | 50 | Structural damage — water ingress risk, safety hazard, most severe |
| **bump** | 15 | Physical deformation — affects structural integrity and resale |
| **paint** | 10 | Cosmetic oxidation — visible but not structural |
| **scratch** | 5 | Normal wear — expected on used devices, least severe |

The 10× gap between crack and scratch reflects the real-world price
difference between a cracked phone and one with minor scratches.

#### 2. Area Scaling (`log₁₀(1 + area_ratio × 1000)`)

- `area_ratio` = bounding box area / (640 × 640), the fraction of the
  model input image occupied by the defect.
- The `× 1000` factor scales typical area ratios (0.001–0.05) into a
  range where log₁₀ produces meaningful values.
- **Logarithmic scaling** means:
  - Small defects: near-linear penalty growth
  - Large defects: diminishing returns (a crack covering 5% of the screen
    is not 10× worse than one covering 0.5%)
- This prevents a single very large defect from dominating the score
  unrealistically.

**Example values:**

| area_ratio | log₁₀(1 + area×1000) | Interpretation |
|------------|---------------------|----------------|
| 0.001 | 0.301 | Tiny speck |
| 0.005 | 0.778 | Small scratch |
| 0.01 | 1.041 | Noticeable defect |
| 0.05 | 1.707 | Large defect |
| 0.1 | 2.004 | Very large defect |

#### 3. Position Weight (`position_weight[view]`)

Different phone views matter differently to buyers:

| View | Weight | Rationale |
|------|--------|-----------|
| `front` | 1.0 | Screen is the most visible and expensive component |
| `side_diagonal1` | 0.8 | Side/frame damage is visible but less critical |
| `side_diagonal2` | 0.8 | Same as above |
| `back` | 0.6 | Back glass is less visible; many users use a case |
| *unknown* | 0.8 | Default fallback |

## Grade Mapping

The final score (0-100) maps to a grade and discount rate:

| Score Range | Grade | Discount Rate | Label | Description |
|-------------|-------|---------------|-------|-------------|
| ≥ 98 | **S** | 92% | Like-New | Virtually no defects, pristine condition |
| 90 – 97.9 | **A** | 85% | Excellent | Minor wear only, no structural damage |
| 75 – 89.9 | **B** | 70% | Good | Visible but non-critical defects |
| 60 – 74.9 | **C** | 50% | Fair | Noticeable damage, functional issues possible |
| < 60 | **D** | 30% | For Parts | Significant damage, may not be fully functional |

### Price Estimation

```
estimated_price = base_price × (discount_rate / 100)
```

Where `base_price` is the reference price of the same phone model in
"like-new" condition, provided by the user.

## Worked Example

**Phone**: iPhone 12, base price ¥3000

**Detections:**

| View | Defect | area_ratio | penalty | area_factor | pos_weight | deduction |
|------|--------|------------|---------|-------------|------------|-----------|
| front | scratch | 0.002 | 5 | log₁₀(3)=0.477 | 1.0 | 2.39 |
| front | scratch | 0.001 | 5 | log₁₀(2)=0.301 | 1.0 | 1.51 |
| back | bump | 0.008 | 15 | log₁₀(9)=0.954 | 0.6 | 8.59 |
| side | paint | 0.003 | 10 | log₁₀(4)=0.602 | 0.8 | 4.82 |

**Calculation:**
```
total_deduction = 2.39 + 1.51 + 8.59 + 4.82 = 17.31
final_score = 100 - 17.31 = 82.69
grade = B (Good)  [75 ≤ 82.69 < 90]
discount_rate = 70%
est_price = 3000 × 0.70 = ¥2100
```

## Design Decisions & Limitations

### Why not a learned grading model?

1. **No labeled grade data**: We have detection labels (bounding boxes)
   but not human-assigned grades. Training a grading model would require
   additional annotation.
2. **Interpretability**: Recruiters, buyers, and sellers can understand
   and verify each deduction. A neural grader would be a black box.
3. **Adjustability**: Penalty weights can be tuned per market segment
   (e.g., collectors vs. budget buyers) without retraining.

### Limitations

1. **Area approximation**: `area_ratio` uses the bounding box area, not
   the actual defect pixel area. A loose bbox overestimates severity.
   Instance segmentation would improve this.
2. **No defect depth**: 2D images cannot distinguish a surface scratch
   from a deep gouge. 3D scanning or multi-light imaging would help.
3. **View coverage**: The system assumes the user captures all relevant
   views. A hidden defect (e.g., under a case) is not detected.
4. **Lighting dependence**: Low-light or glossy reflections can cause
   false negatives. The data augmentation pipeline partially mitigates
   this.

### Future Improvements

- Replace bbox area with segmentation mask area for precise measurement.
- Add a confidence-weighted deduction (low-confidence defects penalize less).
- Learn penalty weights from human grading data if available.
- Support per-phone-model calibration (e.g., screen cracks on iPhones
  penalize more than on budget devices).
