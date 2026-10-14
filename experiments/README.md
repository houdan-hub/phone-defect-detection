# Experiments: Ablation Study

This directory contains the results of a **10-group ablation study**
conducted to identify the most effective improvements to the baseline
YOLOv11s model for phone defect detection. Raw per-epoch training curves
for every run are archived in [`raw_results/`](raw_results/).

## Quick Summary

| Metric | Baseline | **Best (Exp 6)** | Improvement |
|--------|----------|-------------------|-------------|
| mAP@0.5 | 0.907 | **0.965** | **+5.8%** |
| Recall | 0.848 | **0.922** | **+7.4%** |

**Base configuration (fixed across all runs)**: box=12.0, cls=3.0,
imgsz=640, 100 epochs, lr0=0.0005.
**Best configuration (Exp 6)** = base config + physics-prior augmentation
(mosaic=0.4, flipud=0.5), no Inner-IoU, no CBAM.

## Charts

| Chart | Description |
|-------|-------------|
| [`01_map50_comparison.png`](charts/01_map50_comparison.png) | mAP@0.5 bar chart for all 10 experiments |
| [`02_performance_improvement.png`](charts/02_performance_improvement.png) | Performance delta vs. baseline (green=positive, red=negative) |
| [`03_module_benefit_trend.png`](charts/03_module_benefit_trend.png) | Improvement module effectiveness ranking |
| [`04_parameter_heatmap.png`](charts/04_parameter_heatmap.png) | Hyperparameter configuration heatmap |
| [`05_parameter_contribution.png`](charts/05_parameter_contribution.png) | Parameter contribution & importance analysis |
| [`06_performance_trend.png`](charts/06_performance_trend.png) | Performance trend across experiments |
| [`07_best_model_comparison.png`](charts/07_best_model_comparison.png) | Best model vs. baseline detailed comparison |

## Data Files

- [`experiment_summary.csv`](experiment_summary.csv) — 10-experiment
  comparison with mAP, precision, recall, and configuration.
- [`experiment_details.csv`](experiment_details.csv) — Per-experiment
  file paths and detailed metric analysis.

## Key Findings

1. **Data augmentation is the most effective improvement** (+5.8% mAP).
   Reduced mosaic (0.4) + vertical flip (0.5) outperformed all
   architectural modifications.

2. **Inner-IoU loss helps when combined with augmentation** (+4.7%),
   but is ineffective alone (-0.5%).

3. **CBAM attention hurts performance** on this dataset (-3.3% to -4.3%).
   The small dataset and already-capable model lead to overfitting when
   attention parameters are added. This is a documented negative result.

4. **Hyperparameter tuning matters**: Increasing box loss gain from 7.5
   to 12.0 and cls loss gain to 3.0 contributed to the best result.

5. **Recall improved most dramatically** (+7.4%), meaning the improved
   model catches significantly more true defects — critical for a
   quality-inspection use case where missing a defect is worse than
   a false positive.

## Methodology

All experiments:
- Trained for 100 epochs with early stopping (patience=20)
- Input size: 640×640 (except Exp 5 at 800)
- Batch size: 16
- Optimizer: SGD (default Ultralytics)
- Initial learning rate: 0.001
- Same train/val split across all experiments for fair comparison
- Evaluated on the held-out validation set
