"""
Reproduce the 10-group controlled ablation study (Thesis Chapter 4 / Table 4-2)
===============================================================================

Every run below uses the EXACT hyperparameters recorded in the archived
`args.yaml` of each experiment. Base configuration shared by all runs:

    model:    yolo11s.pt (Exp 9/10: yolov11s-cbam.yaml)
    data:     2,707-image custom phone-defect dataset (4 classes)
    epochs:   100      patience: 25     batch: 8     imgsz: 640
    lr0:      0.0005   optimizer: auto
    box:      12.0     cls: 3.0         dfl: 3.0
    fliplr:   0.5      translate: 0.1   scale: 0.2   erasing: 0.4

Usage:
    python reproduce_experiments.py --data /path/to/dataset.yaml --exp 6
    python reproduce_experiments.py --data /path/to/dataset.yaml --exp all
"""

import argparse

from ultralytics import YOLO

# Shared base hyperparameters (verified from archived args.yaml)
BASE = dict(
    epochs=100,
    patience=25,
    batch=8,
    imgsz=640,
    optimizer="auto",
    lr0=0.0005,
    box=12.0,
    cls=3.0,
    dfl=3.0,
    fliplr=0.5,
    translate=0.1,
    scale=0.2,
    erasing=0.4,
)

# Experiment matrix — mirrors thesis Table 4-2 (columns: box, Inner-IoU,
# CBAM, imgsz, mosaic, flipud). Final metrics are the last-epoch values of
# experiments/raw_results/<exp>/results.csv.
EXPERIMENTS = {
    1: dict(
        name="Baseline (YOLOv11s, no augmentation)",
        model="yolo11s.pt",
        overrides=dict(mosaic=0.0, flipud=0.0),
        result=dict(mAP50=0.907, precision=0.944, recall=0.848),
    ),
    2: dict(
        name="Baseline with lower box gain (box=7.5)",
        model="yolo11s.pt",
        overrides=dict(box=7.5, mosaic=0.0, flipud=0.0),
        result=dict(mAP50=0.899, precision=0.955, recall=0.848),
    ),
    3: dict(
        name="Inner-IoU loss only (ratio=0.80)",
        model="yolo11s.pt",
        # Inner-IoU requires the patched loss in train_cbam_hook-style tooling;
        # see training/README.md for the custom-loss integration notes.
        inner_iou_ratio=0.80,
        overrides=dict(mosaic=0.0, flipud=0.0),
        result=dict(mAP50=0.902, precision=0.959, recall=0.844),
    ),
    4: dict(
        name="Inner-IoU (ratio=0.75) + lower box gain",
        model="yolo11s.pt",
        inner_iou_ratio=0.75,
        overrides=dict(box=7.5, mosaic=0.0, flipud=0.0),
        result=dict(mAP50=0.893, precision=0.942, recall=0.845),
    ),
    5: dict(
        name="Inner-IoU (0.80) + multi-parameter (imgsz=800, mosaic=0.5, cls=4.0)",
        model="yolo11s.pt",
        inner_iou_ratio=0.80,
        overrides=dict(imgsz=800, mosaic=0.5, cls=4.0),
        result=dict(mAP50=0.905, precision=0.951, recall=0.844),
    ),
    6: dict(
        name="Improved Baseline (BEST): physics-prior augmentation only",
        model="yolo11s.pt",
        overrides=dict(mosaic=0.4, flipud=0.5),
        result=dict(mAP50=0.965, precision=0.948, recall=0.922),
    ),
    7: dict(
        name="Inner-IoU (0.70) + flipud/mosaic augmentation",
        model="yolo11s.pt",
        inner_iou_ratio=0.70,
        overrides=dict(mosaic=0.4, flipud=0.5),
        result=dict(mAP50=0.954, precision=0.969, recall=0.884),
    ),
    8: dict(
        name="Inner-IoU (0.70) + single-layer CBAM + augmentation",
        model="yolov11s-cbam.yaml",
        inner_iou_ratio=0.70,
        overrides=dict(mosaic=0.4, flipud=0.5),
        result=dict(mAP50=0.913, precision=0.976, recall=0.882),
    ),
    9: dict(
        name="Single-layer CBAM only",
        model="yolov11s-cbam.yaml",
        overrides=dict(mosaic=0.0, flipud=0.0),
        result=dict(mAP50=0.877, precision=0.957, recall=0.827),
    ),
    10: dict(
        name="Multi-layer CBAM only",
        model="yolov11s-cbam-multi.yaml",  # CBAM injected at backbone end + neck fusion
        overrides=dict(mosaic=0.0, flipud=0.0),
        result=dict(mAP50=0.868, precision=0.930, recall=0.814),
    ),
}


def run(exp_id: int, data: str) -> None:
    cfg = EXPERIMENTS[exp_id]
    print(f"\n=== Exp {exp_id}: {cfg['name']} ===")
    overrides = {**BASE, **cfg["overrides"], "data": data, "name": f"exp{exp_id}"}

    if "inner_iou_ratio" in cfg:
        # Inner-IoU experiments were run with a patched loss (ratio configured
        # via the custom trainer). See training/README.md.
        print(f"[note] requires Inner-IoU loss patch (ratio={cfg['inner_iou_ratio']})")

    model = YOLO(cfg["model"])
    model.train(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument(
        "--exp",
        default="all",
        help="Experiment id 1-10, comma list (e.g. '1,6'), or 'all'",
    )
    args = parser.parse_args()

    if args.exp.strip().lower() == "all":
        ids = sorted(EXPERIMENTS)
    else:
        ids = [int(x) for x in args.exp.split(",")]

    for exp_id in ids:
        run(exp_id, args.data)


if __name__ == "__main__":
    main()
