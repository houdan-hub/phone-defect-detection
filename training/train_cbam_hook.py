import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from my_modules.inject_hook import inject_cbam_into_model

# ===================== 你的配置 =========================
DATA_YAML = "D:/Graduation_Project/final_dataset_v2_new_scratch/yolov11s-cbam.yaml"
EPOCHS = 50
BATCH = 16
IMGZ = 640
PROJECT = "D:/Graduation_Project/yolo_experiments"
NAME = "exp2_cbam_hook"
# ========================================================

if __name__ == "__main__":
    print("="*70)
    print("       YOLOv11 + CBAM Hook 方案（零修改库）")
    print("="*70)

    # 1. 加载原生模型
    model = YOLO("yolov11n.pt")

    # 2. 自动注入 CBAM（安全、不破坏结构）
    model = inject_cbam_into_model(model, target_layer_name="model.3")
    model = inject_cbam_into_model(model, target_layer_name="model.5")

    # 3. 开始训练
    print("\n开始训练...\n")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGZ,
        batch=BATCH,
        lr0=0.001,
        patience=20,
        project=PROJECT,
        name=NAME,
        exist_ok=False,
        device=0,
        verbose=True
    )

    print("\n训练完成！")