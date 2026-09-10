# Thesis Abstract / 论文摘要

## English Abstract

**Title**: Design of a Second-Hand Phone Appearance Defect Detection and
Intelligent Grading System Based on Improved YOLOv11

**Author**: Tianbo Hou (侯天博)
**Advisor**: Prof. Xiong Caiquan (熊才权)
**Institution**: Detroit Green Industry Institute, Software Engineering
(Sino-US Cooperative Program)

### Abstract

The rapid growth of the second-hand smartphone market has created an
urgent need for objective and efficient quality assessment. Traditional
manual inspection is subjective, inconsistent, and difficult to scale.
This thesis presents an end-to-end deep-learning system that automatically
detects appearance defects on second-hand smartphones and assigns an
intelligent quality grade with an estimated resale price.

The system is built on an improved YOLOv11n object detection model,
fine-tuned on a custom dataset of 1,901 annotated images spanning four
defect categories: scratch, bump, paint loss, and crack. A systematic
10-group ablation study was conducted to evaluate the effectiveness of
data augmentation strategies, Inner-IoU loss, and CBAM attention modules.
The best configuration — combining optimized mosaic augmentation
(mosaic=0.4), vertical flip (flipud=0.5), and tuned loss weights
(box=12.0, cls=3.0) — achieves **mAP@0.5 = 0.965**, representing a
**5.8 percentage point improvement** over the baseline model, with
recall increasing from 0.848 to 0.922.

The system architecture comprises three components: (1) a WeChat Mini
Program frontend for multi-view photo capture and result visualization;
(2) a FastAPI backend serving an ONNX-exported model for CPU/GPU
inference; and (3) a transparent, rule-based grading algorithm that
aggregates defects across phone views with position-aware weights and
severity penalties, mapping the resulting score to a grade (S/A/B/C/D)
and an estimated resale price.

Experimental results demonstrate that the system provides consistent,
interpretable, and efficient quality grading, offering a practical
solution for second-hand phone recycling platforms and inspection
workflows.

**Keywords**: YOLOv11; defect detection; data augmentation; ablation
study; intelligent grading; second-hand phone; ONNX deployment; FastAPI

---

## 中文摘要

**题目**: 基于改进 YOLOv11 的二手手机外观瑕疵检测与智能定级系统设计

**作者**: 侯天博
**指导教师**: 熊才权 教授
**学院**: 底特律绿色工业学院 软件工程（中美合作办学）

### 摘要

随着二手智能手机市场的快速增长，客观、高效的质量评估需求日益迫切。
传统的人工检测方式存在主观性强、标准不一、难以规模化等问题。本文
设计并实现了一套端到端的深度学习系统，能够自动检测二手手机外观
瑕疵，并进行智能定级与估价。

系统以改进的 YOLOv11n 目标检测模型为核心，在包含 1,901 张标注图像
的自定义数据集上进行微调，覆盖划痕（scratch）、磕碰（bump）、掉漆
（paint）和裂纹（crack）四类瑕疵。通过系统的 10 组消融实验，评估了
数据增强策略、Inner-IoU 损失函数和 CBAM 注意力模块的有效性。最佳配置
——结合优化的马赛克增强（mosaic=0.4）、垂直翻转（flipud=0.5）和调整
的损失权重（box=12.0, cls=3.0）——达到 **mAP@0.5 = 0.965**，较基线模型
提升 **5.8 个百分点**，召回率从 0.848 提升至 0.922。

系统架构包含三个组件：（1）微信小程序前端，用于多视角拍照和结果
可视化；（2）FastAPI 后端，部署 ONNX 格式模型支持 CPU/GPU 推理；
（3）透明的规则制定级算法，结合视角权重和瑕疵严重度惩罚，对多视角
检测结果进行聚合，将最终得分映射为等级（S/A/B/C/D）和预估转售价格。

实验结果表明，该系统能够提供一致、可解释且高效的质量定级，为二手
手机回收平台和检测流程提供了切实可行的解决方案。

**关键词**: YOLOv11；瑕疵检测；数据增强；消融实验；智能定级；二手手机；
ONNX 部署；FastAPI

---

## Thesis Documents

The full thesis (Chinese, .docx format) and supporting documents
(开题报告, 任务书, 答辩材料, etc.) are available upon request.
This repository contains the code, experiment data, and a condensed
summary suitable for technical review.

### Reference Highlights

Key papers that informed this work:

- *Ultralytics YOLO11* — base object detection architecture
- *CBAM: Convolutional Block Attention Module* — attention mechanism explored
- *Inner-IoU* — improved bounding box regression loss
- *MobileNetV3* — lightweight model design principles
- *Surface defect detection surveys* — industrial inspection methodology
