# Thesis Abstract / 论文摘要

**Title**: 基于改进 YOLOv11 的二手手机外观瑕疵检测与智能定级系统设计
**English Title**: Design of a Second-Hand Phone Appearance Defect Detection and Intelligent Grading System Based on Improved YOLOv11

**Author**: 侯天博 (Tianbo Hou)
**Advisor**: 熊才权 (Prof. Xiong Caiquan)
**Institution**: 湖北工业大学底特律绿色工业学院 软件工程（中美合作办学）
Detroit Green Industry Institute, Hubei University of Technology — Software Engineering (Sino-US Cooperative Program)

The full thesis (Chinese, after-defense final version): [`thesis_final_zh.pdf`](./thesis_final_zh.pdf)
Reference list: [`references.md`](./references.md)

---

## 中文摘要

随着二手智能手机交易量不断增加，传统的人工目测的方式进行外观成色判断存在主观性强、工作效率低、缺少统一评判标准等问题，严重影响二手手机市场的发展。为解决这些问题，在此基础上，基于YOLOv11目标检测算法提出一种面向中小商户的二手手机外观缺陷自动识别及等级划分的方法。

本文的主要研究工作与成果如下：

第一，在特定多材质混合场景下的工业数据集不足的情况下，本文自制一个包含 **2707** 张高质量实拍图像的手机表面缺陷专用数据集。利用严格的标注标准及轻度物理增强方法避免了之前基于AI辅助标注造成的大量虚假正例带来的影响，使得对于小目标检测有了较好的数据支持。

第二，进行10组消融实验进一步研究特定算法模块对于手机强反光、目标极端长宽比等情况下的表现。结果表明，在对比盲目使用CBAM注意力模块以及Inner-IoU损失所造成的负面影响后，根据实际情况而设定的空间翻转（flipud=0.5）及动态背景均衡（mosaic=0.4）的数据增强方法更具有实用性。这种方法无需增加额外参数量的情况下，使基础模型的 mAP@0.5 达到 **96.5%**，同时提高召回率至 **92.16%**。

最后，实现从视觉检测直至商业评价的整体流程。提出一种"对角两拍法"的多角度拍摄方式以及利用对数平滑的一种非线性加权扣分的数学模型，将视觉缺陷转化为商业成色等级（S/A/B/C/D）。在工程技术上使用 ONNX 推理框架、FastAPI 异步后端和微信小程序前端进行开发，在保证单次推理耗时小于 **150 毫秒** 的同时做到全流程耗时不超过 **1.5 秒**。

**关键词**：YOLOv11；缺陷检测；消融实验；智能定级；数据增强

---

## English Abstract

Since the second-hand mobile phone market is still growing, the conventional manual visual assessment to determine the condition is revealing some critical pain points, including high level of subjectivity, inefficiency, and absence of uniform quantitative criteria that strongly hinder the standardization of the industry. To solve these problems, the current study offers to design and implement an intelligent appearance defect detection and quantification system of a second-hand mobile phone based on the YOLOv11 algorithm of object detection, but with a smaller scope of use (small and mid-size retailers).

The main contributions and results of this study are as follows:

First, to address the absence of industrial data on particular multi-material mix conditions, a special smartphone surface defect dataset consisting of **2,707** high-purity real-shot images was built anew. With stringent manual annotation guidelines and gentle physical augmentation methods, the initial AI-assisted annotation noise of high-noise false-positive interference was efficiently tackled and a solid data basis of small object perception was established.

Secondly, the mechanisms of failures of particular algorithm elements in conditions with high reflections and extreme aspect ratios were also thoroughly explored using 10 sets of systematic ablation experiments. Experiments proved that in comparison with the negative impacts of introducing the CBAM attention mechanism or the Inner-IoU loss function randomly, a physics-prior-based data augmentation strategy (flipud=0.5 and dynamic mosaic=0.4) has higher engineering value. This strategy did not increase any network parameters but increased the baseline model mAP@0.5 to **96.5%** and its recall rate to **92.16%**.

Lastly, a comprehensive closed loop system of visual identification to commercial assessment was developed and deployed. The concept of a diagonal dual-shot multiple-view acquisition plan and the formulation of a non-linear weighted deduction model in conjunction with a logarithmic smoothing device were suggested in order to objectively convert visual defect characteristics into commercial condition ratings (S/A/B/C/D). In terms of engineering, the system uses the ONNX inference engine, FastAPI asynchronous backend and WeChat Mini Program frontend, which leads to a lightweight full-stack deployment of the system, a single inference latency of less than **150 ms** as well as a full-process response of less than **1.5 seconds**.

**Keywords**: YOLOv11; Defect detection; Ablation study; Intelligent grading; Data augmentation

---

## Key Results Summary (from thesis Tables 4-2 / 4-3)

| Item | Value |
|------|-------|
| Base model | YOLOv11s (~9.4M params, ~21.7 GFLOPs @ 640) |
| Dataset | 2,707 self-built real-shot images, 4 defect classes |
| Best config (Exp 6) | box=12.0, cls=3.0, imgsz=640, mosaic=0.4, flipud=0.5 (no Inner-IoU, no CBAM) |
| Baseline (Exp 1) | mAP@0.5 = 0.907, Precision = 94.36%, Recall = 84.78% |
| **Best (Exp 6)** | **mAP@0.5 = 0.965 (+5.8pp), Precision = 94.80%, Recall = 92.16% (+7.38pp)** |
| Per-class AP@0.5 (best) | scratch 97.2% / crack 97.8% / bump 97.1% / paint 93.9% |
| Grading model | Multi-view weighted deduction with log smoothing → S/A/B/C/D + discount rate |
| Engineering metrics | Single inference < 150 ms; full pipeline < 1.5 s |
