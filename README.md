# 智能驾驶感知：2D/3D 目标检测实验（从零开始）

面向教学的完整复现工程：基于 nuScenes v1.0-mini，从零搭建 2D（相机图像）与
3D（LiDAR 点云）两条目标检测流水线，横向对比经典检测器的三条技术路线
（2D 单阶段/两阶段 + 3D 无锚框）。所有训练脚本均可断点续训，中断后重跑自动接续。

> **从零开始的动手顺序**：§0 概览 → §1 环境 → §2 数据 → §3 数据转换 → §4 训练
> → §5 评估 → §6 可视化与出片。自上而下做一遍即可跑通整个项目。

---

## 0. 项目概览

### 0.1 模型阵容（3 个）

| 阵营 | 2D（相机图像） | 3D（LiDAR 点云） |
|---|---|---|
| 单阶段 · 有锚框 | YOLO11n | — |
| 两阶段 · R-CNN | Faster R-CNN | — |
| 单阶段 · 无锚框 | — | CenterPoint（BEV 热力图峰值） |

- 2D 模型：Ultralytics（YOLO）/ torchvision 官方实现（Faster R-CNN）微调
- 3D 模型：**纯 PyTorch 手写**柱体骨干 + CenterPoint 检测头（不依赖 OpenPCDet /
  MMDetection3D），无锚框、无先验框匹配
- 2D / 3D 管线严格分离：图像只喂 2D 模型，点云只喂 3D 模型，无多模态融合

### 0.2 建议学习路径

1. **先 2D 后 3D**：2D 链路（§3 → §4.1 → §4.3）复用官方实现，最容易跑通、建立直觉
2. **3D 看无锚框路线**：CenterPoint（柱体编码 + BEV 热力图峰值），与 2D 的锚框范式
   形成对照，展示"无锚框"检测思路
3. 每训完一个模型就用 §5 统一评估器与 §6 可视化核验，再做下一个

---

## 1. 环境准备

- Python 3.11 + PyTorch(CUDA) + torchvision + ultralytics + numpy + opencv + matplotlib + pillow
- 以下为本工程实测通过的版本（`D:\software\miniconda3\envs\dl`）：

| 包 | 版本 |
|---|---|
| python | 3.11.16 |
| torch | 2.9.0+cu128 |
| torchvision | 0.24.0+cu128 |
| ultralytics | 8.4.138 |
| numpy | 2.4.4 |
| opencv-contrib-python | 5.0.0.93 |
| matplotlib | 3.11.1 |
| pillow | 12.2.0 |

安装（以 CUDA 12.8 为例）：

```powershell
conda create -n dl python=3.11 -y
conda activate dl
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.4.138 numpy==2.4.4 opencv-contrib-python==5.0.0.93 matplotlib==3.11.1 pillow==12.2.0
```

验证：

```powershell
python -c "import torch, ultralytics, cv2; print(torch.__version__, torch.cuda.is_available())"
```

注意事项：

- **Windows 下 DataLoader 固定单进程**（`num_workers=0`）避免死锁，脚本已内置
- 训练默认使用 GPU；显存不足时下调各自脚本里的 `batch_size` / `max_voxels`
- 一键链脚本 `run_chain2.ps1` 默认调用 `python`（可用环境变量 `PY` 指定解释器），
  运行前请先 `conda activate dl`

---

## 2. 数据准备

本工程使用 **nuScenes v1.0-mini**（10 个场景，含 6 路相机 + LiDAR 点云 + 3D 标注）。

1. 到 [nuScenes 官网](https://www.nuscenes.org/download) 注册并下载 `v1.0-mini` 压缩包
   （`v1.0-mini.tgz`，约 4 GB，含 `samples/`、`sweeps/`、`maps/` 与元数据 JSON）
2. 解压到仓库根目录，使目录结构恰为：

```
teach/
└─ v1.0-mini/            # 注意: 仓库根下的一个 v1.0-mini 目录
   ├─ v1.0-mini/         # 元数据 JSON (category/scene/sample/sample_annotation/... )
   ├─ samples/           # CAM_* / LIDAR_TOP 等关键帧 (jpg / bin)
   ├─ sweeps/            # 非关键帧 LiDAR 点云 (bin)
   ├─ maps/  demo/  LICENSE
   └─ .v1.0-mini.txt
```

> 路径由 `nuscenes_common.py` / `convert_2d.py` / `convert_3d.py` 中的 `DATAROOT` 决定
> （= 仓库根目录下的 `v1.0-mini/`），请勿改动层级。

数据划分（2D / 3D 共用同一划分，保证口径可比）：

- mini 共 10 个场景，**按场景名排序后前 8 个为训练集、后 2 个为验证集**
- 3D 训练集额外并入 `sweeps` 帧（约 320 → 2900 帧），验证集只保留 keyframe

原始数据集与转换产物均不入库（见 `.gitignore`）。

---

## 3. 数据转换

**convert_2d.py** — nuScenes → YOLO 2D 数据集

```powershell
python convert_2d.py
```

- 输入：`v1.0-mini/`
- 输出：`nu2d/images/{train,val}`、`nu2d/labels/{train,val}`、`nu2d/data.yaml`
- 做法：遍历 6 路相机 keyframe，把 3D 框投影到图像得到 2D 框（YOLO 归一化格式）
- 关键约束：投影时要求 8 个角点全部 `z > 0.5m`，否则丢弃该框
  （防近零深度使投影坐标爆炸、生成横跨整幅图的错误巨框）

**convert_3d.py** — nuScenes → 3D 点云数据集

```powershell
python convert_3d.py
```

- 输入：`v1.0-mini/`
- 输出：`nu3d/{train,val}/*.npz`，每帧含 `points (N,4)` 与
  `boxes (M,8) = [cx,cy,cz,w,l,h,yaw,cls]`
- 说明：训练集用标注 prev/next 链按时间戳线性插值补 sweeps；验证集保持 keyframe 不变
- 点云范围：`(-50,-50,-5) ~ (50,50,3)`；框中心不在范围内或最小尺寸 < 0.2m 则丢弃

> 两个脚本在重跑前会清空各自的输出目录，避免逻辑变更后残留陈旧样本。

---

## 4. 训练

所有脚本的产物写入 `outputs/<模型名>/`，`train_log.txt` 记录逐轮指标。

### 4.1 2D 训练

**train_yolo.py** — YOLO11 微调

```powershell
python train_yolo.py [yolo11]                            # 缺省则训练全部 (当前仅 yolo11)
python train_yolo.py yolo11 --aug strong --epochs 150    # 更强增强 + 更长训练
python train_yolo.py yolo11 --aug strong --kfold 4       # 场景级 4 折交叉验证
```

- 超参：默认 `EPOCHS=50, BATCH=16, IMGSZ=640`，可用 `--epochs/--imgsz/--batch` 覆盖
- `--aug strong`：加强空间增强（degrees/scale/shear/perspective）+ 混合增强（mosaic/mixup）
  + 余弦退火、末段关 mosaic；颜色类仅常规 HSV 抖动（不反色/不灰度）
- `--kfold K`：按**场景**切 K 折（避免同场景相邻帧泄漏），逐折训练报平均 mAP（±标准差）
- `--tag`：给输出目录加后缀，便于与默认结果并存对比
- 起点权重：根目录 `yolo11n.pt`（Ultralytics 官方）
- 断点续训：Ultralytics 自带 `last.pt`（`resume=True`）
- 输出：`outputs/yolo11/train/weights/best.pt`

**train_frcnn.py** — Faster R-CNN 微调

```powershell
python train_frcnn.py
```

- 超参：`EPOCHS=15, BATCH=4`
- 断点续训：`outputs/frcnn/last.pth`（含模型/优化器/调度器/轮次）
- 输出：`outputs/frcnn/best.pth`

### 4.2 3D 训练

**train_3d.py** — CenterPoint

```powershell
python train_3d.py [epochs]          # 默认 50
```

- 默认 batch = 4（`max_voxels=18000`）
- 优化：AMP 混合精度 + AdamW(lr=1e-3) + 余弦退火；梯度裁剪 `clip_grad_norm=10`
- 断点续训：`outputs/centerpoint/last.pth`（含模型/优化器/调度器/轮次）
- 输出：`outputs/centerpoint/best.pth`（每 5 轮或末轮按验证 mAP 存最优）、`train_log.txt`
- 损失口径：**热力图目标 = 官方高斯斑**（`gaussian_radius` + `draw_gaussian_to_heatmap`，
  OpenPCDet 口径），**损失 = 官方 `neg_loss_cornernet`**（penalty-reduced focal，
  负样本按 `(1-gt)^4` 降权，`sum/num_pos` 归一化）

### 4.3 一键训练链

**run_chain2.ps1**（Windows）— 3 模型串行（YOLO11 → Faster R-CNN → CenterPoint），
已完成的自动跳过、中断后重跑自动接续

```powershell
conda activate dl
powershell -File run_chain2.ps1
```

**run_chain3d.sh**（Linux / 服务器）— 3D 链（CenterPoint）

```bash
bash run_chain3d.sh
```

- 前置：装好 torch(cuda) / numpy / pillow，数据放 `./v1.0-mini/`
- 日志：逐模型写入 `chain_{name}.log`

---

## 5. 评估

统一 2D / 3D 评估口径，保证跨模型可比。

**eval2d.py** — 【库】统一 2D 评估器（YOLO / Faster R-CNN 共用同一 GT 与 AP 计算口径）

- 指标：COCO 风格 mAP@0.5 与 mAP@0.5:0.95（全点插值）
- 关键函数：`load_gt_labels` / `evaluate_full` / `get_val_images` / `iou_matrix`

**eval3d.py** — 【库】统一 3D 评估器

- 匹配：BEV 中心距离阈值 0.5 / 1 / 2 / 4 m，按置信度贪心一对一匹配
- 关键函数：`evaluate_3d`

**tools/eval_unified.py** — YOLO 系权重一键评估 + 预测可视化

```powershell
python tools/eval_unified.py <weights.pt> [viz_n]     # viz_n 默认 6
```

- 预测用 `conf=0.001 / max_det=300 / iou=0.7` 构建完整 PR 曲线（AP 标准协议）
- 输出：终端 mAP 表 + `outputs/{model}/unified_results.json` + 若干预测可视化图

**tools/report_confusion.py** — 固定工作点（conf=0.25, IoU=0.5）的查准/查全与混淆矩阵

```powershell
python tools/report_confusion.py
```

**tools/audit_2d.py** — 用当前 `convert_2d.py` 逻辑全量重算标签，与磁盘标签逐图比对

```powershell
python tools/audit_2d.py
```

- 输出：不一致图数/框数、巨框统计（`w*h>0.15` 或 `w>0.5` 或 `h>0.7`）

---

## 6. 可视化与出片

**tools/viz_bev.py** — BEV 鸟瞰图（点云 + 真值框绿 + 预测框红）

```powershell
python tools/viz_bev.py [帧数]
```

- 输出：`outputs/viz_bev/centerpoint_bev_XX.jpg`

**tools/make_video_2d.py** — CAM_FRONT 连续帧 2D 检测行车视频

```powershell
python tools/make_video_2d.py [fps] [thr] [max_frames] [--model frcnn|yolo11] [--split train|val]
```

- 默认：`fps=12`（nuScenes 原生帧率）、`thr=0.3`、`max_frames=0`（全部帧）
- 输出：`outputs/videos/{model}_cam_front.mp4`（val 加后缀 `_val`）

**tools/make_video_3d.py** — LIDAR_TOP 连续帧 3D 检测 BEV 视频

```powershell
python tools/make_video_3d.py [fps] [thr] [max_frames] [ckpt_path]
```

- 默认：`fps=12`、`thr=0.5`、`max_frames=0`；`ckpt_path` 默认 `outputs/centerpoint/best.pth`
- 观感控制（脚本内常量）：`NMS_DIST`（同类框最小中心距）、`MAX_DET`（每帧限框数）、
  `CLASS_FILTER`（只留动态目标，滤掉 barrier / traffic_cone 等静止杂波）
- 输出：`outputs/videos/centerpoint_bev.mp4`
- 提示：中途 `Ctrl+C` 硬停会破坏 mp4 索引，需等写入器正常收尾

---

## 7. 目录结构

```
teach/
├─ README.md                 # 本文件
├─ nuscenes_common.py        # 【库】nuScenes JSON 表解析 + 坐标变换 (全局<->传感器)
├─ convert_2d.py             # 数据: nuScenes -> YOLO 2D 数据集 (nu2d/)
├─ convert_3d.py             # 数据: nuScenes -> 3D 点云数据集 (nu3d/)
├─ dataset3d.py              # 【库】3D 数据集 + 点云柱体化编码 + 3D 几何增强
├─ models3d.py               # 【库】CenterPoint 检测头与损失
├─ eval2d.py                 # 【库】统一 2D 评估器 (COCO 风格 mAP)
├─ eval3d.py                 # 【库】统一 3D 评估器 (nuScenes 中心距离 mAP)
├─ train_yolo.py             # 训练: YOLO11
├─ train_frcnn.py            # 训练: Faster R-CNN
├─ train_3d.py               # 训练: CenterPoint
├─ run_chain2.ps1            # 一键链: 3 模型串行训练 (Windows PowerShell)
├─ run_chain3d.sh            # 一键链: 3D 链训练 (Linux/服务器 bash)
├─ yolo11n.pt                # YOLO 预训练起始权重 (Ultralytics 官方)
├─ tools/                    # 常用工具 (评估 / 可视化 / 出片 / 审计)
│   ├─ eval_unified.py       # YOLO 系权重 -> 统一口径 mAP + 预测可视化
│   ├─ report_confusion.py   # 固定工作点下的查准/查全与混淆矩阵
│   ├─ viz_bev.py            # BEV 鸟瞰图: 点云 + 真值框 + 预测框
│   ├─ make_video_2d.py      # 出片: CAM_FRONT 逐帧 2D 检测行车视频
│   ├─ make_video_3d.py      # 出片: LIDAR_TOP 逐帧 3D 检测 BEV 视频
│   └─ audit_2d.py           # 审计: nu2d 标签全量重算比对 + 巨框统计
├─ v1.0-mini/                # 原始数据集 (不入库)
├─ nu2d/  nu3d/              # 转换后的数据集 (不入库)
└─ outputs/                  # 全部训练产物与视频 (不入库)
```

> 命名约定：`【库】` 开头的文件不单独运行，供脚本 `import` 使用。

---

## 8. 当前结果

### 8.1 2D（相机图像，480 张验证图，COCO 风格统一评估）

| 模型 | mAP@0.5 | mAP@0.5:0.95 | car AP50 |
|---|---|---|---|
| Faster R-CNN | 0.2823 | 0.1020 | 0.605 |
| YOLO11n | 0.1617 | 0.0670 | 0.557 |

> 数值取自 `outputs/{frcnn,yolo11}_results.json`。
> **YOLO11n 当前偏弱（需注意）**：mAP 明显低于两阶段 Faster R-CNN（0.1617 vs 0.2823）。
> 主因是 nano 级模型容量小、且 v1.0-mini 仅 8 个训练场景（1944 张图）不足以支撑 10 类
> 检测；其价值在速度而非精度，作为单阶段路线对照保留。如需提升，走
> `train_yolo.py --aug strong --epochs 150` 或 `--kfold 4`。

### 8.2 3D（LiDAR 点云，BEV 中心距离 mAP）

**CenterPoint 正在用对齐官方后的口径（高斯热力图目标 + `neg_loss_cornernet`）重训**，
最新数值以 `outputs/centerpoint/train_log.txt` 为准；重训完成前的历史对照：

| 模型 | 口径 | mAP | car | bus | motorcycle | bicycle | pedestrian |
|---|---|---|---|---|---|---|---|
| CenterPoint | 旧·单格 0/1 热力图目标 | 0.0583 | 0.213 | 0.032 | 0.001 | 0.000 | 0.045 |

> 该 0.0583 为单格硬标签口径下的第 30 轮验证值，权重已归档至
> `outputs/centerpoint/_oldtarget/`（`last.pth` / `best.pth` / `train_log.txt`）。
> 由于热力图目标已改为官方高斯斑，该数值与新代码不再对应，仅作历史对照。

### 8.3 视频演示产物

| 文件 | 内容 |
|---|---|
| `outputs/videos/frcnn_cam_front.mp4` | Faster R-CNN，CAM_FRONT 逐帧 2D 检测行车视频 |
| `outputs/videos/yolo11_cam_front.mp4` | YOLO11，CAM_FRONT 逐帧 2D 检测行车视频 |
| `outputs/videos/centerpoint_bev.mp4` | CenterPoint，LIDAR_TOP 逐帧 3D 检测 BEV 俯视视频 |

---

## 9. 已知限制

- v1.0-mini 仅 324 个样本帧（每样本 6 路相机，展开为 1944 张训练图像），
  小目标（远景行人）与稀有类数量受数据量制约
- 验证集只有 5 类存在真值（car=920、bus=191、motorcycle=108、bicycle=53、
  pedestrian=1171，其余 5 类 `num_gt=0`），mAP 只在这 5 类上取平均
- 3D 无锚框头（CenterPoint）热力图目标已对齐官方口径，与「旧口径」的差异如下：
  - 官方 CenterNet/CenterPoint 用**高斯斑热力图目标**（`gaussian_radius` +
    `draw_gaussian_to_heatmap`）配 `neg_loss_cornernet`（负样本按 `(1-gt)^4` 降权，
    `sum/num_pos` 归一化）；旧实现用**单格 0/1 硬标签**且近中心衰减格无降权，
    会把高斯邻域当假阳性全量惩罚。见 `models3d.py: CenterHead.loss`
