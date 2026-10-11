# 智能驾驶感知：2D/3D 目标检测实验

基于 nuScenes v1.0-mini 数据集的 2D/3D 目标检测全流程实现与横向对比实验。
覆盖经典检测器的四条技术路线：单阶段/两阶段 × 有锚框/无锚框。
所有脚本均可断点续训，中断后重跑自动接续。

---

## 1. 模型阵容（8 个）

| 阵营 | 2D（相机图像） | 3D（LiDAR 点云） |
|---|---|---|
| 单阶段·有锚框 | YOLO11n、YOLO26n、SSD-Lite | PointPillars、VoxelNet |
| 两阶段·R-CNN | Faster R-CNN | Voxel R-CNN 风格（伪图像 RoI 精修） |
| 单阶段·无锚框 | — | CenterPoint 风格（BEV 热力图峰值） |

- 2D 模型：Ultralytics（YOLO）/ torchvision 官方实现（SSD、Faster R-CNN）微调
- 3D 模型：纯 PyTorch 手写骨干 + 检测头（不依赖 OpenPCDet / MMDetection3D），
  仅 RoI 池化等成熟算子用 `torchvision.ops`
- 2D / 3D 管线严格分离：图像只喂 2D 模型，点云只喂 3D 模型，无多模态融合

---

## 2. 目录结构

```
teach/
├─ README.md                 # 本文件
├─ nuscenes_common.py        # 【库】nuScenes JSON 表解析 + 坐标变换 (全局<->传感器)
├─ convert_2d.py             # 数据: nuScenes -> YOLO 2D 数据集 (nu2d/)
├─ convert_3d.py             # 数据: nuScenes -> 3D 点云数据集 (nu3d/)
├─ dataset3d.py              # 【库】3D 数据集 + 点云体素化 + 3D 几何增强
├─ models3d.py               # 【库】PointPillars / VoxelNet / CenterPoint 检测头与损失
├─ eval2d.py                 # 【库】统一 2D 评估器 (COCO 风格 mAP)
├─ eval3d.py                 # 【库】统一 3D 评估器 (nuScenes 中心距离 mAP)
├─ train_yolo.py             # 训练: YOLO11 / YOLO26
├─ train_ssd.py              # 训练: SSD-Lite
├─ train_frcnn.py            # 训练: Faster R-CNN
├─ train_3d.py               # 训练: PointPillars / VoxelNet / CenterPoint
├─ train_rcnn3d.py           # 训练: 两阶段 3D (PointPillars 提候选 + RoI 精修)
├─ run_chain2.ps1            # 一键链: 8 模型串行训练 (Windows PowerShell)
├─ run_chain3d.sh            # 一键链: 3D 四模型串行训练 (Linux/服务器 bash)
├─ yolo11n.pt / yolo26n.pt   # YOLO 预训练起始权重 (Ultralytics 官方)
├─ tools/                    # 常用工具 (评估 / 可视化 / 出片 / 审计)
│   ├─ eval_unified.py       # YOLO 系权重 -> 统一口径 mAP + 预测可视化
│   ├─ report_confusion.py   # 固定工作点下的查准/查全与混淆矩阵
│   ├─ viz_bev.py            # BEV 鸟瞰图: 点云 + 真值框 + 预测框
│   ├─ make_video_2d.py      # 出片: CAM_FRONT 逐帧 2D 检测行车视频
│   ├─ make_video_3d.py      # 出片: LIDAR_TOP 逐帧 3D 检测 BEV 视频
│   ├─ audit_2d.py           # 审计: nu2d 标签全量重算比对 + 巨框统计
│   └─ one_off/              # 归档: 一次性诊断/根因验证脚本 (见 §5.5)
├─ v1.0-mini/                # 原始数据集 (不入库)
├─ nu2d/  nu3d/              # 转换后的数据集 (不入库)
└─ outputs/                  # 全部训练产物与视频 (不入库)
```

> 命名约定：`【库】` 开头的文件不单独运行，供脚本 `import` 使用。

---

## 3. 环境

- conda env `dl`（示例路径 `D:\software\miniconda3\envs\dl\python.exe`）
- Python + PyTorch(CUDA) + torchvision + ultralytics + numpy + opencv + matplotlib + pillow
- Windows 下 DataLoader 固定单进程（`num_workers=0`）避免死锁
- 数据划分：mini 共 10 场景，**前 8 场景训练、后 2 场景验证**，2D/3D 共用同一划分

---

## 4. 快速开始（完整流程）

```powershell
# ① 数据转换（首次运行必需）
python convert_2d.py      # -> nu2d/  (images/labels 的 train/val + data.yaml)
python convert_3d.py      # -> nu3d/  (train 含 sweeps, val 仅 keyframe)

# ② 训练（可单跑，也可用一键链）
powershell -File run_chain2.ps1            # 8 模型串行；已完成的自动跳过，中断自动接续

# ③ 评估（统一口径）
python tools/eval_unified.py outputs\yolo11\train\weights\best.pt 6

# ④ 出片（可视化演示）
python tools/make_video_2d.py 12 0.3 0 --model frcnn
python tools/make_video_3d.py 12 0.3 0 outputs\centerpoint\best.pth
```

---

## 5. 各脚本用法

### 5.1 数据转换

**convert_2d.py** — 把 nuScenes 6 路相机的 keyframe 图像 + 3D 框投影生成 2D 框
```
python convert_2d.py
```
- 参数：无
- 输入：`v1.0-mini/`
- 输出：`nu2d/images/{train,val}`、`nu2d/labels/{train,val}`、`nu2d/data.yaml`
- 关键约束：投影时要求 8 个角点全部 z>0.5m，否则丢弃该框（防近零深度投影爆炸成巨框）

**convert_3d.py** — 生成 LiDAR 点云 3D 检测数据集
```
python convert_3d.py
```
- 参数：无（`INCLUDE_SWEEPS=True` 控制训练集是否并入 sweeps）
- 输入：`v1.0-mini/`
- 输出：`nu3d/{train,val}/*.npz`，每帧含 `points (N,4)` 与 `boxes (M,8) = [cx,cy,cz,w,l,h,yaw,cls]`
- 说明：训练集用标注 prev/next 链按时间戳线性插值补 sweeps（约 320→2900 帧）；验证集保持 keyframe 不变，保证指标口径可比

### 5.2 训练

**train_yolo.py** — YOLO11 / YOLO26 微调
```
python train_yolo.py [yolo11|yolo26]     # 不填则两个依次训练
```
- 超参：EPOCHS=50, BATCH=16, IMGSZ=640（脚本内常量）
- 起点权重：`yolo11n.pt` / `yolo26n.pt`（根目录）
- 断点续训：Ultralytics 自带 `last.pt`（`resume=True`）
- 输出：`outputs/{yolo11|yolo26}/train/weights/best.pt`

**train_ssd.py** — SSD-Lite 微调
```
python train_ssd.py
```
- 超参：EPOCHS=20, BATCH=16, IMGSZ=320
- 输出：`outputs/ssd/best.pth`

**train_frcnn.py** — Faster R-CNN 微调
```
python train_frcnn.py
```
- 超参：EPOCHS=15, BATCH=4
- 断点续训：`outputs/frcnn/last.pth`（含 模型/优化器/调度器/轮次）
- 输出：`outputs/frcnn/best.pth`

**train_3d.py** — PointPillars / VoxelNet / CenterPoint 训练
```
python train_3d.py [pointpillars|voxelnet|centerpoint] [epochs]
```
- 默认轮数：pointpillars=60、voxelnet=40、centerpoint=50
- 默认 batch：pillar 类=4（max_voxels=18000）、voxel 类=2（max_voxels=30000）
- 优化：AMP 混合精度 + AdamW(lr=1e-3) + 余弦退火
- 断点续训：`outputs/{mode}/last.pth`
- 输出：`outputs/{mode}/best.pth`（每 5 轮或末轮按验证 mAP 存最优）、`train_log.txt`

**train_rcnn3d.py** — 两阶段 3D（Voxel R-CNN 风格）
```
python train_rcnn3d.py [epochs]          # 默认 15
```
- 第一阶段冻结已训练的 PointPillars 提候选，第二阶段在伪图像上 `roi_align` 池化精修
- 输出：`outputs/rcnn3d/best.pth`、`train_log.txt`

### 5.3 评估

**eval2d.py** — 【库】统一 2D 评估器（YOLO/SSD/FRCNN 共用同一 GT 与 AP 计算口径）
- 指标：COCO 风格 mAP@0.5 与 mAP@0.5:0.95（全点插值）
- 关键函数：`load_gt_labels` / `evaluate_full` / `get_val_images` / `iou_matrix`

**eval3d.py** — 【库】统一 3D 评估器
- 匹配：BEV 中心距离阈值 0.5/1/2/4 m，按置信度贪心一对一匹配
- 关键函数：`evaluate_3d`

**tools/eval_unified.py** — YOLO 系权重一键评估 + 预测可视化
```
python tools/eval_unified.py <weights.pt> [viz_n]     # viz_n 默认 6
```
- 预测用 `conf=0.001 / max_det=300 / iou=0.7` 构建完整 PR 曲线（AP 标准协议）
- 输出：终端 mAP 表 + `outputs/{model}/unified_results.json` + 若干预测可视化图

**tools/report_confusion.py** — 固定工作点（conf=0.25, IoU=0.5）的查准/查全与混淆矩阵
```
python tools/report_confusion.py
```

### 5.4 可视化与出片

**tools/viz_bev.py** — BEV 鸟瞰图（点云 + 真值框绿 + 预测框红）
```
python tools/viz_bev.py <pointpillars|voxelnet|centerpoint> [帧数]
```
- 输出：`outputs/viz_bev/<model>_bev_XX.jpg`

**tools/make_video_2d.py** — CAM_FRONT 连续帧 2D 检测行车视频
```
python tools/make_video_2d.py [fps] [thr] [max_frames] [--model frcnn|yolo11|yolo26] [--split train|val]
```
- 默认：`fps=12`（nuScenes 原生帧率，实时观感）、`thr=0.3`、`max_frames=0`（全部帧）
- `--model`：`frcnn`（默认）/ `yolo11` / `yolo26`；`--split train|val`
- 输出：`outputs/videos/{model}_cam_front.mp4`（val 加后缀 `_val`）

**tools/make_video_3d.py** — LIDAR_TOP 连续帧 3D 检测 BEV 视频
```
python tools/make_video_3d.py [fps] [thr] [max_frames] [ckpt_path]
```
- 默认：`fps=12`、`thr=0.5`、`max_frames=0`；`ckpt_path` 默认 `outputs/centerpoint/best.pth`
- 观感控制（脚本内常量）：`NMS_DIST=2.0`（同类框最小中心距）、`MAX_DET=20`（每帧限框数）、
  `CLASS_FILTER`（只留动态目标，滤掉 barrier / traffic_cone 等静止杂波）
- 输出：`outputs/videos/centerpoint_bev.mp4`
- 提示：**第 30 轮等中途快照可先复制 `best.pth` 为独立权重**，再用其出片（避免被后续训练覆盖）；
  中途 `Ctrl+C` 硬停会破坏 mp4 索引，需等写入器正常收尾

### 5.5 审计与诊断

**tools/audit_2d.py** — 用当前 convert_2d 逻辑全量重算标签，与磁盘标签逐图比对
```
python tools/audit_2d.py
```
- 输出：不一致图数/框数、巨框统计（w*h>0.15 或 w>0.5 或 h>0.7）

**tools/one_off/** — 一次性诊断脚本（归档，非流程必需；注意其内部文档字符串写的
`tools/xxx.py` 路径已过时，实际位于 `tools/one_off/`）

| 脚本 | 用途 |
|---|---|
| `diag_axes.py` | 裁决 w/l 轴向约定（LiDAR 点投影 + 两种约定 3D 框叠加） |
| `diag_proj.py` | 复现某验证图 car 的 3D→2D 投影全过程 |
| `diag_scores.py` | 判定"分数标定问题 vs 设计天花板" |
| `prof_rcnn3d.py` | rcnn3d 单步耗时剖析（定位 CPU 瓶颈） |
| `test_overfit.py` | 锚框头过拟合自检（8 帧反复训练能否复现 GT） |
| `test_overfit_balanced.py` | 按类别归一化 focal loss 的根因验证 |
| `test_overfit_center.py` | CenterPoint 无锚框头过拟合自检 |
| `viz_check.py` | 验证集 GT(绿)/预测(红) 并画，人工核验 |
| `viz_giant.py` | 可视化标签中面积最大的框，判定是否真实近距车辆 |

### 5.6 一键训练链

**run_chain2.ps1**（Windows）— 8 模型串行，已完成的自动跳过、中断后重跑自动接续
```
powershell -File run_chain2.ps1
```

**run_chain3d.sh**（Linux/4090 服务器）— 3D 四模型串行（PointPillars → VoxelNet → RCNN3D → CenterPoint）
```
bash run_chain3d.sh
```
- 前置：装好 torch(cuda)/numpy/pillow，数据放 `./v1.0-mini/`
- 日志：逐模型写入 `chain_{name}.log`

---

## 6. 当前结果

### 2D（相机图像，480 张验证图，COCO 风格统一评估）

| 模型 | mAP@0.5 | mAP@0.5:0.95 | car AP50 |
|---|---|---|---|
| Faster R-CNN | 0.2823 | 0.1020 | 0.605 |
| YOLO11n | 0.1617 | 0.0670 | 0.557 |
| YOLO26n | 0.1454 | 0.0587 | 0.462 |
| SSD-Lite | 0.0994 | 0.0384 | 0.380 |

### 3D（LiDAR 点云，BEV 中心距离 mAP）

| 模型 | mAP | car | bus | motorcycle | bicycle | pedestrian |
|---|---|---|---|---|---|---|
| CenterPoint | 0.0086 | 0.017 | 0.001 | 0.003 | 0.000 | 0.023 |
| VoxelNet | 0.0047 | 0.013 | 0.001 | 0.000 | 0.000 | 0.010 |
| PointPillars | 0.0028 | 0.006 | 0.000 | 0.002 | 0.000 | 0.005 |
| Voxel R-CNN | 0.0011 | 0.002 | 0.000 | 0.003 | 0.000 | 0.001 |

> 注：上表 3D 数值为首版结果；CenterPoint 修复分类损失负样本归一化（`.mean()`→`sum()/npos`）
> 后重训，第 30 轮验证 mAP 已达 0.0583，显著优于旧结果，最新数值以 `outputs/*/train_log.txt` 为准。

### 视频演示产物

| 文件 | 内容 |
|---|---|
| `outputs/videos/frcnn_cam_front.mp4` | Faster R-CNN，CAM_FRONT 逐帧 2D 检测行车视频 |
| `outputs/videos/centerpoint_bev.mp4` | CenterPoint，LIDAR_TOP 逐帧 3D 检测 BEV 俯视视频 |

---

## 7. 已知限制

- v1.0-mini 仅 324 个样本帧（每样本 6 路相机，展开为 1944 张训练图像），
  小目标（远景行人）与稀有类数量受数据量制约
- 验证集仅有 5 类存在真值（car/bus/motorcycle/bicycle/pedestrian，其余 5 类 num_gt=0），
  mAP 只在这 5 类上取平均
- 3D 锚框正样本匹配用 BEV 中心距离（<2m）而非标准 BEV IoU 阈值，属简化实现，
  mAP 上限低于标准 PointPillars；为保持已训模型可比性未更改
- 稀有类学习依赖按类归一化的分类损失（见 `models3d.py` 注释与
  `tools/one_off/test_overfit*.py` 验证脚本）
- CenterPoint 分类损失若误用负样本 `.mean()` 归一化，会被总格点数稀释约 1.3 万倍，
  导致热力图饱和、每帧数百个乱框；已改为 CenterNet 原定义 `sum()/npos`（见 `models3d.py`）
