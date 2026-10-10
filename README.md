# 智能驾驶感知：2D/3D 目标检测实验

基于 nuScenes v1.0-mini 数据集的 2D/3D 目标检测全流程实现与对比实验。
覆盖经典检测器的四条技术路线：单阶段/两阶段 × 有锚框/无锚框。

## 模型阵容（8 个）

| 阵营 | 2D（相机图像） | 3D（LiDAR 点云） |
|---|---|---|
| 单阶段·锚框 | YOLO11n、YOLO26n、SSD-Lite | PointPillars、VoxelNet |
| 两阶段·R-CNN | Faster R-CNN | Voxel R-CNN 风格（伪图像 RoI 精修） |
| 单阶段·无锚 | — | CenterPoint 风格（BEV 热力图峰值） |

- 2D 模型：Ultralytics（YOLO）/ torchvision 官方实现（SSD、Faster R-CNN）微调
- 3D 模型：纯 PyTorch 手写骨干+检测头（不依赖 OpenPCDet/MMDetection3D），
  仅 RoI 池化等成熟算子用 `torchvision.ops`
- 2D/3D 管线严格分离：图像只喂 2D 模型，点云只喂 3D 模型，无多模态融合

## 数据

nuScenes v1.0-mini（10 场景）：前 8 场景训练（1944 帧）、后 2 验证（480 帧），
划分按场景严格隔离，2D/3D 共用同一划分。

- `nu2d/`：6 路相机 keyframe 图像 + 3D 框投影生成的 2D 框（YOLO 格式）
- `nu3d/`：LiDAR 点云 (.npz) + 变换到 LiDAR 系的 3D 框标注
- 转换脚本：`convert_2d.py` / `convert_3d.py`，公共解析在 `nuscenes_common.py`

## 目录结构

```
├─ nuscenes_common.py       # nuScenes JSON 表解析 + 坐标变换 (全局<->传感器)
├─ convert_2d.py/3d.py      # 数据转换
├─ train_yolo.py            # YOLO11/YOLO26 训练 (断点续训)
├─ train_ssd.py             # SSD 训练 (断点续训)
├─ train_frcnn.py           # Faster R-CNN 训练 (断点续训)
├─ train_3d.py              # PointPillars/VoxelNet/CenterPoint 训练 (断点续训)
├─ train_rcnn3d.py          # 两阶段 3D (PointPillars 提候选 + RoI 精修)
├─ eval2d.py                # 统一 2D 评估器 (COCO 风格 mAP, 各模型同口径)
├─ eval3d.py                # 3D 评估 (nuScenes 风格 BEV 中心距离 mAP)
├─ tools/                   # 审计/诊断/统一评估入口/可视化
├─ v1.0-mini/               # 原始数据集 (不入库)
└─ outputs/                 # 训练产物 (不入库)
```

## 环境

conda env `dl`：Python + PyTorch (CUDA) + Ultralytics + torchvision。
Windows 下 DataLoader 固定单进程（`num_workers=0`）避免死锁。

## 快速开始

```powershell
# 全链串行训练 (8 模型, 完成的自动跳过, 中断后重跑自动接续)
powershell -File run_chain2.ps1

# 单独训练某个模型
python train_3d.py centerpoint
python train_yolo.py yolo26

# 统一口径评估 + 可视化
python tools/eval_unified.py outputs\yolo11\train\weights\best.pt 6
```

## 当前结果

2D 统一评估（480 张验证图，AP 标准协议）：

| 模型 | mAP@0.5 | car AP50 |
|---|---|---|
| YOLO11n | 0.1617 | 0.557 |
| 其余 7 个模型 | 链式训练中 | — |

已知限制：v1.0-mini 仅 324 训练帧，小目标（远景行人）与稀有类数量受数据量
制约；稀有类学习依赖按类归一化的分类损失（见 `models3d.py` 注释与
`tools/test_overfit*.py` 验证脚本）。
