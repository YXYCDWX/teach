# -*- coding: utf-8 -*-
"""用训练好的 CenterPoint (3D 最优模型) 在 nuScenes LIDAR_TOP 连续帧上逐帧检测,
渲染成 BEV 俯视行车视频。
- 素材: 训练集前 8 个场景的 LIDAR_TOP (keyframe + sweep, 约 12Hz 连续)
- 模型: outputs/centerpoint/best.pth (CenterPoint, 无锚框热力图检测)
- 点云/标注在推理时按帧现场处理 (sweeps 无预转换 npz)
用法: python tools/make_video_3d.py [fps] [score_thr] [max_frames] [ckpt_path]
输出: outputs/videos/centerpoint_bev.mp4
"""
import os
import sys
import time
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from nuscenes_common import (NuScenesTables, CLASS_NAMES, ann_box_in_frame,
                             frames_for_sample_data)
from dataset3d import voxelize
from models3d import PillarBackbone

DATAROOT = os.path.join(BASE, "v1.0-mini")
OUT_DIR = os.path.join(BASE, "outputs", "videos")
CKPT = os.path.join(BASE, "outputs", "centerpoint", "best.pth")
SENSOR = "LIDAR_TOP"
PILLAR_SIZE = (0.5, 0.5, 6.0)
MAX_POINTS = 32
MAX_VOXELS = 18000
POINT_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)  # x1,y1,z1,x2,y2,z2

# 出片呈现参数 (CenterPoint 热力图峰成片, 靠 稀疏化+限框数+类别过滤 控制观感)
NMS_DIST = 2.0    # 同类框最小中心间距 (m), 越大越稀疏
MAX_DET = 20      # 每帧最多保留框数 (按分数)
# 只展示动态目标, 去掉 barrier / traffic_cone 等静止杂波 (None 表示全部展示)
CLASS_FILTER = ["car", "truck", "construction_vehicle", "bus", "trailer",
                "motorcycle", "bicycle", "pedestrian"]


def open_writer(path_no_ext, fps, size):
    """优先 mp4v, 失败回退 avc1 / MJPG(avi), 返回 (writer, 实际路径)"""
    for fourcc, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi")]:
        p = path_no_ext + ext
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            print(f"视频编码 {fourcc} -> {p}", flush=True)
            return w, p
        w.release()
    raise RuntimeError("无法打开视频写入器 (检查 opencv 编码器)")


def build_model(device, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = PillarBackbone().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"载入 {ckpt_path}  (epoch {ckpt.get('epoch')}, mAP={ckpt.get('mAP'):.4f})",
          flush=True)
    return model


def collect_frames(tables):
    """训练集 8 场景的 LIDAR_TOP 连续帧, 按场景与时间戳排序, 返回 (场景名, sd记录)"""
    scenes = sorted(tables.scenes(), key=lambda s: s["name"])[:8]
    frames = []
    for sc in scenes:
        for st in tables.scene_samples(sc["token"]):
            sds = tables.list_sample_data(st, SENSOR, keyframe_only=False)
            sds.sort(key=lambda r: r["timestamp"])
            for r in sds:
                frames.append((sc["name"], r))
    return frames


def load_points_boxes(tables, sd):
    """读取一帧点云与 GT 框 (标注为全局系, 变换到该帧传感器系)"""
    bin_path = os.path.join(DATAROOT, sd["filename"].replace("/", os.sep))
    raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    points = raw[:, :4].astype(np.float32)              # x,y,z,intensity
    fr = frames_for_sample_data(tables, sd)
    x1, y1, z1, x2, y2, z2 = POINT_RANGE
    boxes = []
    for ann in tables.sample_annotations(sd["sample_token"]):
        out = ann_box_in_frame(ann, fr["T_sensor_from_global"],
                               fr["T_global_from_ego"], fr["T_ego_from_sensor"])
        if out is None:
            continue
        center, size, yaw, cls_id = out
        if not (x1 < center[0] < x2 and y1 < center[1] < y2 and z1 < center[2] < z2):
            continue
        if min(size) < 0.2:
            continue
        boxes.append([center[0], center[1], center[2],
                      size[0], size[1], size[2], yaw, cls_id])
    return points, np.asarray(boxes, dtype=np.float32).reshape(-1, 8)


def draw_box(ax, box, color, lw=1.5):
    """BEV 画单个框 box=[cx,cy,cz,w,l,h,yaw], nuScenes: 长 l 沿朝向, 宽 w 横向"""
    cx, cy, w, l, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    lx = np.array([l / 2, l / 2, -l / 2, -l / 2])
    ly = np.array([w / 2, -w / 2, -w / 2, w / 2])
    px = cx + c * lx - s * ly
    py = cy + s * lx + c * ly
    ax.add_patch(Polygon(np.stack([px, py], axis=1), closed=True, fill=False,
                         edgecolor=color, linewidth=lw))
    ax.plot([cx, cx + c * l * 0.6], [cy, cy + s * l * 0.6], color=color,
            linewidth=lw * 0.8)


class BevRenderer:
    """复用同一个 matplotlib figure, 直接取画布像素, 避免反复建图开销"""

    def __init__(self, dpi=90, size=8.0):
        self.fig, self.ax = plt.subplots(figsize=(size, size), dpi=dpi)

    def render(self, points, gt_boxes, pred_boxes, pred_cls, pred_scores, title):
        ax = self.ax
        ax.clear()
        order = np.argsort(-points[:, 3])
        ax.scatter(points[order, 0], points[order, 1], s=0.3,
                   c=-points[order, 2], cmap="gray_r", alpha=0.6)
        for b in gt_boxes:
            draw_box(ax, b, "limegreen", 1.6)
        for j, b in enumerate(pred_boxes):
            draw_box(ax, b, "red", 1.4)
            ax.annotate(f"{CLASS_NAMES[int(pred_cls[j])]} {pred_scores[j]:.2f}",
                        (b[0], b[1]), fontsize=6, color="red")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.invert_yaxis()                 # x 向前, 前方朝上
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x 前 (m)")
        ax.set_ylabel("y 左 (m)")
        self.fig.tight_layout(pad=0.5)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]   # RGB
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def put_text(img, txt, org, scale=0.8):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                2, cv2.LINE_AA)


def main():
    fps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ckpt_path = sys.argv[4] if len(sys.argv) > 4 else CKPT

    device = torch.device("cuda")
    torch.manual_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    tables = NuScenesTables(DATAROOT)
    frames = collect_frames(tables)
    if max_frames:
        frames = frames[:max_frames]
    print(f"共 {len(frames)} 帧 (训练集 LIDAR_TOP 连续帧), conf>={thr}", flush=True)

    model = build_model(device, ckpt_path)
    renderer = BevRenderer()
    writer = None
    out_path = os.path.join(OUT_DIR, "centerpoint_bev")
    t0 = time.time()

    for gi, (sc, sd) in enumerate(frames, 1):
        points, gt_boxes = load_points_boxes(tables, sd)
        v, c, n = voxelize(points, PILLAR_SIZE, MAX_POINTS, MAX_VOXELS)
        v = torch.from_numpy(v).unsqueeze(0).to(device)
        c = torch.from_numpy(c).unsqueeze(0).to(device)
        n = torch.from_numpy(n).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hm, reg_m = model(v, c, n, (200, 200))
            res = model.head.predict(hm.float(), reg_m.float(), score_thr=thr,
                                     nms_dist=NMS_DIST, max_det=MAX_DET)[0]
        pboxes = res["boxes"]
        pcls = res["cls"]
        pscores = res["scores"]
        if CLASS_FILTER is not None:      # 过滤静止杂波类别
            keep_ids = [CLASS_NAMES.index(n) for n in CLASS_FILTER]
            m_keep = np.isin(pcls, keep_ids)
            pboxes, pcls, pscores = pboxes[m_keep], pcls[m_keep], pscores[m_keep]

        title = f"CenterPoint BEV  |  {sc}  |  frame {gi}/{len(frames)}  " \
                f"|  pred {len(pboxes)}"
        vis = renderer.render(points, gt_boxes, pboxes, pcls, pscores, title)
        put_text(vis, f"CenterPoint BEV  {sc}  {gi}/{len(frames)}", (12, 30), 0.7)
        if writer is None:
            h, w = vis.shape[:2]
            writer, out_path = open_writer(out_path, fps, (w, h))
        writer.write(vis)

        if gi % 50 == 0 or gi == len(frames):
            el = time.time() - t0
            print(f"  {gi}/{len(frames)}  用时 {el:.0f}s  "
                  f"({el / gi:.2f}s/帧)", flush=True)

    if writer is not None:
        writer.release()
    print(f"完成: {out_path}  总用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
