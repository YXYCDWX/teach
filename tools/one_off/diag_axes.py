# -*- coding: utf-8 -*-
"""决定性验证: LiDAR 点投影 + 两种 w/l 轴向约定的 3D 框投影叠加
红=当前约定 (w 沿局部 x, l 沿局部 y)  蓝=备选约定 (l 沿局部 x, w 沿局部 y)
白点=LiDAR 点云投影
"""
import os
import sys
import math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nuscenes_common import (NuScenesTables, frames_for_sample_data, ann_box_in_frame,
                             box_corners, quaternion_to_matrix, make_transform, CATEGORY_MAP)

DATAROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "v1.0-mini")
TARGET = "n015-2018-11-21-19-38-26+0800__CAM_FRONT_RIGHT__1542800850920339.jpg"

T = NuScenesTables(DATAROOT)
sd_cam = next(s for s in T.tables["sample_data"] if s["filename"].endswith(TARGET))
sample_token = sd_cam["sample_token"]
fr = frames_for_sample_data(T, sd_cam)
K = fr["K"]

# --- LiDAR 同 sample 关键帧 -> 相机系 ---
sd_lidar = T.list_sample_data(sample_token, "LIDAR_TOP")[0]
fl = frames_for_sample_data(T, sd_lidar)
pts = np.fromfile(os.path.join(DATAROOT, sd_lidar["filename"].replace("/", os.sep)),
                  dtype=np.float32).reshape(-1, 5)[:, :3]
pts_glob = pts @ fl["T_global_from_ego"][:3, :3].T + fl["T_global_from_ego"][:3, 3]
pts_cam = pts_glob @ fr["T_sensor_from_global"][:3, :3].T + fr["T_sensor_from_global"][:3, 3]

img = cv2.imread(os.path.join(DATAROOT, sd_cam["filename"].replace("/", os.sep)))
P = (pts_cam @ K.T)
ok = P[:, 2] > 0.5
uv = P[ok][:, :2] / P[ok][:, 2:3]
inside = (uv[:, 0] >= 0) & (uv[:, 0] < 1600) & (uv[:, 1] >= 0) & (uv[:, 1] < 900)
for u, v in uv[inside].astype(int):
    cv2.circle(img, (u, v), 1, (255, 255, 255), -1)

def corners_swap(center, size, yaw):
    """备选约定: l 沿局部 x, w 沿局部 y"""
    w, l, h = size
    return box_corners(center, (l, w, h), yaw)

def draw(img, cs, color):
    xs, ys = [], []
    for c3 in cs:
        if c3[2] <= 0.1:
            return None
        p = c3 @ K.T
        xs.append(p[0] / p[2]); ys.append(p[1] / p[2])
    for i in range(8):
        j = (i + 1) % 4 if i % 4 != 3 else i - 3
        cv2.line(img, (int(xs[i]), int(ys[i])), (int(xs[j]), int(ys[j])), color, 2)
        if i < 4:
            cv2.line(img, (int(xs[i]), int(ys[i])), (int(xs[i+4]), int(ys[i+4])), color, 1)
    return min(xs), min(ys), max(xs), max(ys)

n_car = 0
for ann in T.sample_annotations(sample_token):
    if CATEGORY_MAP.get(ann["category_name"], "") != "car":
        continue
    box = ann_box_in_frame(ann, fr["T_sensor_from_global"],
                           fr["T_global_from_ego"], fr["T_ego_from_sensor"])
    if box is None:
        continue
    center, size, yaw, cls_id = box
    if center[2] < 0.1:
        continue
    n_car += 1
    bb_a = draw(img, box_corners(center, size, yaw), (0, 0, 255))
    bb_b = draw(img, corners_swap(center, size, yaw), (255, 128, 0))
    print(f"car#{n_car} size(w,l,h)={size} yaw={math.degrees(yaw):.1f} z={center[2]:.1f}")
    print(f"  红框(当前 w->x): {None if bb_a is None else np.round(bb_a,0)}")
    print(f"  蓝框(备选 l->x): {None if bb_b is None else np.round(bb_b,0)}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "diag_axes.jpg")
cv2.imwrite(out, img)
print("saved:", out)
