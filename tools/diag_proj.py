# -*- coding: utf-8 -*-
"""诊断: 复现某验证图 car 标注的 3D->2D 投影全过程"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuscenes_common import NuScenesTables, frames_for_sample_data, ann_box_in_frame, box_corners, CATEGORY_MAP

DATAROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v1.0-mini")
TARGET = "n015-2018-11-21-19-38-26+0800__CAM_FRONT_RIGHT__1542800850920339.jpg"

T = NuScenesTables(DATAROOT)
sd = next(s for s in T.tables["sample_data"] if s["filename"].endswith(TARGET))
sample_token = sd["sample_token"]
fr = frames_for_sample_data(T, sd)
K = fr["K"]

for ann in T.sample_annotations(sample_token):
    if CATEGORY_MAP.get(ann["category_name"], "") is None:
        continue
    box = ann_box_in_frame(ann, fr["T_sensor_from_global"],
                           fr["T_global_from_ego"], fr["T_ego_from_sensor"])
    if box is None:
        continue
    center, size, yaw, cls_id = box
    vis = int(T.get("visibility", ann["visibility_token"])["level"].split("-")[0][1:])
    if vis < 2 or center[2] < 0.1:
        print(f"cls={cls_id} size={size} center_cam=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})"
              f" vis={vis} -> 跳过(可见性/中心深度)")
        continue
    corners = box_corners(center, size, yaw)
    print(f"cls={cls_id} size(w,l,h)={size} yaw={math.degrees(yaw):.1f}deg vis={vis}")
    print(f"  center_cam = x={center[0]:7.2f} y={center[1]:7.2f} z={center[2]:7.2f}")
    print(f"  corner depths = {np.round(corners[:,2],2)}")
    uv = []
    for c3 in corners:
        if c3[2] > 0.1:
            p = c3 @ K.T
            uv.append(p[:2] / p[2])
    uv = np.array(uv)
    print(f"  有效角点 {len(uv)}/8, u {uv[:,0].min():.0f}..{uv[:,0].max():.0f},"
          f" v {uv[:,1].min():.0f}..{uv[:,1].max():.0f}")
    x1, y1 = max(0, uv[:,0].min()), max(0, uv[:,1].min())
    x2, y2 = min(1600, uv[:,0].max()), min(900, uv[:,1].max())
    keep = x2 - x1 >= 4 and y2 - y1 >= 4
    print(f"  裁剪后 2D 框: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) -> {'保留' if keep else '丢弃'}"
          f"  归一化 cx={(x1+x2)/2/1600:.3f} cy={(y1+y2)/2/900:.3f} w={(x2-x1)/1600:.3f} h={(y2-y1)/900:.3f}")
