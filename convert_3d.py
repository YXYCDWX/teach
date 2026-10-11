# -*- coding: utf-8 -*-
"""nuScenes v1.0-mini -> 3D 点云检测数据集
- 每个 keyframe 的 LIDAR_TOP 点云 (x,y,z,intensity)
- 3D 标注框变换到 LiDAR 传感器坐标系
- 按场景划分训练/验证 (与 2D 一致: 前 8 场景训练, 后 2 验证)
- 训练集额外并入 sweeps (约 50ms 一帧), 用标注 prev/next 链按时间戳线性插值,
  把训练帧从 ~320 扩到 ~2900, 缓解 3D 数据饥饿
- 验证集保持 keyframe 不变, 保证与既有指标口径可比
输出: nu3d/{train,val}/*.npz  (points: N x 4, boxes: M x 8 [cx,cy,cz,w,l,h,yaw,cls])
"""
import os
import sys
import bisect
import shutil
import math
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from nuscenes_common import (NuScenesTables, ann_box_in_frame, frames_for_sample_data,
                             quaternion_to_matrix, CLSID, CATEGORY_MAP)

DATAROOT = os.path.join(BASE, "v1.0-mini")
OUT_DIR = os.path.join(BASE, "nu3d")
POINT_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)  # x1,y1,z1,x2,y2,z2
INCLUDE_SWEEPS = True   # 训练集是否并入 sweeps


def build_timelines(tables):
    """每条 instance 的标注时间线: {inst_token: {ts:(N,), tr:(N,3), rot:[...], sz:(N,3), cls}}"""
    sample_ts = {s["token"]: s["timestamp"] for s in tables.tables["sample"]}
    tmp = {}
    for a in tables.tables["sample_annotation"]:
        cls_id = CLSID.get(CATEGORY_MAP.get(a["category_name"], ""), None)
        if cls_id is None:
            continue
        tmp.setdefault(a["instance_token"], []).append(
            (sample_ts[a["sample_token"]], np.asarray(a["translation"], dtype=np.float64),
             a["rotation"], np.asarray(a["size"], dtype=np.float64), cls_id))
    tl = {}
    for inst, rows in tmp.items():
        rows.sort(key=lambda x: x[0])
        tl[inst] = {
            "ts": [int(r[0]) for r in rows],
            "tr": np.stack([r[1] for r in rows]),
            "rot": [r[2] for r in rows],
            "sz": np.stack([r[3] for r in rows]),
            "cls": rows[0][4],
        }
    return tl


def pose_at(tl, t):
    """返回 t 时刻该 instance 的 (translation(3,), rotation, size(3,)), 不在其生命周期内返回 None"""
    ts = tl["ts"]
    if t < ts[0] or t > ts[-1]:
        return None
    i = bisect.bisect_left(ts, int(t))
    if i < len(ts) and ts[i] == t:
        return tl["tr"][i], tl["rot"][i], tl["sz"][i]
    if i == 0:
        return tl["tr"][0], tl["rot"][0], tl["sz"][0]
    if i >= len(ts):
        return tl["tr"][-1], tl["rot"][-1], tl["sz"][-1]
    a, b = i - 1, i
    w = (t - ts[a]) / float(ts[b] - ts[a])
    tr = (1.0 - w) * tl["tr"][a] + w * tl["tr"][b]
    near = a if (t - ts[a]) <= (ts[b] - t) else b
    return tr, tl["rot"][near], tl["sz"][near]


def sweep_boxes(tables, sd, timelines, rng):
    """sweep 帧: 用时间插值标注生成 3D 框 (全局位姿 -> 该帧传感器系)"""
    fr = frames_for_sample_data(tables, sd)
    T = fr["T_sensor_from_global"]
    R_inv, t_inv = T[:3, :3], T[:3, 3]
    x1, y1, z1, x2, y2, z2 = rng
    t = int(sd["timestamp"])
    boxes = []
    for tl in timelines.values():
        pose = pose_at(tl, t)
        if pose is None:
            continue
        tr, rot, size = pose
        center = R_inv @ tr + t_inv
        R_local = R_inv @ quaternion_to_matrix(rot)
        yaw = math.atan2(R_local[1, 0], R_local[0, 0])
        if not (x1 < center[0] < x2 and y1 < center[1] < y2 and z1 < center[2] < z2):
            continue
        if min(size) < 0.2:
            continue
        boxes.append([center[0], center[1], center[2],
                      size[0], size[1], size[2], yaw, tl["cls"]])
    return boxes


def keyframe_boxes(tables, sample_token, fr, rng):
    """keyframe 帧: 直接用该 sample 的标注 (既有逻辑, 保持不变)"""
    x1, y1, z1, x2, y2, z2 = rng
    boxes = []
    for ann in tables.sample_annotations(sample_token):
        box = ann_box_in_frame(ann, fr["T_sensor_from_global"],
                               fr["T_global_from_ego"], fr["T_ego_from_sensor"])
        if box is None:
            continue
        center, size, yaw, cls_id = box
        if not (x1 < center[0] < x2 and y1 < center[1] < y2 and z1 < center[2] < z2):
            continue
        if min(size) < 0.2:
            continue
        boxes.append([center[0], center[1], center[2],
                      size[0], size[1], size[2], yaw, cls_id])
    return boxes


def save_frame(out_split, name, points, boxes):
    np.savez_compressed(os.path.join(OUT_DIR, out_split, name + ".npz"),
                        points=points,
                        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 8))


def main():
    # 重跑前清空旧产物, 防止逻辑变更后残留陈旧样本
    for split in ["train", "val"]:
        p = os.path.join(OUT_DIR, split)
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)
    tables = NuScenesTables(DATAROOT)
    timelines = build_timelines(tables)
    scenes = sorted(tables.scenes(), key=lambda s: s["name"])
    train_scenes = {s["token"] for s in scenes[:8]}
    print("共 %d 场景: 训练 %d, 验证 %d  (instance %d 条)"
          % (len(scenes), len(train_scenes), len(scenes) - len(train_scenes),
             len(timelines)))

    rng = POINT_RANGE
    stats = {"train": [0, 0], "val": [0, 0]}  # [样本数, 框数]

    for si, scene in enumerate(scenes):
        split = "train" if scene["token"] in train_scenes else "val"
        use_sweeps = INCLUDE_SWEEPS and split == "train"   # 仅在训练集并入 sweeps
        n_frame = 0
        for sample_token in tables.scene_samples(scene["token"]):
            lsd_list = tables.list_sample_data(sample_token, "LIDAR_TOP",
                                               keyframe_only=not use_sweeps)
            if not lsd_list:
                continue
            lsd_list.sort(key=lambda r: r["timestamp"])
            for lsd in lsd_list:
                bin_path = os.path.join(DATAROOT, lsd["filename"].replace("/", os.sep))
                raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
                points = raw[:, :4].astype(np.float32)   # x,y,z,intensity

                if lsd["is_key_frame"]:
                    fr = frames_for_sample_data(tables, lsd)
                    boxes = keyframe_boxes(tables, sample_token, fr, rng)
                else:
                    boxes = sweep_boxes(tables, lsd, timelines, rng)

                name = f"{scene['name']}_{lsd['token'][:8]}"
                save_frame(split, name, points, boxes)
                stats[split][0] += 1
                stats[split][1] += len(boxes)
                n_frame += 1
        print(f"[{si+1}/{len(scenes)}] {scene['name']} 完成 ({split}, {n_frame} 帧)")

    print("转换完成:")
    print(f"  训练: {stats['train'][0]} 帧, {stats['train'][1]} 框")
    print(f"  验证: {stats['val'][0]} 帧, {stats['val'][1]} 框")
    print(f"  点云范围: {POINT_RANGE}  (训练并入 sweeps: {INCLUDE_SWEEPS})")


if __name__ == "__main__":
    main()
