# -*- coding: utf-8 -*-
"""nuScenes v1.0-mini -> 3D 点云检测数据集
- 每个 keyframe 的 LIDAR_TOP 点云 (x,y,z,intensity)
- 3D 标注框变换到 LiDAR 传感器坐标系
- 按场景划分训练/验证 (与 2D 一致: 前 8 场景训练, 后 2 验证)
输出: nu3d/{train,val}/*.npz  (points: N x 4, boxes: M x 8 [cx,cy,cz,w,l,h,yaw,cls])
"""
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from nuscenes_common import (NuScenesTables, ann_box_in_frame, frames_for_sample_data,
                             CLSID)

DATAROOT = os.path.join(BASE, "v1.0-mini")
OUT_DIR = os.path.join(BASE, "nu3d")
POINT_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)  # x1,y1,z1,x2,y2,z2


def main():
    tables = NuScenesTables(DATAROOT)
    scenes = sorted(tables.scenes(), key=lambda s: s["name"])
    train_scenes = {s["token"] for s in scenes[:8]}
    print("共 %d 场景: 训练 %d, 验证 %d" % (len(scenes), len(train_scenes), len(scenes) - len(train_scenes)))

    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

    x1, y1, z1, x2, y2, z2 = POINT_RANGE
    stats = {"train": [0, 0], "val": [0, 0]}  # [样本数, 框数]

    for si, scene in enumerate(scenes):
        split = "train" if scene["token"] in train_scenes else "val"
        for sample_token in tables.scene_samples(scene["token"]):
            lsd = tables.list_sample_data(sample_token, "LIDAR_TOP")
            if not lsd:
                continue
            lsd = lsd[0]
            fr = frames_for_sample_data(tables, lsd)
            # 读取点云: float32 x,y,z,intensity,ring
            bin_path = os.path.join(DATAROOT, lsd["filename"].replace("/", os.sep))
            raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
            points = raw[:, :4].astype(np.float32)  # x,y,z,intensity

            boxes = []
            for ann in tables.sample_annotations(sample_token):
                box = ann_box_in_frame(ann, fr["T_sensor_from_global"],
                                       fr["T_global_from_ego"], fr["T_ego_from_sensor"])
                if box is None:
                    continue
                center, size, yaw, cls_id = box
                # 只保留点云范围内且尺寸合理的框
                if not (x1 < center[0] < x2 and y1 < center[1] < y2 and z1 < center[2] < z2):
                    continue
                if min(size) < 0.2:
                    continue
                boxes.append([center[0], center[1], center[2],
                              size[0], size[1], size[2], yaw, cls_id])

            name = f"{scene['name']}_{lsd['token'][:8]}"
            np.savez_compressed(os.path.join(OUT_DIR, split, name + ".npz"),
                                points=points,
                                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 8))
            stats[split][0] += 1
            stats[split][1] += len(boxes)
        print(f"[{si+1}/{len(scenes)}] {scene['name']} 完成 ({split})")

    print("转换完成:")
    print(f"  训练: {stats['train'][0]} 帧, {stats['train'][1]} 框")
    print(f"  验证: {stats['val'][0]} 帧, {stats['val'][1]} 框")
    print(f"  点云范围: {POINT_RANGE}")


if __name__ == "__main__":
    main()
