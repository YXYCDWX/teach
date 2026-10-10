# -*- coding: utf-8 -*-
"""nuScenes v1.0-mini -> YOLO 2D 检测数据集转换
- 6 个相机 keyframe 图像, 将 3D 框投影到图像得到 2D 框
- 按场景划分训练/验证集 (mini 共 10 个场景: 前 8 训练, 后 2 验证)
- 输出 Ultralytics YOLO 格式: images/{train,val}, labels/{train,val}, data.yaml
"""
import os
import sys
import shutil
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nuscenes_common import (NuScenesTables, ann_box_in_frame, frames_for_sample_data,
                             box_corners, CLSID)

DATAROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1.0-mini")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nu2d")
MIN_VISIBILITY = 2   # visibility token 0-3, >=2 表示遮挡程度不超过 60%, 过滤几乎不可见目标
MIN_BOX_PX = 4       # 投影后 2D 框最小像素尺寸


def project_box_to_image(center, size, yaw, K):
    """3D 框角点投影, 返回 (u1,v1,u2,v2) 或 None
    要求全部 8 个角点在相机前方 (z > 0.5m): 角点位于相机平面附近时,
    近零深度会使投影坐标爆炸, 产生横跨整幅图像的错误巨框, 故直接丢弃该框
    """
    corners = box_corners(center, size, yaw)          # (8,3) 传感器系
    if corners[:, 2].min() <= 0.5:                    # 任一角点距相机过近 -> 不可靠
        return None
    pts = corners @ K.T
    uv = pts[:, :2] / pts[:, 2:3]
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    if x2 - x1 < 1e-6 or y2 - y1 < 1e-6:
        return None
    return x1, y1, x2, y2


def main():
    tables = NuScenesTables(DATAROOT)
    scenes = sorted(tables.scenes(), key=lambda s: s["name"])
    train_scenes = {s["token"] for s in scenes[:8]}
    val_scenes = {s["token"] for s in scenes[8:]}
    print("共 %d 个场景: 训练 %d, 验证 %d" % (len(scenes), len(train_scenes), len(val_scenes)))

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(OUT_DIR, sub.replace("/", os.sep)), exist_ok=True)

    stats = {"train": 0, "val": 0}       # 图像数
    box_stats = {"train": 0, "val": 0}   # 标注框数
    skipped_imgs = 0

    for scene in scenes:
        split = "train" if scene["token"] in train_scenes else "val"
        for sample_token in tables.scene_samples(scene["token"]):
            sample = tables.get("sample", sample_token)
            anns = tables.sample_annotations(sample_token)
            # 遍历 6 个相机
            for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
                sd_list = tables.list_sample_data(sample_token, cam)
                if not sd_list:
                    continue
                sd = sd_list[0]
                fr = frames_for_sample_data(tables, sd)
                K = fr["K"]
                filename = sd["filename"]  # samples/CAM_xxx/xxx.jpg
                src = os.path.join(DATAROOT, filename.replace("/", os.sep))
                img = Image.open(src)
                W, H = img.size

                lines = []
                for ann in anns:
                    vis = int(tables.get("visibility", ann["visibility_token"])["level"].split("-")[0][1:])
                    if vis < MIN_VISIBILITY:
                        continue
                    box = ann_box_in_frame(ann, fr["T_sensor_from_global"],
                                           fr["T_global_from_ego"], fr["T_ego_from_sensor"])
                    if box is None:
                        continue
                    center, size, yaw, cls_id = box
                    # 框中心必须在相机前方
                    if center[2] < 0.1:
                        continue
                    bb = project_box_to_image(center, size, yaw, K)
                    if bb is None:
                        continue
                    x1, y1, x2, y2 = bb
                    # 裁剪到图像范围
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                        continue
                    # YOLO 归一化 cx, cy, w, h
                    cx = (x1 + x2) / 2 / W
                    cy = (y1 + y2) / 2 / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                base = os.path.splitext(os.path.basename(filename))[0]
                dst_img = os.path.join(OUT_DIR, "images", split, f"{scene['name']}_{base}.jpg")
                dst_lbl = os.path.join(OUT_DIR, "labels", split, f"{scene['name']}_{base}.txt")
                shutil.copyfile(src, dst_img)
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                stats[split] += 1
                box_stats[split] += len(lines)
                if not lines:
                    skipped_imgs += 1

    # 写 data.yaml
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT_DIR}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(CLSID)}\nnames:\n")
        for n in CLSID:
            f.write(f"  - {n}\n")

    print("转换完成:")
    print(f"  训练: {stats['train']} 图, {box_stats['train']} 框")
    print(f"  验证: {stats['val']} 图, {box_stats['val']} 框")
    print(f"  无标注图像: {skipped_imgs}")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
