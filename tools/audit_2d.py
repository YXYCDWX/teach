# -*- coding: utf-8 -*-
"""审计 nu2d 标签: 用当前 convert_2d 逻辑全量重算, 与磁盘标签逐图比对
输出: 不一致图数/框数, 巨框统计 (w*h>0.15 或 w>0.5 或 h>0.7)
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuscenes_common import (NuScenesTables, ann_box_in_frame, frames_for_sample_data,
                             box_corners)
from convert_2d import DATAROOT, OUT_DIR, MIN_VISIBILITY, MIN_BOX_PX, project_box_to_image

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def current_lines(tables, split_map):
    """按当前代码逻辑重算所有标签 -> {磁盘标签路径: [lines]}"""
    result = {}
    for scene in tables.scenes():
        split = split_map.get(scene["token"])
        if split is None:
            continue
        for sample_token in tables.scene_samples(scene["token"]):
            anns = tables.sample_annotations(sample_token)
            for cam in CAMS:
                sd_list = tables.list_sample_data(sample_token, cam)
                if not sd_list:
                    continue
                sd = sd_list[0]
                fr = frames_for_sample_data(tables, sd)
                K = fr["K"]
                src = os.path.join(DATAROOT, sd["filename"].replace("/", os.sep))
                W, H = Image.open(src).size
                lines = []
                for ann in anns:
                    vis = int(tables.get("visibility", ann["visibility_token"])["level"].split("-")[0][1:])
                    if vis < MIN_VISIBILITY:
                        continue
                    box = ann_box_in_frame(ann, fr["T_sensor_from_global"])
                    if box is None:
                        continue
                    center, size, yaw, cls_id = box
                    if center[2] < 0.1:
                        continue
                    bb = project_box_to_image(center, size, yaw, K)
                    if bb is None:
                        continue
                    x1, y1, x2, y2 = bb
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                        continue
                    lines.append(f"{cls_id} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                                 f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
                base = os.path.splitext(os.path.basename(sd["filename"]))[0]
                lbl = os.path.join(OUT_DIR, "labels", split, f"{scene['name']}_{base}.txt")
                result[lbl] = lines
    return result


def main():
    tables = NuScenesTables(DATAROOT)
    scenes = sorted(tables.scenes(), key=lambda s: s["name"])
    split_map = {s["token"]: ("train" if i < 8 else "val") for i, s in enumerate(scenes)}

    print("[1/3] 全量重算当前逻辑标签 ...")
    expect = current_lines(tables, split_map)

    print("[2/3] 与磁盘标签比对 ...")
    n_img_diff = 0
    n_box_disk = n_box_expect = 0
    diff_examples = []
    giant = []
    for lbl_path, lines in expect.items():
        if not os.path.exists(lbl_path):
            n_img_diff += 1
            diff_examples.append(("缺失", lbl_path))
            continue
        disk = [l.strip() for l in open(lbl_path, encoding="utf-8") if l.strip()]
        n_box_disk += len(disk)
        n_box_expect += len(lines)
        if sorted(disk) != sorted(lines):
            n_img_diff += 1
            if len(diff_examples) < 8:
                diff_examples.append((f"磁盘{len(disk)}框/重算{len(lines)}框", lbl_path))
    print(f"  比对图像数: {len(expect)}")
    print(f"  磁盘总框数: {n_box_disk}, 当前逻辑重算总框数: {n_box_expect}")
    print(f"  内容不一致的图: {n_img_diff}")
    for tag, p in diff_examples:
        print(f"    [{tag}] {os.path.basename(p)}")

    print("[3/3] 磁盘标签巨框统计 (w*h>0.15 或 w>0.5 或 h>0.7) ...")
    cnt = {}
    total = 0
    for split in ["train", "val"]:
        d = os.path.join(OUT_DIR, "labels", split)
        for f in os.listdir(d):
            for l in open(os.path.join(d, f), encoding="utf-8"):
                p = l.split()
                if len(p) != 5:
                    continue
                c = int(p[0]); w = float(p[3]); h = float(p[4])
                if w * h > 0.15 or w > 0.5 or h > 0.7:
                    total += 1
                    cnt[c] = cnt.get(c, 0) + 1
    from nuscenes_common import CLASS_NAMES
    print(f"  巨框总数: {total}")
    for c, n in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"    {CLASS_NAMES[c]}: {n}")


if __name__ == "__main__":
    main()
