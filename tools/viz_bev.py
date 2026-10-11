# -*- coding: utf-8 -*-
"""BEV 鸟瞰图可视化: 点云 + 真值框(绿) + 预测框(红)
用法: python tools/viz_bev.py [帧数]
输出: outputs/viz_bev/centerpoint_bev_XX.jpg
"""
import os
import sys
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
from dataset3d import NuScenes3D
from models3d import PillarBackbone
from nuscenes_common import CLASS_NAMES

OUT_DIR = os.path.join(BASE, "outputs", "viz_bev")
MODE = "centerpoint"


def draw_box(ax, box, color, lw=1.5, label=None):
    """BEV 画单个框 box=[cx,cy,cz,w,l,h,yaw], nuScenes: 长 l 沿朝向, 宽 w 横向"""
    cx, cy, w, l, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    # 局部角点: x 轴 ±l/2 (朝向), y 轴 ±w/2 (横向)
    lx = np.array([l/2, l/2, -l/2, -l/2])
    ly = np.array([w/2, -w/2, -w/2, w/2])
    px = cx + c*lx - s*ly
    py = cy + s*lx + c*ly
    ax.add_patch(Polygon(np.stack([px, py], axis=1), closed=True, fill=False,
                         edgecolor=color, linewidth=lw, label=label))
    # 朝向短线
    ax.plot([cx, cx + c*l*0.6], [cy, cy + s*l*0.6], color=color, linewidth=lw*0.8)


def main():
    n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    device = torch.device("cuda")

    ds = NuScenes3D("val", augment=False, max_voxels=18000)
    ckpt_path = os.path.join(BASE, "outputs", MODE, "best.pth")
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = PillarBackbone().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"载入 {ckpt_path}  (epoch {ckpt.get('epoch')}, mAP={ckpt.get('mAP'):.4f})")

    # 选真值框最多的前 n_show 帧, 保证图里有目标可看
    n_boxes = []
    for i in range(len(ds)):
        d = np.load(os.path.join(ds.root, ds.files[i]))
        n_boxes.append(len(d["boxes"]))
    pick = np.argsort(-np.asarray(n_boxes))[:n_show]

    os.makedirs(OUT_DIR, exist_ok=True)
    for rank, i in enumerate(pick):
        v, c, n, gt = ds[int(i)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hm, reg_m = model(v.unsqueeze(0).to(device),
                              c.unsqueeze(0).to(device),
                              n.unsqueeze(0).to(device), (200, 200))
            res = model.head.predict(hm.float(), reg_m.float(),
                                     score_thr=0.3, nms_dist=0.6)[0]

        d = np.load(os.path.join(ds.root, ds.files[int(i)]))
        pts = d["points"]           # (N,4)
        gtb = d["boxes"]            # (M,8)

        fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
        order = np.argsort(-pts[:, 3])
        sc = ax.scatter(pts[order, 0], pts[order, 1], s=0.3, c=-pts[order, 2],
                        cmap="gray_r", alpha=0.6)
        for b in gtb:
            draw_box(ax, b, "limegreen", 1.8, "GT" if b is gtb[0] else None)
        n_pred = len(res["boxes"])
        for j, b in enumerate(res["boxes"]):
            name = CLASS_NAMES[res["cls"][j]]
            draw_box(ax, b, "red", 1.5,
                     f"pred" if j == 0 else None)
            ax.annotate(f"{name} {res['scores'][j]:.2f}",
                        (b[0], b[1]), fontsize=6, color="red")
        ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
        ax.set_aspect("equal"); ax.invert_yaxis()  # x 向前, 图像习惯: 前方朝上
        ax.set_title(f"{MODE} BEV  {ds.files[int(i)]}\n"
                     f"GT {len(gtb)} 框 (绿)  预测 {n_pred} 框 (红, conf>=0.3)")
        ax.set_xlabel("x 前 (m)"); ax.set_ylabel("y 左 (m)")
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"{MODE}_bev_{rank:02d}.jpg")
        fig.savefig(out)
        plt.close(fig)
        gt_cls = [CLASS_NAMES[int(b[7])] for b in gtb]
        pr_cls = [f"{CLASS_NAMES[res['cls'][j]]}:{res['scores'][j]:.2f}"
                  for j in range(min(5, n_pred))]
        print(f"[{rank}] {ds.files[int(i)]}  GT({len(gtb)}): {gt_cls}")
        print(f"    预测({n_pred}): {pr_cls}")
    print(f"已保存 {n_show} 张到 {OUT_DIR}")


if __name__ == "__main__":
    main()
