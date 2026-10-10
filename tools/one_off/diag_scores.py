# -*- coding: utf-8 -*-
"""裁决实验: 分数标定问题 vs 设计天花板问题
同一份 PointPillars best.pth, 用不同 score_thr 评估:
- 若降阈值后 mAP 大幅回升 -> 预测分数被压低被滤掉 (标定问题, 可修)
- 若 mAP 几乎不动 -> 框本身位置不对 (匹配/数据质量天花板)
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import train_3d as T
from dataset3d import NuScenes3D
from eval3d import evaluate_3d
from models3d import PointPillars


def main():
    device = "cuda"
    stats = T.compute_anchor_stats("train")
    anchors = T.build_anchors(stats, T.GRID_XY, T.GRID_XY)
    val_ds = NuScenes3D("val", mode="pillar", augment=False, max_voxels=18000)
    loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                        num_workers=0, collate_fn=NuScenes3D.collate)
    model = PointPillars(stats).to(device)
    ck = torch.load(os.path.join("outputs", "pointpillars", "best.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    print("权重加载完成 (best.pth)")

    for thr in [0.05, 0.01, 0.002, 0.0]:
        preds, gts, key = {}, {}, 0
        scores_all = []
        with torch.no_grad():
            for v, c, n, gt in loader:
                base = key
                key += len(gt)
                with torch.autocast("cuda", dtype=torch.float16):
                    cls_m, reg_m, dir_m = model(
                        v.to(device), c.to(device), n.to(device),
                        T.get_grid("pointpillars"))
                res = model.head.predict(
                    cls_m.float(), reg_m.float(), dir_m.float(), anchors,
                    score_thr=thr, nms_dist=0.6)
                for b, r in enumerate(res):
                    preds[f"f{base + b}"] = r
                    if len(r["scores"]) and thr == 0.0:
                        scores_all += list(np.asarray(r["scores"])[:50])
                for b, g in enumerate(gt):
                    gts[f"f{base + b}"] = {"cls": g[:, 7].numpy().astype(int),
                                           "boxes": g[:, :7].numpy(),
                                           "scores": np.ones(len(g))}
        m = evaluate_3d(preds, gts)
        head = {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in m.items() if not isinstance(v, dict)}
        line = f"thr={thr:<6} {head}"
        if "per_class" in m:
            line += "  " + str({k: round(v, 4) for k, v in m["per_class"].items()})
        print(line)
        if thr == 0.0 and scores_all:
            s = np.array(scores_all)
            print(f"分数分布(每帧前50框): n={len(s)} max={s.max():.4f} "
                  f"p99={np.percentile(s,99):.4f} p90={np.percentile(s,90):.4f} "
                  f"mean={s.mean():.4f}")


if __name__ == "__main__":
    main()
