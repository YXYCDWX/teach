# -*- coding: utf-8 -*-
"""过拟合自检: 只用 8 帧反复训练, 在这 8 帧上测能否复现 GT
若过拟合都失败 -> 模型/代码存在 bug (与数据量无关)
若过拟合成功 -> 说明代码链路通, 差距来自训练充分度与数据规模
用法: python tools/test_overfit.py
"""
import os
import sys
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from dataset3d import NuScenes3D, compute_anchor_stats
from models3d import PointPillars, build_anchors
from nuscenes_common import CLASS_NAMES

N_FRAMES = 8
ITERS = 200

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda")

stats = compute_anchor_stats("train")
anchors = build_anchors(stats, 200, 200)
ds = NuScenes3D("train", mode="pillar", augment=False, max_voxels=18000)

# 选 car 框最多的 8 帧 (近距稠密目标为主)
cnt = []
for i in range(len(ds)):
    d = np.load(os.path.join(ds.root, ds.files[i]))
    cnt.append((int((d["boxes"][:, 7] == 0).sum()), i))
cnt.sort(reverse=True)
pick = [i for _, i in cnt[:N_FRAMES]]
batch = [ds[i] for i in pick]
v = torch.stack([b[0] for b in batch]).to(device)
c = torch.stack([b[1] for b in batch]).to(device)
n = torch.stack([b[2] for b in batch]).to(device)
gts = [b[3] for b in batch]

model = PointPillars(stats).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(f"用 {N_FRAMES} 帧反复训练 {ITERS} 轮 (过拟合自检)...")
model.train()
for it in range(1, ITERS + 1):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        cls_m, reg_m, dir_m = model(v, c, n, (200, 200))
    cls_l, reg_l, dir_l = model.head.loss(
        cls_m.float(), reg_m.float(), dir_m.float(), anchors, gts)
    loss = cls_l + reg_l + dir_l
    loss.backward()
    opt.step()
    if it % 50 == 0:
        print(f"  iter {it}/{ITERS}  cls={cls_l.item():.4f} reg={reg_l.item():.4f} dir={dir_l.item():.4f}")

# 在训练用的这 8 帧上做预测 (过拟合成功的话应接近全检出)
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    cls_m, reg_m, dir_m = model(v, c, n, (200, 200))
res = model.head.predict(cls_m.float(), reg_m.float(), dir_m.float(),
                         anchors, score_thr=0.3, nms_dist=0.6)

total_gt, total_hit = 0, 0
per_cls = {}
for bi, (r, gt) in enumerate(zip(res, gts)):
    g = gt.numpy()
    used = np.zeros(len(r["boxes"]), dtype=bool)
    for b in g:
        total_gt += 1
        cls_ = int(b[7])
        d = np.linalg.norm(r["boxes"][:, :2] - b[:2], axis=1)
        d = np.where(used | (r["cls"] != cls_), 1e9, d)
        j = int(np.argmin(d)) if len(d) else -1
        hit = j >= 0 and d[j] <= 1.0
        if hit:
            used[j] = True
            total_hit += 1
        a = per_cls.setdefault(cls_, [0, 0])
        a[0] += 1
        a[1] += hit
    if bi == 0:
        names = [f"{CLASS_NAMES[r['cls'][k]]}:{r['scores'][k]:.2f}"
                 for k in range(min(8, len(r["boxes"])))]
        print(f"  帧0 预测 {len(r['boxes'])} 框, 前8: {names}")
print(f"\n结果: 训练帧上检出 {total_hit}/{total_gt} ({total_hit/max(total_gt,1)*100:.0f}%)")
for cls_ in sorted(per_cls):
    a, b_ = per_cls[cls_]
    print(f"  {CLASS_NAMES[cls_]:12s} {b_}/{a} ({b_/a*100:.0f}%)")

# 诊断: 非车类是没打分, 还是分数低于阈值?
res_low = model.head.predict(cls_m.float(), reg_m.float(), dir_m.float(),
                             anchors, score_thr=0.02, nms_dist=0.6)
print("\n各类别在训练帧上的最高置信度 (score_thr=0.02 重解码):")
for cls_ in sorted(per_cls):
    mx, cnt_near = 0.0, 0
    for r, gt in zip(res_low, gts):
        g = gt.numpy()
        m = (r["cls"] == cls_) & (r["scores"] > mx)
        mx = float(r["scores"][m].max()) if m.any() else mx
        for b in g[g[:, 7] == cls_]:
            d = np.linalg.norm(r["boxes"][:, :2] - b[:2], axis=1)
            cnt_near += int(((r["cls"] == cls_) & (d <= 1.5)).sum())
    print(f"  {CLASS_NAMES[cls_]:12s} 最高分 {mx:.3f}  GT 1.5m 内同类预测 {cnt_near} 个")
print("\n判定: 检出率>=90% -> 代码链路正常, 问题在训练充分度/数据规模;")
print("      检出率低   -> 模型或代码存在待查 bug")
