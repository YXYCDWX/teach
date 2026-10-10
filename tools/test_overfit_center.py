# -*- coding: utf-8 -*-
"""过拟合自检 (CenterPoint 无锚框头): 只用 8 帧反复训练, 在这 8 帧上测能否复现 GT
与 tools/test_overfit.py (锚框头) 同一验证口径, 便于对比两种头
用法: python tools/test_overfit_center.py
"""
import os
import sys
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from dataset3d import NuScenes3D, compute_anchor_stats
from models3d import PointPillars
from nuscenes_common import CLASS_NAMES

N_FRAMES = 8
ITERS = 400

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda")

stats = compute_anchor_stats("train")
ds = NuScenes3D("train", mode="pillar", augment=False, max_voxels=18000)

# 选 car 框最多的 8 帧 (与锚框头自检同一批帧)
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

model = PointPillars(stats, head="center").to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
print(f"用 {N_FRAMES} 帧反复训练 {ITERS} 轮 (CenterPoint 无锚框头过拟合自检)...")
model.train()
for it in range(1, ITERS + 1):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        hm, reg_m = model(v, c, n, (200, 200))
    cls_l, reg_l = model.head.loss(hm.float(), reg_m.float(), gts)
    loss = cls_l + reg_l
    loss.backward()
    opt.step()
    if it % 100 == 0:
        with torch.no_grad():
            mx = hm.sigmoid().max().item()
        print(f"  iter {it}/{ITERS}  cls={cls_l.item():.4f} reg={reg_l.item():.4f} "
              f"热力图最高分={mx:.3f}", flush=True)

# 在训练用的这 8 帧上做预测
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    hm, reg_m = model(v, c, n, (200, 200))
res = model.head.predict(hm.float(), reg_m.float(), score_thr=0.3, nms_dist=0.6)

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
print("\n判定: 检出率>=90% -> 无锚框头代码链路正常; 检出率低 -> 待查 bug")
