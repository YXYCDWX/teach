# -*- coding: utf-8 -*-
"""一次性性能剖析: 定位 rcnn3d 单步耗时构成 (CPU 瓶颈在哪)"""
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from dataset3d import NuScenes3D, compute_anchor_stats
from models3d import PointPillars, build_anchors
from train_rcnn3d import get_proposals, boxes_to_rois, RefineHead, GRID_XY

device = torch.device("cuda")
torch.manual_seed(0)

anchor_stats = compute_anchor_stats("train")
anchors = build_anchors(anchor_stats, GRID_XY, GRID_XY)

ds = NuScenes3D("train", mode="pillar", max_voxels=18000)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                    collate_fn=NuScenes3D.collate)

pp = PointPillars(anchor_stats).to(device)
ck = torch.load(os.path.join(BASE, "outputs", "pointpillars", "best.pth"),
                map_location="cpu", weights_only=False)
pp.load_state_dict(ck["model"])
pp.eval()
for p in pp.parameters():
    p.requires_grad_(False)
refine = RefineHead().to(device)

from torchvision.ops import roi_align

v, c, n, gt = next(iter(loader))
v, c, n = v.to(device), c.to(device), n.to(device)
print("batch:", v.shape, c.shape, n.shape)

# 1) 骨干前向
torch.cuda.synchronize(); t = time.time()
with torch.autocast("cuda", dtype=torch.float16):
    _, _, _, u = pp(v, c, n, (GRID_XY, GRID_XY), return_feat=True)
u = u.float()
torch.cuda.synchronize(); print(f"pp forward: {time.time()-t:.3f}s")

# 2) 候选生成 (含 CPU NMS) —— 重点
t = time.time()
props = get_proposals(pp, u, anchors, device)
print(f"get_proposals: {time.time()-t:.3f}s")
for i, p in enumerate(props):
    print(f"  sample{i}: boxes={len(p['boxes'])} "
          f"score_max={p['scores'].max().item():.4f}")

# 2b) 单独量 score_thr 过滤后的数量
with torch.no_grad():
    cls_m, reg_m, dir_m = pp.head(u.float())
    conf = torch.sigmoid(cls_m.permute(0, 2, 3, 1).reshape(cls_m.shape[0], -1))
    for thr in (0.05, 0.02, 0.01):
        print(f"  score_thr={thr}: 每样本通过框数="
              f"{(conf > thr).sum(1).tolist()}")

# 3) roi + 精修头
rois = torch.cat([boxes_to_rois(p["boxes"].cpu().numpy(), b, device)
                  for b, p in enumerate(props)], dim=0)
t = time.time()
pooled = roi_align(u, rois, output_size=(7, 7))
fg, res = refine(pooled)
torch.cuda.synchronize()
print(f"roi_align+refine: {time.time()-t:.3f}s  rois={len(rois)}")
