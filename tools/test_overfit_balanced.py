# -*- coding: utf-8 -*-
"""根因验证: 按类别归一化的 focal loss 重跑过拟合自检
对照 test_overfit.py (全局归一化): 若稀有类从 0 分变为正常出分, 则根因确认
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from dataset3d import NuScenes3D, compute_anchor_stats
from models3d import PointPillars, build_anchors, sigmoid_focal_loss, encode_boxes
from nuscenes_common import CLASS_NAMES

N_FRAMES = 8
ITERS = 200
N_CLS = len(CLASS_NAMES)

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda")

stats = compute_anchor_stats("train")
anchors = build_anchors(stats, 200, 200)
ds = NuScenes3D("train", mode="pillar", augment=False, max_voxels=18000)

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


def balanced_loss(cls_map, reg_map, dir_map, anchors, gt_boxes_list,
                  pos_thr=2.0, neg_thr=3.5):
    """与 AnchorHead.loss 相同的匹配逻辑, 但分类损失按类别各自归一化"""
    B, A, X, Y = cls_map.shape
    device = cls_map.device
    anc = anchors.view(-1, 7).to(device)
    pos_thr = torch.as_tensor(pos_thr, device=device)

    cls_flat = cls_map.permute(0, 2, 3, 1).reshape(B, -1)
    reg_flat = reg_map.permute(0, 2, 3, 1).reshape(B, -1, 7)
    dir_flat = dir_map.permute(0, 2, 3, 1).reshape(B, -1, 2)
    anchor_cls = (torch.arange(A, device=device) // 2).repeat(X * Y)

    cls_targets, reg_targets, dir_targets = [], [], []
    pos_masks = []
    for b in range(B):
        gt = gt_boxes_list[b].to(device)
        M = len(gt)
        P = anc.shape[0]
        cls_t = torch.zeros(P, device=device)
        reg_t = torch.zeros(P, 7, device=device)
        dir_t = torch.zeros(P, 2, device=device)
        pos_m = torch.zeros(P, dtype=torch.bool, device=device)
        if M > 0:
            dist = torch.cdist(anc[:, :2], gt[:, :2])
            for m in range(M):
                cl = int(gt[m, 7].item())
                mask = (anchor_cls == cl) & (dist[:, m] < pos_thr)
                if mask.sum() == 0:
                    mask = torch.zeros(P, dtype=torch.bool, device=device)
                    mask[torch.argmin(dist[:, m])] = True
                cls_t[mask] = 1.0
                idx = torch.where(mask)[0]
                res = encode_boxes(gt[m, :7].unsqueeze(0).expand(len(idx), 7), anc[idx])
                reg_t[idx] = res
                ang_diff = gt[m, 6] - anc[idx, 6]
                bin_t = (torch.cos(ang_diff) < 0).long()
                dir_t[idx, 0] = (1 - bin_t).float()
                dir_t[idx, 1] = bin_t.float()
                pos_m |= mask
        cls_targets.append(cls_t)
        reg_targets.append(reg_t)
        dir_targets.append(dir_t)
        pos_masks.append(pos_m)
    cls_t = torch.stack(cls_targets)
    reg_t = torch.stack(reg_targets)
    dir_t = torch.stack(dir_targets)
    pos_m = torch.stack(pos_masks)

    # 正样本: 每类各自归一化, 只对有正样本的类平均; 负样本: 全局平均 (不随类数量爆炸)
    focal = sigmoid_focal_loss(cls_flat, cls_t)        # (B,P)
    pos_loss = cls_flat.new_zeros(())
    n_with_pos = 0
    for cl in range(N_CLS):
        cmask = anchor_cls == cl
        pmask = cmask.unsqueeze(0) & (cls_t == 1)
        npos_c = pmask.sum()
        if npos_c > 0:
            pos_loss = pos_loss + focal[pmask].sum() / npos_c.float()
            n_with_pos += 1
    cls_loss = pos_loss / max(n_with_pos, 1)
    neg_mask = cls_t == 0
    cls_loss = cls_loss + focal[neg_mask].mean()

    n_pos = pos_m.sum().clamp(min=1).float()
    if pos_m.any():
        reg_loss = F.smooth_l1_loss(reg_flat[pos_m], reg_t[pos_m], reduction="sum") / n_pos
        dir_loss = F.cross_entropy(dir_flat[pos_m], dir_t[pos_m], reduction="none").sum() / n_pos
    else:
        reg_loss = cls_flat.new_zeros(())
        dir_loss = cls_flat.new_zeros(())
    return cls_loss, reg_loss, dir_loss


print(f"按类别归一化 focal loss, {N_FRAMES} 帧过拟合 {ITERS} 轮...")
model.train()
for it in range(1, ITERS + 1):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        cls_m, reg_m, dir_m = model(v, c, n, (200, 200))
    cls_l, reg_l, dir_l = balanced_loss(cls_m.float(), reg_m.float(), dir_m.float(), anchors, gts)
    loss = cls_l + reg_l + dir_l
    loss.backward()
    opt.step()
    if it % 50 == 0:
        print(f"  iter {it}/{ITERS}  cls={cls_l.item():.4f} reg={reg_l.item():.4f} dir={dir_l.item():.4f}")

model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    cls_m, reg_m, dir_m = model(v, c, n, (200, 200))
res = model.head.predict(cls_m.float(), reg_m.float(), dir_m.float(),
                         anchors, score_thr=0.1, nms_dist=0.6)

total_gt, total_hit = 0, 0
per_cls = {}
for bi, (r, gt) in enumerate(zip(res, gts)):
    g = gt.numpy()
    used = np.zeros(len(r["boxes"]), dtype=bool)
    for b in g:
        total_gt += 1
        cl = int(b[7])
        d = np.linalg.norm(r["boxes"][:, :2] - b[:2], axis=1)
        d = np.where(used | (r["cls"] != cl), 1e9, d)
        j = int(np.argmin(d)) if len(d) else -1
        hit = j >= 0 and d[j] <= 1.0
        if hit:
            used[j] = True
            total_hit += 1
        a = per_cls.setdefault(cl, [0, 0])
        a[0] += 1
        a[1] += hit
print(f"\n结果: 训练帧上检出 {total_hit}/{total_gt} ({total_hit/max(total_gt,1)*100:.0f}%)")
for cl in sorted(per_cls):
    a, b_ = per_cls[cl]
    print(f"  {CLASS_NAMES[cl]:12s} {b_}/{a} ({b_/a*100:.0f}%)")

# 逐类诊断: 各通道最高分 / 预测框总数 / GT 1.5m 内同类预测数
res_low = model.head.predict(cls_m.float(), reg_m.float(), dir_m.float(),
                             anchors, score_thr=0.02, nms_dist=0.6)
print("\n诊断 (score_thr=0.02):")
for cl in range(10):
    mx, n_pred, near = 0.0, 0, 0
    for r, gt in zip(res_low, gts):
        n_pred += int((r["cls"] == cl).sum())
        m = (r["cls"] == cl) & (r["scores"] > mx)
        if m.any():
            mx = float(r["scores"][m].max())
        g = gt.numpy()
        for b in g[g[:, 7] == cl]:
            d = np.linalg.norm(r["boxes"][:, :2] - b[:2], axis=1)
            near += int(((r["cls"] == cl) & (d <= 1.5)).sum())
    print(f"  {CLASS_NAMES[cl]:20s} 最高分 {mx:.3f}  预测总数 {n_pred:4d}  GT1.5m内 {near}")
