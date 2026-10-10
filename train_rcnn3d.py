# -*- coding: utf-8 -*-
"""3D 两阶段检测 (Voxel R-CNN 风格): PointPillars 提候选 + RoI 特征精修
- 第一阶段: 已训练的 PointPillars (冻结), 解码候选框, 取置信度 top-K
- 第二阶段: 候选框转 BEV 包围盒 -> torchvision roi_align 在伪图像上池化特征
            -> MLP -> 前景分数 + 7 维框残差 (复用 models3d 的 encode/decode)
- 评估与 train_3d 完全一致 (eval3d, nuScenes 中心距离 mAP), 支持断点续训
用法: python train_rcnn3d.py [epochs=15]
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import roi_align

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from dataset3d import NuScenes3D, compute_anchor_stats
from models3d import PointPillars, build_anchors, encode_boxes, decode_boxes, sigmoid_focal_loss
from eval3d import evaluate_3d

OUT_DIR = os.path.join(BASE, "outputs", "rcnn3d")
GRID_XY = 200
CELL = 100.0 / GRID_XY          # 0.5 m/格, POINT_RANGE 100m
PROPOSAL_K = 32                 # 每样本候选框数
POS_DIST = 2.0                  # 候选-GT 同类中心距离匹配阈值 (m)
EPOCHS = 15
BATCH = 4


def boxes_to_rois(boxes_np, batch_idx, device):
    """3D 框 (M,7)[x,y,z,w,l,h,yaw] -> BEV 轴对齐包围盒 (M,5)[b,x1,y1,x2,y2] (网格坐标)"""
    if len(boxes_np) == 0:
        return torch.zeros(0, 5, device=device)
    x, y, w, l, yaw = (boxes_np[:, 0], boxes_np[:, 1],
                       boxes_np[:, 3], boxes_np[:, 4], boxes_np[:, 6])
    hx = np.abs(l / 2 * np.cos(yaw)) + np.abs(w / 2 * np.sin(yaw))
    hy = np.abs(l / 2 * np.sin(yaw)) + np.abs(w / 2 * np.cos(yaw))
    gx1 = np.clip((x - hx + 50.0) / CELL, 0, GRID_XY - 1e-3)
    gy1 = np.clip((y - hy + 50.0) / CELL, 0, GRID_XY - 1e-3)
    gx2 = np.clip((x + hx + 50.0) / CELL, 1e-3, GRID_XY)
    gy2 = np.clip((y + hy + 50.0) / CELL, 1e-3, GRID_XY)
    rois = np.stack([np.full(len(x), batch_idx, dtype=np.float32),
                     gx1, gy1, gx2, gy2], axis=1)
    return torch.as_tensor(rois, dtype=torch.float32, device=device)


class RefineHead(nn.Module):
    """第二阶段精修头: RoI 特征 -> 前景分数 + 7 维残差"""

    def __init__(self, in_ch=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch * 7 * 7, 512), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.cls = nn.Linear(512, 1)
        self.reg = nn.Linear(512, 7)

    def forward(self, feat):
        """feat: (K, C, 7, 7) -> fg_logit (K,), res (K, 7)"""
        z = self.mlp(feat.flatten(1))
        return self.cls(z).squeeze(-1), self.reg(z)


def get_proposals(pp, u, anchors, device, k=PROPOSAL_K):
    """第一阶段解码候选: 返回 list[dict(cls, boxes(M,7) tensor, scores(M,))]"""
    cls_m, reg_m, dir_m = pp.head(u.float())      # u 为 fp16, 头用 fp32 跑
    results = pp.head.predict(cls_m.float(), reg_m.float(), dir_m.float(),
                              anchors, score_thr=0.05, nms_dist=0.6, max_det=128)
    out = []
    for r in results:
        order = np.argsort(-r["scores"])[:k]
        out.append({"cls": r["cls"][order],
                    "boxes": torch.as_tensor(r["boxes"][order], dtype=torch.float32,
                                             device=device),
                    "scores": torch.as_tensor(r["scores"][order], dtype=torch.float32,
                                              device=device)})
    return out


def match_and_target(props, gt, device):
    """候选-GT 同类 BEV 中心距离匹配 -> fg 目标 (K,), 残差目标 (K,7)"""
    K = len(props["boxes"])
    fg_t = torch.zeros(K, device=device)
    res_t = torch.zeros(K, 7, device=device)
    if K == 0:
        return fg_t, res_t
    if len(gt) > 0:
        gt = gt.to(device)
        d = torch.cdist(props["boxes"][:, :2], gt[:, :2])      # (K,M)
        same = props["cls"].view(-1, 1) == gt[:, 7].view(1, -1)
        for m in range(len(gt)):
            cand = torch.where(same[:, m] & (d[:, m] < POS_DIST))[0]
            if len(cand) == 0:
                continue
            best = cand[torch.argmin(d[cand, m])]
            fg_t[best] = 1.0
            res_t[best] = encode_boxes(gt[m, :7].view(1, 7),
                                       props["boxes"][best].view(1, 7)).squeeze(0)
    return fg_t, res_t


def forward_batch(pp, refine, batch, anchors, device, with_head=True):
    """共同前向: 返回 (每样本候选列表, 精修输出 fg_logit/res 或 None)"""
    v, c, n, gt = batch
    v, c, n = v.to(device), c.to(device), n.to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        _, _, _, u = pp(v, c, n, (GRID_XY, GRID_XY), return_feat=True)
    u = u.float()
    props = get_proposals(pp, u, anchors, device)
    rois = torch.cat([boxes_to_rois(p["boxes"].cpu().numpy(), b, device)
                      for b, p in enumerate(props)], dim=0)
    if len(rois) == 0 or not with_head:
        return props, None, None, u
    pooled = roi_align(u, rois, output_size=(7, 7))
    fg_logit, res = refine(pooled)
    return props, fg_logit, res, u


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    stage1_path = os.path.join(BASE, "outputs", "pointpillars", "best.pth")
    if not os.path.exists(stage1_path):
        print("[rcnn3d] 缺少第一阶段权重 outputs/pointpillars/best.pth, 先训练 pointpillars")
        sys.exit(1)

    anchor_stats = compute_anchor_stats("train")
    anchors = build_anchors(anchor_stats, GRID_XY, GRID_XY)

    train_ds = NuScenes3D("train", mode="pillar", max_voxels=18000)
    val_ds = NuScenes3D("val", mode="pillar", augment=False, max_voxels=18000)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                              collate_fn=NuScenes3D.collate, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                            collate_fn=NuScenes3D.collate, pin_memory=True)

    # 第一阶段: 冻结的 PointPillars
    pp = PointPillars(anchor_stats).to(device)
    ck = torch.load(stage1_path, map_location="cpu", weights_only=False)
    pp.load_state_dict(ck["model"])
    pp.eval()
    for p in pp.parameters():
        p.requires_grad_(False)

    refine = RefineHead().to(device)
    n_params = sum(p.numel() for p in refine.parameters()) / 1e6
    print(f"[rcnn3d] 两阶段: PointPillars(冻结, stage1 mAP={ck.get('mAP', -1):.4f}) "
          f"+ 精修头 {n_params:.2f}M  轮数: {epochs}  候选数: {PROPOSAL_K}")

    opt = torch.optim.AdamW(refine.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    last_path = os.path.join(OUT_DIR, "last.pth")
    start_ep, best_map = 1, -1.0
    if os.path.exists(last_path):
        c = torch.load(last_path, map_location="cpu", weights_only=False)
        refine.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"])
        start_ep, best_map = c["epoch"] + 1, c.get("best_map", -1.0)
        print(f"[rcnn3d] 从 last.pth 续训: 已完成 {c['epoch']}/{epochs} 轮", flush=True)

    log_path = os.path.join(OUT_DIR, "train_log.txt")

    @torch.no_grad()
    def evaluate():
        """验证: 精修后框 -> eval3d (与 train_3d 同一口径)"""
        refine.eval()
        predictions, gts = {}, {}
        key = 0
        for v, c, n, gt in val_loader:
            props, fg_logit, res, _ = forward_batch(pp, refine, (v, c, n, gt),
                                                    anchors, device)
            if fg_logit is not None:
                fg_all = torch.sigmoid(fg_logit).cpu().numpy()
                off = 0
                for b, p in enumerate(props):
                    k = len(p["boxes"])
                    fg = fg_all[off:off + k]
                    boxes = decode_boxes(res[off:off + k], p["boxes"]).cpu().numpy()
                    scores = p["scores"].cpu().numpy() * fg
                    predictions[f"f{key}"] = {"cls": p["cls"],
                                              "boxes": boxes.astype(np.float32),
                                              "scores": scores.astype(np.float32)}
                    off += k
            else:
                for p in props:
                    predictions[f"f{key}"] = {"cls": p["cls"],
                                              "boxes": p["boxes"].cpu().numpy().astype(np.float32),
                                              "scores": p["scores"].cpu().numpy()}
            for b, g in enumerate(gt):
                gts[f"f{key}"] = {"cls": g[:, 7].numpy().astype(int),
                                  "boxes": g[:, :7].numpy(),
                                  "scores": np.ones(len(g))}
                key += 1
        refine.train()
        return evaluate_3d(predictions, gts)

    for ep in range(start_ep, epochs + 1):
        refine.train()
        t0 = time.time()
        tot, n = {"cls": 0.0, "reg": 0.0}, 0
        for v, c, npts, gt in train_loader:
            props, fg_logit, res, _ = forward_batch(pp, refine, (v, c, npts, gt),
                                                    anchors, device)
            if fg_logit is None:
                continue
            opt.zero_grad(set_to_none=True)
            fg_ts, res_ts = [], []
            for p, g in zip(props, gt):
                ft, rt = match_and_target(p, g, device)
                fg_ts.append(ft)
                res_ts.append(rt)
            fg_t = torch.cat(fg_ts)
            res_t = torch.cat(res_ts)
            cls_loss = sigmoid_focal_loss(fg_logit, fg_t).mean()
            pos_m = fg_t > 0.5
            reg_loss = (F.smooth_l1_loss(res[pos_m], res_t[pos_m])
                        if pos_m.any() else fg_logit.new_zeros(()))
            loss = cls_loss + reg_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot["cls"] += cls_loss.item(); tot["reg"] += reg_loss.item(); n += 1

        sched.step()
        msg = (f"[rcnn3d] epoch {ep}/{epochs}  cls={tot['cls']/max(n,1):.4f} "
               f"reg={tot['reg']/max(n,1):.4f}  用时 {time.time()-t0:.0f}s")
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if ep % 5 == 0 or ep == epochs:
            r = evaluate()
            msg = f"[rcnn3d] epoch {ep} 验证: mAP={r['mAP']:.4f}  " + \
                  "  ".join(f"{k}={v:.3f}" for k, v in r.items()
                            if k != "mAP" and v is not None)
            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            if r["mAP"] > best_map:
                best_map = r["mAP"]
                torch.save({"model": refine.state_dict(), "mAP": best_map,
                            "epoch": ep}, os.path.join(OUT_DIR, "best.pth"))

        torch.save({"model": refine.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best_map": best_map}, last_path)

    print(f"[rcnn3d] 训练完成, 最佳 mAP={best_map:.4f}")


if __name__ == "__main__":
    main()
