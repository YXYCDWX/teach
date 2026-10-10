# -*- coding: utf-8 -*-
"""3D 检测训练: PointPillars / VoxelNet
用法: python train_3d.py pointpillars|voxelnet|centerpoint
- AMP 混合精度, AdamW + 余弦退火
- 评估: nuScenes 中心距离 mAP (eval3d)
"""
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from dataset3d import NuScenes3D, compute_anchor_stats
from models3d import PointPillars, VoxelNet, build_anchors
from eval3d import evaluate_3d

OUT_DIR = os.path.join(BASE, "outputs")
GRID_XY = 200


def get_grid(mode):
    # CLI 名: pointpillars/voxelnet/centerpoint; pillar 骨干用 2D 网格, voxel 用 3D
    if mode in ("pointpillars", "pillar", "centerpoint"):
        return (GRID_XY, GRID_XY)          # pillar: z 维为 1
    return (GRID_XY, GRID_XY, 8)           # voxel: z 8 格


@torch.no_grad()
def evaluate(model, val_loader, anchors, device, mode):
    model.eval()
    predictions, gts = {}, {}
    is_center = (mode == "centerpoint")
    key = 0                                   # 递增计数器: 预测/GT 逐帧对齐,
    for v, c, n, gt in val_loader:            # 不依赖 batch 尺寸恒定
        base = key
        key += len(gt)
        v, c, n = v.to(device), c.to(device), n.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            if is_center:
                hm, reg_m = model(v, c, n, get_grid(mode))
                results = model.head.predict(hm.float(), reg_m.float(),
                                             score_thr=0.1, nms_dist=0.6)
            else:
                cls_m, reg_m, dir_m = model(v, c, n, get_grid(mode))
                cls_m = cls_m.float()
                reg_m = reg_m.float()
                dir_m = dir_m.float()
                results = model.head.predict(cls_m, reg_m, dir_m, anchors,
                                             score_thr=0.05, nms_dist=0.6)
        for b, r in enumerate(results):
            predictions[f"f{base + b}"] = r
        for b, g in enumerate(gt):
            gts[f"f{base + b}"] = {"cls": g[:, 7].numpy().astype(int),
                                   "boxes": g[:, :7].numpy(),
                                   "scores": np.ones(len(g))}
    return evaluate_3d(predictions, gts)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pointpillars"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else \
        (60 if mode == "pointpillars" else (40 if mode == "voxelnet" else 50))
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    anchor_stats = compute_anchor_stats("train")
    anchors = build_anchors(anchor_stats, GRID_XY, GRID_XY)

    if mode in ("pointpillars", "centerpoint"):
        train_ds = NuScenes3D("train", mode="pillar", max_voxels=18000)
        val_ds = NuScenes3D("val", mode="pillar", augment=False, max_voxels=18000)
        batch_size = 4
    else:
        train_ds = NuScenes3D("train", mode="voxel", max_voxels=30000)
        val_ds = NuScenes3D("val", mode="voxel", augment=False, max_voxels=30000)
        batch_size = 2

    # Windows: 单进程 DataLoader, 避免死锁
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=NuScenes3D.collate,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=NuScenes3D.collate,
                            pin_memory=True)

    if mode == "pointpillars":
        model = PointPillars(anchor_stats)
    elif mode == "centerpoint":
        model = PointPillars(anchor_stats, head="center")
    else:
        model = VoxelNet(anchor_stats)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型: {mode}  参数量: {n_params:.2f}M  训练样本: {len(train_ds)}  "
          f"轮数: {epochs}  batch: {batch_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = os.path.join(OUT_DIR, mode)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.txt")

    # 断点续训: last.pth 保存完整可恢复状态 (模型/优化器/调度器/轮次)
    last_path = os.path.join(run_dir, "last.pth")
    start_ep, best_map = 1, -1.0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, best_map = ck["epoch"] + 1, ck.get("best_map", -1.0)
        print(f"[{mode}] 从 last.pth 续训: 已完成 {ck['epoch']}/{epochs} 轮", flush=True)

    if start_ep > epochs:
        print(f"[{mode}] 训练已完成 ({epochs}/{epochs} 轮), 跳过训练", flush=True)
        return

    for ep in range(start_ep, epochs + 1):
        model.train()
        t0 = time.time()
        tot = {"cls": 0.0, "reg": 0.0, "dir": 0.0, "n": 0}
        for it, (v, c, n, gt) in enumerate(train_loader):
            v, c, n = v.to(device, non_blocking=True), c.to(device, non_blocking=True), n.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                if mode == "centerpoint":
                    hm, reg_m = model(v, c, n, get_grid(mode))
                    hm, reg_m = hm.float(), reg_m.float()
                    cls_l, reg_l = model.head.loss(
                        hm, reg_m, [g.to(device) for g in gt])
                    dir_l = torch.zeros((), device=device)
                else:
                    cls_m, reg_m, dir_m = model(v, c, n, get_grid(mode))
                    # 损失用 fp32 计算, 避免数值问题
                    cls_l, reg_l, dir_l = model.head.loss(
                        cls_m.float(), reg_m.float(), dir_m.float(),
                        anchors, [g.to(device) for g in gt])
            loss = cls_l + reg_l + dir_l
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            tot["cls"] += cls_l.item(); tot["reg"] += reg_l.item()
            tot["dir"] += dir_l.item(); tot["n"] += 1

        sched.step()
        msg = (f"[{mode}] epoch {ep}/{epochs}  cls={tot['cls']/tot['n']:.4f} "
               f"reg={tot['reg']/tot['n']:.4f} dir={tot['dir']/tot['n']:.4f} "
               f"用时 {time.time()-t0:.0f}s")
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        # 每 5 轮评估一次, 最后一轮必评估
        if ep % 5 == 0 or ep == epochs:
            res = evaluate(model, val_loader, anchors, device, mode)
            msg = f"[{mode}] epoch {ep} 验证: mAP={res['mAP']:.4f}  " + \
                  "  ".join(f"{k}={v:.3f}" for k, v in res.items()
                            if k != "mAP" and v is not None)
            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            if res["mAP"] > best_map:
                best_map = res["mAP"]
                torch.save({"model": model.state_dict(), "mode": mode,
                            "anchor_stats": anchor_stats, "mAP": best_map,
                            "epoch": ep},
                           os.path.join(run_dir, "best.pth"))

        # 每轮存断点 (含优化器/调度器状态, 可精确续训)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best_map": best_map, "mode": mode,
                    "anchor_stats": anchor_stats}, last_path)

    print(f"[{mode}] 训练完成, 最佳 mAP={best_map:.4f}")


if __name__ == "__main__":
    main()
