# -*- coding: utf-8 -*-
"""Faster R-CNN (torchvision 官方两阶段模型) 在 nu2d 上的微调与统一评估
- COCO 预训练 fasterrcnn_resnet50_fpn, 替换分类头为 10+1 类
- 统一评估器 eval2d, 与 YOLO 可比
- 支持断点续训: last.pth (模型/优化器/调度器/轮次), 已完成则跳过
用法: python train_frcnn.py
"""
import os
import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                          FasterRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from nuscenes_common import CLASS_NAMES
from eval2d import load_gt_labels, evaluate_full, get_val_images

DATA_ROOT = os.path.join(BASE, "nu2d")
OUT_DIR = os.path.join(BASE, "outputs", "frcnn")
EPOCHS = 10
BATCH = 4
NUM_CLASSES = len(CLASS_NAMES) + 1  # +背景


class Nu2DFRCNN(Dataset):
    """Faster R-CNN 数据集: 原始尺寸图 (尺寸变换交给模型内部 transform), 水平翻转增强"""

    def __init__(self, split, augment=True):
        self.img_dir = os.path.join(DATA_ROOT, "images", split)
        self.lbl_dir = os.path.join(DATA_ROOT, "labels", split)
        self.files = sorted(os.listdir(self.img_dir))
        self.augment = augment and split == "train"
        self.to_tensor = torchvision.transforms.ToTensor()
        # 光学增强: 仅改亮度/对比度/饱和度, 不用颜色反转或灰度化 (保持颜色特征可信)
        self.jitter = torchvision.transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        W, H = img.size
        boxes, labels = [], []
        for line in open(os.path.join(self.lbl_dir, os.path.splitext(name)[0] + ".txt"),
                         encoding="utf-8"):
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, w, h = map(float, p)
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H])
            labels.append(int(c) + 1)  # 0 是背景
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.augment and len(boxes):
            # 水平翻转 (几何增强)
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                b = boxes.clone()
                boxes[:, 0] = W - b[:, 2]
                boxes[:, 2] = W - b[:, 0]
            # 光学增强: 亮度/对比度/饱和度抖动 (不改几何, 不破坏框位置)
            if np.random.rand() < 0.5:
                img = self.jitter(img)

        target = {"boxes": boxes, "labels": labels,
                  "image_id": torch.tensor([i]), "iscrowd": torch.zeros(len(labels), dtype=torch.int64)}
        return self.to_tensor(img), target


def collate(batch):
    return tuple(list(b) for b in zip(*batch))


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds = Nu2DFRCNN("train", augment=True)
    val_ds = Nu2DFRCNN("val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, collate_fn=collate, drop_last=True)
    print(f"Faster R-CNN 微调: 训练 {len(train_ds)} 图, 轮数 {EPOCHS}, batch {BATCH}")

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    # 断点续训
    last_path = os.path.join(OUT_DIR, "last.pth")
    start_ep = 1
    best_map = 0.0
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1
        best_map = ck.get("best_map", 0.0)
        print(f"[frcnn] 从 last.pth 续训: 已完成 {ck['epoch']}/{EPOCHS} 轮", flush=True)

    log_path = os.path.join(OUT_DIR, "train_log.txt")

    # last.pth 丢失但训练实际已跑完 (best.pth 在且日志已到最后轮) 时跳过重训,
    # 避免 best.pth 尚在却从头重跑 15 轮的浪费
    if (start_ep == 1 and os.path.exists(os.path.join(OUT_DIR, "best.pth"))
            and os.path.exists(log_path)):
        with open(log_path, encoding="utf-8") as f:
            if f"epoch {EPOCHS}/{EPOCHS}" in f.read():
                start_ep = EPOCHS + 1
                print(f"[frcnn] 训练已完成 ({EPOCHS}/{EPOCHS} 轮, best.pth+train_log 在), 跳过训练",
                      flush=True)

    # 验证集评估器: 每轮算 mAP, 按验证集最优权重存 best.pth
    model.eval()
    names = get_val_images(DATA_ROOT)
    gt_sizes = {n: Image.open(os.path.join(DATA_ROOT, "images", "val", n)).size
                for n in names}
    gts = load_gt_labels(os.path.join(DATA_ROOT, "labels", "val"), names, gt_sizes)

    def eval_map():
        model.eval()
        preds = {}
        with torch.no_grad():
            for i in range(0, len(val_ds), 4):
                idx = list(range(i, min(i + 4, len(val_ds))))
                chunk = [val_ds[j] for j in idx]
                xs = [c[0].to(device) for c in chunk]
                outs = model(xs)
                for j, out in zip(idx, outs):
                    name = val_ds.files[j]
                    b = out["boxes"].cpu().numpy()
                    s = out["scores"].cpu().numpy()
                    l = out["labels"].cpu().numpy() - 1
                    keep = s > 0.001
                    b, s, l = b[keep], s[keep], l[keep]
                    preds[name] = [(int(l[k]), float(b[k][0]), float(b[k][1]),
                                    float(b[k][2]), float(b[k][3]), float(s[k]))
                                   for k in range(len(b))]
        return evaluate_full(preds, gts)["AP50"]["mAP"]

    for ep in range(start_ep, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for imgs, targets in train_loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: t.to(device) for k, t in tg.items()} for tg in targets]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); n += 1
        sched.step()
        msg = f"[frcnn] epoch {ep}/{EPOCHS}  loss={tot/max(n,1):.4f}  用时 {time.time()-t0:.0f}s"
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best_map": best_map}, last_path)
        # 按验证集 mAP 选最优权重 (原实现存最后一轮, 过拟合时会吃亏)
        mAP = eval_map()
        flag = ""
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pth"))
            flag = "  <- best.pth"
        msg2 = f"[frcnn] epoch {ep}  val mAP@0.5={mAP:.4f}{flag}"
        print(msg2, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg2 + "\n")

    # ---- 统一评估 (加载验证集最优权重 best.pth, 使报告结果与所存最优模型一致) ----
    best_path = os.path.join(OUT_DIR, "best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu",
                                         weights_only=False))
        print(f"[frcnn] 统一评估使用 best.pth", flush=True)
    else:
        print("[frcnn] 未找到 best.pth, 使用当前权重评估", flush=True)
    model.eval()
    names = get_val_images(DATA_ROOT)
    gt_sizes = {n: Image.open(os.path.join(DATA_ROOT, "images", "val", n)).size
                for n in names}
    gts = load_gt_labels(os.path.join(DATA_ROOT, "labels", "val"), names, gt_sizes)
    preds = {}
    with torch.no_grad():
        for i in range(0, len(val_ds), 4):
            idx = list(range(i, min(i + 4, len(val_ds))))
            chunk = [val_ds[j] for j in idx]
            xs = [c[0].to(device) for c in chunk]
            outs = model(xs)
            for j, out in zip(idx, outs):
                name = val_ds.files[j]
                b = out["boxes"].cpu().numpy()
                s = out["scores"].cpu().numpy()
                l = out["labels"].cpu().numpy() - 1
                keep = s > 0.001
                b, s, l = b[keep], s[keep], l[keep]
                preds[name] = [(int(l[k]), float(b[k][0]), float(b[k][1]),
                                float(b[k][2]), float(b[k][3]), float(s[k]))
                               for k in range(len(b))]
    res = evaluate_full(preds, gts)
    out = {"model": "frcnn", "epochs": EPOCHS,
           "unified_AP50": res["AP50"]["mAP"],
           "unified_AP5095": res["AP5095"]["mAP"],
           "AP50_per_class": res["AP50"]["per_class_ap"],
           "AP5095_per_class": res["AP5095"]["per_class_ap"]}
    with open(os.path.join(BASE, "outputs", "frcnn_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[frcnn] 统一评估: mAP@0.5={res['AP50']['mAP']:.4f} "
          f"mAP@0.5:0.95={res['AP5095']['mAP']:.4f}")


if __name__ == "__main__":
    main()
