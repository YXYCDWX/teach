# -*- coding: utf-8 -*-
"""SSD (torchvision ssdlite320_mobilenet_v3_large) 在 nuScenes-mini 2D 检测上的微调与评估
- ImageNet 预训练骨干 + COCO 预训练检测头 (替换分类层为 10 类)
- 训练后用统一评估器 (eval2d) 计算 mAP, 与 YOLO 系列可比
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from nuscenes_common import CLASS_NAMES
from eval2d import load_gt_labels, evaluate_full, get_val_images

DATA_ROOT = os.path.join(BASE, "nu2d")
OUT_DIR = os.path.join(BASE, "outputs", "ssd")
EPOCHS = 20
BATCH = 16
IMGSZ = 320


class Nu2D(Dataset):
    """SSD 训练数据集: 320x320 直接缩放"""

    def __init__(self, split, augment=True):
        self.img_dir = os.path.join(DATA_ROOT, "images", split)
        self.lbl_dir = os.path.join(DATA_ROOT, "labels", split)
        self.files = sorted(os.listdir(self.img_dir))
        self.augment = augment and split == "train"
        self.to_tensor = torchvision.transforms.ToTensor()

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

        if self.augment and np.random.rand() < 0.5:
            # 水平翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                b = boxes.clone()
                boxes[:, 0] = W - b[:, 2]
                boxes[:, 2] = W - b[:, 0]

        img = img.resize((IMGSZ, IMGSZ), Image.BILINEAR)
        # 只做 ToTensor (0-1): ssdlite320 内部 transform 自带 ImageNet 归一化,
        # 数据集侧再归一化会造成双重归一化, 压低特征幅值损害收敛
        x = self.to_tensor(img)
        if len(boxes):
            # 框坐标缩放到 320
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= IMGSZ / W
            boxes[:, [1, 3]] *= IMGSZ / H
        target = {"boxes": boxes, "labels": labels}
        return x, target, name, (W, H)


def collate(batch):
    return tuple(list(b) for b in zip(*batch))


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds = Nu2D("train", augment=True)
    val_ds = Nu2D("val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, collate_fn=collate)  # Windows 单进程
    print(f"SSD 微调: 训练 {len(train_ds)} 图, 轮数 {EPOCHS}, batch {BATCH}")

    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    # 替换分类头: COCO 预训练头为 91 类, 将每级末尾 1x1 分类卷积重建为 10+1 类
    ch = model.head.classification_head
    C = len(CLASS_NAMES) + 1
    for mod in ch.module_list:
        old = mod[1]  # 末层分类卷积 (1x1)
        na = old.out_channels // 91  # 每位置锚框数
        mod[1] = nn.Conv2d(old.in_channels, na * C, old.kernel_size,
                           stride=old.stride, padding=old.padding)
    model.num_classes = C
    ch.num_columns = C  # SSDScoringHead reshape 参数同步
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    log_path = os.path.join(OUT_DIR, "train_log.txt")
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
        print(f"[ssd] 从 last.pth 续训: 已完成 {ck['epoch']}/{EPOCHS} 轮", flush=True)

    # 验证集评估器: 每轮算 mAP, 按验证集选最优权重存 best.pth
    model.eval()
    names = get_val_images(DATA_ROOT)
    gt_sizes = {n: Image.open(os.path.join(DATA_ROOT, "images", "val", n)).size
                for n in names}
    gts = load_gt_labels(os.path.join(DATA_ROOT, "labels", "val"), names, gt_sizes)

    def eval_map():
        model.eval()
        preds = {}
        with torch.no_grad():
            for i in range(0, len(val_ds), 8):
                chunk = [val_ds[j] for j in range(i, min(i + 8, len(val_ds)))]
                xs = [c[0].to(device) for c in chunk]
                outs = model(xs)
                for (x, _, name, (W, H)), out in zip(chunk, outs):
                    b = out["boxes"].cpu().numpy()
                    s = out["scores"].cpu().numpy()
                    l = out["labels"].cpu().numpy() - 1
                    keep = s > 0.001
                    b, s, l = b[keep], s[keep], l[keep]
                    # 缩放回原图
                    b[:, [0, 2]] *= W / IMGSZ
                    b[:, [1, 3]] *= H / IMGSZ
                    preds[name] = [(int(l[k]), float(b[k][0]), float(b[k][1]),
                                    float(b[k][2]), float(b[k][3]), float(s[k]))
                                   for k in range(len(b))]
        return evaluate_full(preds, gts)["AP50"]["mAP"]

    for ep in range(start_ep, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for imgs, targets, _, _ in train_loader:
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
        msg = f"[ssd] epoch {ep}/{EPOCHS}  loss={tot/max(n,1):.4f}  用时 {time.time()-t0:.0f}s"
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        # 每轮存断点
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
        msg2 = f"[ssd] epoch {ep}  val mAP={mAP:.4f}{flag}"
        print(msg2, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg2 + "\n")

    # ---- 统一评估 (加载验证集最优权重 best.pth, 使报告结果与所存最优模型一致) ----
    best_path = os.path.join(OUT_DIR, "best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu",
                                         weights_only=False))
        print(f"[ssd] 统一评估使用 best.pth", flush=True)
    else:
        print("[ssd] 未找到 best.pth, 使用当前权重评估", flush=True)
    model.eval()
    names = get_val_images(DATA_ROOT)
    gt_sizes = {n: Image.open(os.path.join(DATA_ROOT, "images", "val", n)).size
                for n in names}
    gts = load_gt_labels(os.path.join(DATA_ROOT, "labels", "val"), names, gt_sizes)
    preds = {}
    with torch.no_grad():
        for i in range(0, len(val_ds), 8):
            chunk = [val_ds[j] for j in range(i, min(i + 8, len(val_ds)))]
            xs = [c[0].to(device) for c in chunk]
            outs = model(xs)
            for (x, _, name, (W, H)), out in zip(chunk, outs):
                b = out["boxes"].cpu().numpy()
                s = out["scores"].cpu().numpy()
                l = out["labels"].cpu().numpy() - 1
                keep = s > 0.001
                b, s, l = b[keep], s[keep], l[keep]
                # 缩放回原图
                b[:, [0, 2]] *= W / IMGSZ
                b[:, [1, 3]] *= H / IMGSZ
                preds[name] = [(int(l[k]), float(b[k][0]), float(b[k][1]),
                                float(b[k][2]), float(b[k][3]), float(s[k]))
                               for k in range(len(b))]
    res = evaluate_full(preds, gts)
    out = {"model": "ssd", "epochs": EPOCHS, "imgsz": IMGSZ,
           "unified_AP50": res["AP50"]["mAP"],
           "unified_AP5095": res["AP5095"]["mAP"],
           "AP50_per_class": res["AP50"]["per_class_ap"],
           "AP5095_per_class": res["AP5095"]["per_class_ap"]}
    with open(os.path.join(BASE, "outputs", "ssd_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[ssd] 统一评估: mAP@0.5={res['AP50']['mAP']:.4f} "
          f"mAP@0.5:0.95={res['AP5095']['mAP']:.4f}")


if __name__ == "__main__":
    main()
