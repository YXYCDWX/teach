# -*- coding: utf-8 -*-
"""YOLO11 在 nuScenes-mini 2D 检测上的训练与评估
- Ultralytics 预训练权重微调
- 训练结束后用统一评估器 (eval2d) 在验证集上计算 mAP, 保证与 Faster R-CNN 可比
- 可选增强/交叉验证 (为提升小数据集上的 YOLO 表现):
    --aug strong   打开更强的空间+混合增强 (degrees/scale/mixup/...)
    --kfold K      场景级 k 折交叉验证 (按场景切分, 避免同场景泄漏)
用法示例:
    python train_yolo.py yolo11                      # 原有行为, 50 轮默认增强
    python train_yolo.py yolo11 --aug strong --epochs 150
    python train_yolo.py yolo11 --aug strong --kfold 4
"""
import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from nuscenes_common import CLASS_NAMES
from eval2d import load_gt_labels, evaluate_full, get_val_images

from ultralytics import YOLO

NU2D = os.path.join(BASE, "nu2d")
DATA_YAML = os.path.join(NU2D, "data.yaml")
TRAIN_IMG_DIR = os.path.join(NU2D, "images", "train")
TRAIN_LBL_DIR = os.path.join(NU2D, "labels", "train")
VAL_IMG_DIR = os.path.join(NU2D, "images", "val")
VAL_LBL_DIR = os.path.join(NU2D, "labels", "val")
FOLD_DIR = os.path.join(NU2D, "folds")
OUT_DIR = os.path.join(BASE, "outputs")

MODELS = {"yolo11": "yolo11n.pt"}

# 增强预设: default 沿用 Ultralytics 默认; strong 针对小数据集加强空间/混合增强
# 说明: 颜色类仅做常规 HSV 抖动(不反色/不灰度), 空间类用旋转/缩放/平移/剪切/透视
AUG_PRESETS = {
    "default": {},
    "strong": dict(
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
        mosaic=1.0, mixup=0.15, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, close_mosaic=10, cos_lr=True,
    ),
}


def collect_predictions(model, img_dir, names, imgsz):
    """在给定图像目录上推理, 返回统一格式 {文件名: [(cls,x1,y1,x2,y2,score), ...]}"""
    preds = {}
    for name in names:
        r = model.predict(os.path.join(img_dir, name), conf=0.001, iou=0.7,
                          max_det=300, imgsz=imgsz, verbose=False)[0]
        boxes = r.boxes
        plist = []
        if boxes is not None and len(boxes):
            b = boxes.xyxy.cpu().numpy()
            c = boxes.cls.cpu().numpy().astype(int)
            s = boxes.conf.cpu().numpy()
            for i in range(len(b)):
                plist.append((int(c[i]), float(b[i][0]), float(b[i][1]),
                              float(b[i][2]), float(b[i][3]), float(s[i])))
        preds[name] = plist
    return preds


def evaluate_unified(model, img_dir, lbl_dir, names, imgsz):
    """用统一评估器计算 mAP (与 Faster R-CNN 同口径)"""
    preds = collect_predictions(model, img_dir, names, imgsz)
    sizes = {n: Image.open(os.path.join(img_dir, n)).size for n in names}
    gts = load_gt_labels(lbl_dir, names, sizes)
    return evaluate_full(preds, gts)


def _write_list(path, items):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(items))


def _write_fold_yaml(path, train_list, val_list):
    lines = [f"train: {train_list.replace(os.sep, '/')}",
             f"val: {val_list.replace(os.sep, '/')}",
             f"nc: {len(CLASS_NAMES)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_single(key, weights, args):
    """单折训练 + 统一评估 (默认路径, 与原有逻辑一致)"""
    print(f"\n{'='*50}\n训练 {key}  (aug={args.aug}{args.tag})\n{'='*50}")
    run_dir = os.path.join(OUT_DIR, f"{key}{args.tag}", "train")
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    csv_path = os.path.join(run_dir, "results.csv")
    done = False
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            n_rows = sum(1 for line in f if line.strip()) - 1
        done = n_rows >= args.epochs
    if done:
        print(f"[{key}] 训练已完成 ({args.epochs} 轮), 跳过训练", flush=True)
    elif os.path.exists(last_pt):
        # Ultralytics 自带断点: 从 last.pt 恢复轮次/优化器
        print(f"[{key}] 从 last.pt 断点续训", flush=True)
        model = YOLO(last_pt)
        model.train(resume=True)
    else:
        model = YOLO(weights)
        model.train(data=DATA_YAML, epochs=args.epochs, batch=args.batch,
                    imgsz=args.imgsz, device=0, workers=0,
                    project=os.path.join(OUT_DIR, f"{key}{args.tag}"),
                    name="train", exist_ok=True, verbose=True,
                    patience=args.epochs, seed=0, **AUG_PRESETS[args.aug])

    best = os.path.join(run_dir, "weights", "best.pt")
    model = YOLO(best)
    names = get_val_images(NU2D)
    res = evaluate_unified(model, VAL_IMG_DIR, VAL_LBL_DIR, names, args.imgsz)

    out = {"model": key, "epochs": args.epochs, "imgsz": args.imgsz, "aug": args.aug,
           "unified_AP50": res["AP50"]["mAP"],
           "unified_AP5095": res["AP5095"]["mAP"],
           "AP50_per_class": res["AP50"]["per_class_ap"],
           "AP5095_per_class": res["AP5095"]["per_class_ap"]}
    with open(os.path.join(OUT_DIR, f"{key}{args.tag}_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[{key}] 统一评估: mAP@0.5={res['AP50']['mAP']:.4f} "
          f"mAP@0.5:0.95={res['AP5095']['mAP']:.4f}")


def run_kfold(key, weights, args):
    """场景级 k 折交叉验证: 按场景切分, 每折留一个场景做验证, 避免同场景帧泄漏"""
    k = args.kfold
    print(f"\n{'='*50}\n{key} 场景级 {k} 折交叉验证  (aug={args.aug}{args.tag})\n{'='*50}")
    files = sorted(os.listdir(TRAIN_IMG_DIR))
    scenes = sorted({f.split("_", 1)[0] for f in files})
    if k > len(scenes):
        raise ValueError(f"k={k} 超过场景数 {len(scenes)}")
    os.makedirs(FOLD_DIR, exist_ok=True)
    assign = {sc: i % k for i, sc in enumerate(scenes)}   # 场景轮转分配
    proj = os.path.join(OUT_DIR, f"{key}{args.tag}_kfold")

    aps = []
    for fi in range(k):
        val_files = [f for f in files if assign[f.split("_", 1)[0]] == fi]
        tr_files = [f for f in files if assign[f.split("_", 1)[0]] != fi]
        if not val_files or not tr_files:
            continue
        tpath = os.path.join(FOLD_DIR, f"{key}_fold{fi}_train.txt")
        vpath = os.path.join(FOLD_DIR, f"{key}_fold{fi}_val.txt")
        _write_list(tpath, [os.path.join(TRAIN_IMG_DIR, f) for f in tr_files])
        _write_list(vpath, [os.path.join(TRAIN_IMG_DIR, f) for f in val_files])
        ypath = os.path.join(FOLD_DIR, f"{key}_fold{fi}.yaml")
        _write_fold_yaml(ypath, tpath, vpath)

        best = os.path.join(proj, f"fold{fi}", "weights", "best.pt")
        if not os.path.exists(best):
            print(f"[{key} fold{fi}] 训练 {len(tr_files)} 张, 验证 {len(val_files)} 张",
                  flush=True)
            YOLO(weights).train(data=ypath, epochs=args.epochs, batch=args.batch,
                                imgsz=args.imgsz, device=0, workers=0,
                                project=proj, name=f"fold{fi}", exist_ok=True,
                                verbose=True, patience=args.epochs, seed=0,
                                **AUG_PRESETS[args.aug])
        model = YOLO(best)
        res = evaluate_unified(model, TRAIN_IMG_DIR, TRAIN_LBL_DIR, val_files, args.imgsz)
        ap50 = res["AP50"]["mAP"]
        aps.append(ap50)
        print(f"[{key} fold{fi}] 验证 {len(val_files)} 张  mAP@0.5={ap50:.4f}", flush=True)

    if aps:
        mean = float(np.mean(aps))
        out = {"model": key, "kfold": k, "aug": args.aug, "epochs": args.epochs,
               "fold_AP50": aps, "mean_AP50": mean, "std_AP50": float(np.std(aps))}
        with open(os.path.join(OUT_DIR, f"{key}{args.tag}_kfold_results.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[{key}] {k} 折平均 mAP@0.5={mean:.4f} (±{np.std(aps):.4f})", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None,
                    help="只训练指定模型: yolo11; 缺省则训练全部")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--aug", default="default", choices=list(AUG_PRESETS))
    ap.add_argument("--tag", default="", help="输出目录后缀, 用于区分不同实验")
    ap.add_argument("--kfold", type=int, default=0, help=">1 时做场景级 k 折交叉验证")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    for key, weights in MODELS.items():
        if args.model and key != args.model:
            continue
        if args.kfold and args.kfold > 1:
            run_kfold(key, weights, args)
        else:
            run_single(key, weights, args)


if __name__ == "__main__":
    main()
