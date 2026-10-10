# -*- coding: utf-8 -*-
"""YOLO11 / YOLO26 在 nuScenes-mini 2D 检测上的训练与评估
- Ultralytics 预训练权重微调
- 训练结束后用统一评估器 (eval2d) 在验证集上计算 mAP, 保证与 SSD 可比
"""
import os
import sys
import json
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from nuscenes_common import CLASS_NAMES
from eval2d import load_gt_labels, evaluate_full, get_val_images

from ultralytics import YOLO

DATA_YAML = os.path.join(BASE, "nu2d", "data.yaml")
VAL_IMG_DIR = os.path.join(BASE, "nu2d", "images", "val")
VAL_LBL_DIR = os.path.join(BASE, "nu2d", "labels", "val")
OUT_DIR = os.path.join(BASE, "outputs")

MODELS = {"yolo11": "yolo11n.pt", "yolo26": "yolo26n.pt"}
EPOCHS = 50
BATCH = 16
IMGSZ = 640


def collect_predictions(model):
    """在验证集上推理, 返回统一格式的预测结果"""
    names = get_val_images(os.path.join(BASE, "nu2d"))
    preds = {}
    for name in names:
        r = model.predict(os.path.join(VAL_IMG_DIR, name),
                          conf=0.001, iou=0.7, max_det=300,
                          imgsz=IMGSZ, verbose=False)[0]
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
    return preds, names


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None

    for key, weights in MODELS.items():
        if only and key != only:
            continue
        print(f"\n{'='*50}\n训练 {key}\n{'='*50}")
        run_dir = os.path.join(OUT_DIR, key, "train")
        last_pt = os.path.join(run_dir, "weights", "last.pt")
        csv_path = os.path.join(run_dir, "results.csv")
        done = False
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8") as f:
                n_rows = sum(1 for line in f if line.strip()) - 1
            done = n_rows >= EPOCHS
        if done:
            print(f"[{key}] 训练已完成 ({EPOCHS}/{EPOCHS} 轮), 跳过训练", flush=True)
        elif os.path.exists(last_pt):
            # Ultralytics 自带断点: 从 last.pt 恢复轮次/优化器
            print(f"[{key}] 从 last.pt 断点续训", flush=True)
            model = YOLO(last_pt)
            model.train(resume=True)
        else:
            model = YOLO(weights)
            model.train(data=DATA_YAML, epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ,
                        device=0, workers=0, project=os.path.join(OUT_DIR, key),
                        name="train", exist_ok=True, verbose=True,
                        patience=50, seed=0)

        # 用最优权重做统一评估
        best = os.path.join(OUT_DIR, key, "train", "weights", "best.pt")
        model = YOLO(best)
        preds, names = collect_predictions(model)
        from PIL import Image as _I
        sizes = {n: _I.open(os.path.join(VAL_IMG_DIR, n)).size for n in names}
        gts = load_gt_labels(VAL_LBL_DIR, names, sizes)
        res = evaluate_full(preds, gts)

        out = {"model": key, "epochs": EPOCHS, "imgsz": IMGSZ,
               "unified_AP50": res["AP50"]["mAP"],
               "unified_AP5095": res["AP5095"]["mAP"],
               "AP50_per_class": res["AP50"]["per_class_ap"],
               "AP5095_per_class": res["AP5095"]["per_class_ap"]}
        with open(os.path.join(OUT_DIR, f"{key}_results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[{key}] 统一评估: mAP@0.5={res['AP50']['mAP']:.4f} "
              f"mAP@0.5:0.95={res['AP5095']['mAP']:.4f}")


if __name__ == "__main__":
    main()
