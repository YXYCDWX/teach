# -*- coding: utf-8 -*-
"""可视化核检: 在验证集图片上同画 GT(绿) 与 预测(红), conf=0.25
输出到 outputs/viz_check/ 供人工核验。
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))
from nuscenes_common import CLASS_NAMES
from eval2d import get_val_images
from report_confusion import load_gt, DATA

OUT = r"D:\PythonProject\teach\outputs\viz_check"
os.makedirs(OUT, exist_ok=True)

names = get_val_images(DATA)
gt = load_gt(names)

# 挑真值框多、含小目标(pedestrian/motorcycle/bicycle)的图
def score(n):
    cls = [b[0] for b in gt[n]]
    return len(cls) + 3 * sum(c in (7, 8) for c in cls)

picks = sorted(names, key=score, reverse=True)[:10]

model = YOLO(r"D:\PythonProject\teach\outputs\yolo11\train\weights\best.pt")
r = model.predict(source=[os.path.join(DATA, "images", "val", n) for n in picks],
                  conf=0.25, imgsz=640, verbose=False, stream=True)
for name, res in zip(picks, r):
    img = cv2.imread(os.path.join(DATA, "images", "val", name))
    for b in gt[name]:
        c, x1, y1, x2, y2 = int(b[0]), *map(int, b[1:])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
    pb = res.boxes.xyxy.cpu().numpy().astype(int)
    pc = res.boxes.cls.cpu().numpy().astype(int)
    ps = res.boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c, s in zip(pb, pc, ps):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{CLASS_NAMES[c]} {s:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(OUT, name), img)
    print(name, "gt=", len(gt[name]), "pred=", len(pb))
