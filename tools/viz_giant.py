# -*- coding: utf-8 -*-
"""可视化当前标签中面积最大的框, 人工判定是否为真实的近距车辆"""
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuscenes_common import CLASS_NAMES
from eval2d import get_val_images
from report_confusion import load_gt

DATA = r"D:\PythonProject\teach\nu2d"
OUT = r"D:\PythonProject\teach\outputs\viz_giant"
os.makedirs(OUT, exist_ok=True)

names = get_val_images(DATA)
gt = load_gt(names)
scored = []
for n, boxes in gt.items():
    for b in boxes:
        x1, y1, x2, y2 = b[1:]
        scored.append(((x2 - x1) * (y2 - y1), n, b))
scored.sort(reverse=True)

picked = {}
for area, n, b in scored:
    picked.setdefault(n, []).append(b)
    if len(picked) >= 6:
        break
for n, boxes in picked.items():
    img = cv2.imread(os.path.join(DATA, "images", "val", n))
    for b in boxes:
        c, x1, y1, x2, y2 = int(b[0]), *map(int, b[1:])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, CLASS_NAMES[c], (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.imwrite(os.path.join(OUT, n), img)
    print(os.path.basename(n), "巨框数:", len(boxes))
