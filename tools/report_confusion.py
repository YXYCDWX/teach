# -*- coding: utf-8 -*-
"""在固定工作点 (conf=0.25, IoU=0.5) 下计算各campaign类别的查准率/查全率与混淆矩阵
用法: python report_confusion.py
"""
import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nuscenes_common import CLASS_NAMES
from eval2d import iou_matrix, get_val_images

DATA = r"D:\PythonProject\teach\nu2d"
CONF = 0.25
IOU_THR = 0.5
N_BG = len(CLASS_NAMES)  # 背景作为最后一个下标


def load_gt(names):
    gt = {}
    for name in names:
        img = cv2.imread(os.path.join(DATA, "images", "val", name))
        H, W = img.shape[:2]
        stem = os.path.splitext(name)[0]
        boxes = []
        lp = os.path.join(DATA, "labels", "val", stem + ".txt")
        if os.path.exists(lp):
            for line in open(lp, encoding="utf-8"):
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, w, h = map(float, p)
                boxes.append((int(c),
                              (cx - w / 2) * W, (cy - h / 2) * H,
                              (cx + w / 2) * W, (cy + h / 2) * H))
        gt[name] = boxes
    return gt


def predict(weights, names):
    model = YOLO(weights)
    preds = {}
    r = model.predict(source=[os.path.join(DATA, "images", "val", n) for n in names],
                      conf=CONF, imgsz=640, verbose=False, stream=True)
    for name, res in zip(names, r):
        pb = res.boxes.xyxy.cpu().numpy()
        pc = res.boxes.cls.cpu().numpy().astype(int)
        ps = res.boxes.conf.cpu().numpy()
        preds[name] = [(int(c), *b.tolist(), float(s)) for c, b, s in zip(pc, pb, ps)]
    return preds


def confusion(preds, gt):
    """(n_cls+1, n_cls+1) 混淆矩阵: 行=预测, 列=真值; 最后行列=背景"""
    K = N_BG + 1
    mat = np.zeros((K, K), dtype=int)
    per = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(N_BG)}
    for name in gt:
        g = gt[name]
        p = preds.get(name, [])
        gboxes = np.array([b[1:] for b in g]).reshape(-1, 4)
        gcls = np.array([b[0] for b in g])
        pboxes = np.array([q[1:5] for q in p]).reshape(-1, 4)
        pcls = np.array([q[0] for q in p])
        ious = iou_matrix(pboxes, gboxes)
        used = set()
        for i in range(len(p)):
            if len(g) == 0:
                per[int(pcls[i])]["fp"] += 1
                mat[int(pcls[i]), N_BG] += 1
                continue
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= IOU_THR and j not in used:
                used.add(j)
                per[int(pcls[i])]["tp"] += 1
                mat[int(pcls[i]), int(gcls[j])] += 1
            else:
                per[int(pcls[i])]["fp"] += 1
                mat[int(pcls[i]), N_BG] += 1
        for j in range(len(g)):
            if j not in used:
                per[int(gcls[j])]["fn"] += 1
                mat[N_BG, int(gcls[j])] += 1
    return mat, per


def report(tag, weights):
    names = get_val_images(DATA)
    gt = load_gt(names)
    preds = predict(weights, names)
    mat, _ = confusion(preds, gt)
    print(f"\n===== {tag} (conf={CONF}, IoU={IOU_THR}) =====")
    print("类别, TP, FP, FN, 查准率P, 查全率R")
    tp_all = fp_all = fn_all = 0
    for c, n in enumerate(CLASS_NAMES):
        # 同类匹配才算 TP: 对角线=TP, 行和-对角=FP(误检/跨类), 列和-对角=FN(漏检)
        tp = int(mat[c, c])
        fp = int(mat[c].sum() - tp)
        fn = int(mat[:, c].sum() - tp)
        per = {c: {"tp": tp, "fp": fp, "fn": fn}}
        tp_all += tp; fp_all += fp; fn_all += fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if tp + fp + fn:
            print(f"{n}, {tp}, {fp}, {fn}, {prec:.3f}, {rec:.3f}")
    prec = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    rec = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"总体, {tp_all}, {fp_all}, {fn_all}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    print("混淆矩阵 (行=预测, 列=真值, 末行列=背景/漏检):")
    labels = CLASS_NAMES + ["背景"]
    print("pred\\gt, " + ", ".join(n[:6] for n in labels))
    for i, n in enumerate(labels):
        print(n[:8] + ", " + ", ".join(str(x) for x in mat[i]))


if __name__ == "__main__":
    report("YOLO11", r"D:\PythonProject\teach\outputs\yolo11\train\weights\best.pt")
