# -*- coding: utf-8 -*-
"""统一 2D 检测评估器
所有 2D 模型 (YOLO11 / YOLO26 / SSD) 用同一套 GT 和同一 AP 计算方式,
保证结果可比。指标: COCO 风格 mAP@0.5 与 mAP@0.5:0.95 (全点插值)。
"""
import os
import numpy as np

from nuscenes_common import CLASS_NAMES


def load_gt_labels(label_dir, image_names, img_sizes):
    """读取 YOLO 格式标签 -> {img_name: [(cls, x1,y1,x2,y2 像素), ...]}"""
    gt = {}
    for name in image_names:
        stem = os.path.splitext(name)[0]
        lbl_path = os.path.join(label_dir, stem + ".txt")
        W, H = img_sizes[name]
        boxes = []
        if os.path.exists(lbl_path):
            for line in open(lbl_path, encoding="utf-8"):
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, w, h = map(float, p)
                boxes.append((int(c),
                              (cx - w / 2) * W, (cy - h / 2) * H,
                              (cx + w / 2) * W, (cy + h / 2) * H))
        gt[name] = boxes
    return gt


def iou_matrix(a, b):
    """a: (N,4) b: (M,4) xyxy -> IoU 矩阵 (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _ap_from_matches(scores, matched, num_gt):
    """VOC 2010+ 全点插值 AP"""
    if num_gt == 0:
        return None
    order = np.argsort(-scores)
    matched = np.asarray(matched)[order]
    tp = matched.astype(np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)
    # 全点插值: precision 单调不增包络
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap


def evaluate_predictions(preds, gts, iou_thr=0.5):
    """计算指定 IoU 阈值下各类 AP
    preds: {img_name: [(cls, x1,y1,x2,y2, score), ...]}
    gts:   {img_name: [(cls, x1,y1,x2,y2), ...]}
    """
    per_cls_scores = {c: [] for c in range(len(CLASS_NAMES))}
    per_cls_match = {c: [] for c in range(len(CLASS_NAMES))}
    per_cls_ngt = {c: 0 for c in range(len(CLASS_NAMES))}

    for name in gts:
        gt_list = gts[name]
        pred_list = preds.get(name, [])
        gt_boxes = np.array([b[1:] for b in gt_list]).reshape(-1, 4)
        gt_cls = np.array([b[0] for b in gt_list])
        pred_boxes = np.array([p[1:5] for p in pred_list]).reshape(-1, 4)
        pred_cls = np.array([p[0] for p in pred_list])
        pred_score = np.array([p[5] for p in pred_list])
        ious = iou_matrix(pred_boxes, gt_boxes)

        for c in range(len(CLASS_NAMES)):
            pc = np.where(pred_cls == c)[0]
            gc = np.where(gt_cls == c)[0]
            per_cls_ngt[c] += len(gc)
            # 按分数从高到低贪心匹配
            matched_gt = set()
            m = np.zeros(len(pc))
            for i in pc[np.argsort(-pred_score[pc])]:
                if len(gc) == 0:
                    continue
                j = int(np.argmax(ious[i, gc]))
                if ious[i, gc[j]] >= iou_thr and gc[j] not in matched_gt:
                    matched_gt.add(gc[j])
                    m[list(pc).index(i)] = 1
            per_cls_scores[c].extend(pred_score[pc].tolist())
            per_cls_match[c].extend(m.tolist())

    aps = {}
    for c, name in enumerate(CLASS_NAMES):
        if per_cls_ngt[c] == 0:
            aps[name] = None
            continue
        aps[name] = _ap_from_matches(np.array(per_cls_scores[c]),
                                     np.array(per_cls_match[c]), per_cls_ngt[c])
    vals = [v for v in aps.values() if v is not None]
    return {"mAP": float(np.mean(vals)) if vals else 0.0,
            "per_class_ap": {k: (round(v, 4) if v is not None else None)
                             for k, v in aps.items()},
            "num_gt": per_cls_ngt}


def evaluate_full(preds, gts):
    """mAP@0.5 与 mAP@0.5:0.95"""
    res = {"AP50": evaluate_predictions(preds, gts, 0.5)}
    # 0.5:0.95 平均
    aps = []
    per_cls = {}
    for t in np.arange(0.5, 1.0, 0.05):
        r = evaluate_predictions(preds, gts, float(t))["per_class_ap"]
        for k, v in r.items():
            if v is not None:
                per_cls.setdefault(k, []).append(v)
    for k, v in per_cls.items():
        per_cls[k] = float(np.mean(v))
        aps.append(per_cls[k])
    res["AP5095"] = {"mAP": float(np.mean(aps)) if aps else 0.0,
                     "per_class_ap": {k: round(v, 4) for k, v in per_cls.items()}}
    return res


def get_val_images(data_root):
    val_img_dir = os.path.join(data_root, "images", "val")
    names = sorted(os.listdir(val_img_dir))
    return names
