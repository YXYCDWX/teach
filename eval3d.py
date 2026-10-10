# -*- coding: utf-8 -*-
"""3D 检测评估: nuScenes 官方风格的中心距离 mAP
匹配准则: BEV 中心距离阈值 0.5 / 1 / 2 / 4 m, 按置信度贪心一对一匹配
mAP = 各类别、各距离阈值下 AP 的平均 (AP 用全点插值)
"""
import numpy as np

from nuscenes_common import CLASS_NAMES

DIST_THRS = [0.5, 1.0, 2.0, 4.0]


def match_one_class(gt_boxes, gt_scores_none, pred_boxes, pred_scores, dist_thr):
    """单类别单阈值匹配
    gt_boxes: (G,7) pred_boxes: (P,7) (已按分数降序)
    返回 matched: (P,) 是否命中
    """
    G, P = len(gt_boxes), len(pred_boxes)
    matched = np.zeros(P, dtype=bool)
    if G == 0:
        return matched
    used = np.zeros(G, dtype=bool)
    for i in range(P):
        if len(gt_boxes) == 0:
            break
        d = np.linalg.norm(gt_boxes[:, :2] - pred_boxes[i, :2], axis=1)
        d = np.where(used, 1e9, d)
        j = int(np.argmin(d))
        if d[j] <= dist_thr:
            used[j] = True
            matched[i] = True
    return matched


def _ap(scores, matched, num_gt):
    if num_gt == 0:
        return None
    order = np.argsort(-scores)
    matched = np.asarray(matched)[order].astype(np.float64)
    tp = np.cumsum(matched)
    fp = np.cumsum(1 - matched)
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def evaluate_3d(predictions, gts):
    """predictions/gts: 每帧 dict(cls:(M,), boxes:(M,7), scores:(M,))
    返回 mAP 与各类别 AP
    """
    per_class_ap = {n: [] for n in CLASS_NAMES}
    for thr in DIST_THRS:
        for c, name in enumerate(CLASS_NAMES):
            gt_b, pr_b, pr_s = [], [], []
            for k in gts:
                g = gts[k]
                gt_b.append(g["boxes"][g["cls"] == c])
                p = predictions.get(k, {"cls": np.zeros(0), "boxes": np.zeros((0, 7)),
                                        "scores": np.zeros(0)})
                m = p["cls"] == c
                order = np.argsort(-p["scores"][m])
                pr_b.append(p["boxes"][m][order])
                pr_s.append(p["scores"][m][order])
            # 合并全部验证帧, 记录帧边界以做逐帧匹配
            matched_all, scores_all, ngt = [], [], 0
            for gi in range(len(gt_b)):
                ngt += len(gt_b[gi])
                if len(pr_b[gi]) == 0:
                    continue
                matched = match_one_class(gt_b[gi], None, pr_b[gi], pr_s[gi], thr)
                matched_all.extend(matched.tolist())
                scores_all.extend(pr_s[gi].tolist())
            ap = _ap(np.array(scores_all), np.array(matched_all), ngt) if ngt else None
            if ap is not None:
                per_class_ap[name].append(ap)
    res = {}
    vals = []
    for name in CLASS_NAMES:
        if per_class_ap[name]:
            res[name] = float(np.mean(per_class_ap[name]))
            vals.append(res[name])
        else:
            res[name] = None
    res["mAP"] = float(np.mean(vals)) if vals else 0.0
    return res
