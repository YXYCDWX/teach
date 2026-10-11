# -*- coding: utf-8 -*-
"""用训练好的 2D 模型在 nuScenes CAM_FRONT 连续帧上逐帧检测, 渲染成行车视频
- 素材: 指定 split 场景的 CAM_FRONT (keyframe + sweep, 约 12Hz 连续)
- 模型: --model 选择 frcnn / yolo11 / yolo26, 权重路径见 MODEL_SPECS
用法: python tools/make_video_2d.py [fps] [score_thr] [max_frames] [--model frcnn] [--split train|val]
输出: outputs/videos/{model}_cam_front.mp4 (train) 或 {model}_cam_front_val.mp4 (val)
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from nuscenes_common import NuScenesTables, CLASS_NAMES

DATAROOT = os.path.join(BASE, "v1.0-mini")
OUT_DIR = os.path.join(BASE, "outputs", "videos")
NUM_CLASSES = len(CLASS_NAMES) + 1     # + 背景
CAM = "CAM_FRONT"

# 模型名 -> (推理后端, 权重路径)
MODEL_SPECS = {
    "frcnn":  ("frcnn", os.path.join(BASE, "outputs", "frcnn", "best.pth")),
    "yolo11": ("yolo",  os.path.join(BASE, "outputs", "yolo11", "train", "weights", "best.pt")),
    "yolo26": ("yolo",  os.path.join(BASE, "outputs", "yolo26", "train", "weights", "best.pt")),
}


# 每类一个 BGR 颜色
PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 215, 255), (255, 0, 255), (128, 0, 255),
    (0, 0, 255), (255, 255, 0), (255, 0, 128), (0, 128, 255), (160, 160, 160),
]


def open_writer(path_no_ext, fps, size):
    """优先 mp4v, 失败回退 avc1 / MJPG(avi), 返回 (writer, 实际路径)"""
    for fourcc, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi")]:
        p = path_no_ext + ext
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            print(f"视频编码 {fourcc} -> {p}", flush=True)
            return w, p
        w.release()
    raise RuntimeError("无法打开视频写入器 (检查 opencv 编码器)")


def build_model_2d(model_key, device):
    """按模型键加载对应后端模型"""
    backend, ckpt = MODEL_SPECS[model_key]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"权重不存在: {ckpt}")
    if backend == "frcnn":
        # 权重全在 ckpt 中, 显式关闭 backbone 预训练下载, 避免联网卡住
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        return model.to(device).eval()
    from ultralytics import YOLO
    return YOLO(ckpt)


def collect_frames(tables, split):
    """指定 split 的 CAM_FRONT 连续帧, 按场景与时间戳排序
    与 convert_2d.py 的场景切分保持一致: 前 8 场景训练, 后 2 场景验证
    """
    scenes = sorted(tables.scenes(), key=lambda s: s["name"])
    scenes = scenes[:8] if split == "train" else scenes[8:]
    frames = []
    for sc in scenes:
        for st in tables.scene_samples(sc["token"]):
            sds = tables.list_sample_data(st, CAM, keyframe_only=False)
            sds.sort(key=lambda r: r["timestamp"])
            for r in sds:
                frames.append((sc["name"],
                               os.path.join(DATAROOT, r["filename"].replace("/", os.sep))))
    return frames


def draw_boxes(img, boxes, labels, scores, thr):
    """labels 为 0 起类别下标 (与 CLASS_NAMES 对齐)"""
    for (x1, y1, x2, y2), l, s in zip(boxes, labels, scores):
        if s < thr:
            continue
        c = PALETTE[int(l) % len(PALETTE)]
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, c, 2)
        txt = f"{CLASS_NAMES[int(l)]} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], y0 - th - 4), (p1[0] + tw, y0), c, -1)
        cv2.putText(img, txt, (p1[0], y0 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
    return img


def put_text(img, txt, org, scale=0.8):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                2, cv2.LINE_AA)


def model_label(backend, model, model_key):
    """画面标签直接取自权重内部元数据, 避免'路径标签 != 模型身份'
    YOLO 检查点内记录了 yaml_file (如 yolo11n.yaml), 作为自证依据
    """
    if backend == "frcnn":
        return "Faster R-CNN"
    yf = (getattr(model.model, "yaml", {}) or {}).get("yaml_file") or model_key
    return f"{model_key.upper()} [{os.path.basename(str(yf))}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fps", nargs="?", type=int, default=12)
    ap.add_argument("thr", nargs="?", type=float, default=0.3)
    ap.add_argument("max_frames", nargs="?", type=int, default=0)
    ap.add_argument("--model", default="frcnn", choices=list(MODEL_SPECS))
    ap.add_argument("--split", default="train", choices=["train", "val"])
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    os.makedirs(OUT_DIR, exist_ok=True)

    tables = NuScenesTables(DATAROOT)
    frames = collect_frames(tables, args.split)
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"[{args.model}/{args.split}] 共 {len(frames)} 帧 (CAM_FRONT 连续帧), "
          f"conf>={args.thr}", flush=True)

    backend, ckpt = MODEL_SPECS[args.model]
    model = build_model_2d(args.model, device)
    name = model_label(backend, model, args.model)
    print(f"权重路径: {ckpt}\n画面标签(取自权重元数据): {name}", flush=True)
    to_tensor = torchvision.transforms.ToTensor()

    suffix = "" if args.split == "train" else f"_{args.split}"
    out_path = os.path.join(OUT_DIR, f"{args.model}_cam_front{suffix}")
    writer = None
    t0 = time.time()
    gi = 0
    B = 8 if backend == "frcnn" else 16
    for i in range(0, len(frames), B):
        chunk = frames[i:i + B]
        if backend == "frcnn":
            pils, npimgs = [], []
            for sc, path in chunk:
                pil = Image.open(path).convert("RGB")
                npimgs.append(np.array(pil))
                pils.append(to_tensor(pil))
            with torch.no_grad():
                outs = model([t.to(device) for t in pils])
            dets = []
            for out in outs:
                # torchvision 的 label 含背景(1 起) -> 转 0 起
                dets.append((out["boxes"].cpu().numpy(),
                             out["labels"].cpu().numpy() - 1,
                             out["scores"].cpu().numpy()))
        else:
            paths = [p for _, p in chunk]
            npimgs = [np.array(Image.open(p).convert("RGB")) for p in paths]
            results = model.predict(paths, conf=args.thr, iou=0.7, imgsz=640,
                                    verbose=False)
            dets = [(r.boxes.xyxy.cpu().numpy(),
                     r.boxes.cls.cpu().numpy().astype(int),
                     r.boxes.conf.cpu().numpy()) for r in results]

        for (sc, path), arr, (b, l, s) in zip(chunk, npimgs, dets):
            gi += 1
            keep = s >= args.thr
            vis = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            draw_boxes(vis, b[keep], l[keep], s[keep], args.thr)
            put_text(vis, f"{name}  |  {sc}  |  frame {gi}/{len(frames)}",
                     (12, 34), 0.9)
            if writer is None:
                h, w = vis.shape[:2]
                writer, out_path = open_writer(out_path, args.fps, (w, h))
            writer.write(vis)
        if (i // B) % 25 == 0:
            print(f"  {i + len(chunk)}/{len(frames)}  用时 {time.time() - t0:.0f}s",
                  flush=True)

    if writer is not None:
        writer.release()
    print(f"完成: {out_path}  总用时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
