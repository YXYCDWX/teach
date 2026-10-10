# -*- coding: utf-8 -*-
"""nuScenes 3D 检测数据集: 点云体素化 (PointPillars 柱体 / VoxelNet 立方体)"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

BASE = os.path.dirname(os.path.abspath(__file__))
from nuscenes_common import CLASS_NAMES

POINT_RANGE = np.array([-50.0, -50.0, -5.0, 50.0, 50.0, 3.0], dtype=np.float32)


def voxelize(points, voxel_size, max_points, max_voxels):
    """点云体素化 (CPU, numpy)
    points: (N,4) x,y,z,intensity
    返回:
      voxels: (max_voxels, max_points, 9) 特征: x,y,z,i + 到体素中心偏移 xyz + 体素中心 xy(或 xyz)
      coords: (max_voxels, 3) int32 体素网格坐标 (ix,iy,iz)
      num_points: (max_voxels,) 每个体素实际点数
    """
    vr = POINT_RANGE
    mask = np.all((points[:, :3] >= vr[:3]) & (points[:, :3] < vr[3:]), axis=1)
    pts = points[mask]
    n_feat = 9
    voxels = np.zeros((max_voxels, max_points, n_feat), dtype=np.float32)
    coords = np.zeros((max_voxels, 3), dtype=np.int32)
    num_points = np.zeros(max_voxels, dtype=np.int32)

    if len(pts) == 0:
        return voxels, coords, num_points

    vs = np.asarray(voxel_size, dtype=np.float32)
    grid = np.floor((vr[3:] - vr[:3]) / vs).astype(np.int32)  # (gx,gy,gz)
    idx = np.floor((pts[:, :3] - vr[:3]) / vs).astype(np.int32)  # (N,3)
    np.clip(idx, 0, grid - 1, out=idx)
    lin = (idx[:, 0] * grid[1] + idx[:, 1]) * grid[2] + idx[:, 2]

    uniq, inverse, counts = np.unique(lin, return_inverse=True, return_counts=True)
    if len(uniq) > max_voxels:
        # 优先保留点数多的体素
        keep = np.argsort(-counts)[:max_voxels]
        keep.sort()
        remap = -np.ones(len(uniq), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        sel = remap[inverse] >= 0
        pts = pts[sel]
        inverse = remap[inverse][sel]
        uniq = uniq[keep]
        counts = counts[keep]
        if len(pts) == 0:
            return voxels, coords, num_points

    # 恢复体素 3D 坐标
    iz = uniq % grid[2]
    iy = (uniq // grid[2]) % grid[1]
    ix = uniq // (grid[1] * grid[2])
    k = len(uniq)
    coords[:k, 0], coords[:k, 1], coords[:k, 2] = ix, iy, iz

    # 每个点在其体素内的序号 (按进入顺序)
    order = np.argsort(inverse, kind="stable")
    inv_s = inverse[order]
    first = np.r_[True, inv_s[1:] != inv_s[:-1]]
    rank = np.arange(len(inv_s)) - np.maximum.accumulate(
        np.where(first, np.arange(len(inv_s)), 0))

    # 体素中心坐标
    centers = vr[:3] + (np.stack([ix, iy, iz], axis=1) + 0.5) * vs  # (k,3)
    pts_o = pts[order]
    rank_o = rank[order]
    inv_o = inv_s
    keep_mask = rank_o < max_points
    vo = np.zeros((k, max_points, n_feat), dtype=np.float32)
    r = rank_o[keep_mask]
    v_id = inv_o[keep_mask]
    feats = np.concatenate([
        pts_o[keep_mask][:, :3],
        pts_o[keep_mask][:, 3:4],
        pts_o[keep_mask][:, :3] - centers[v_id],       # 到体素中心偏移
        np.tile(centers[v_id][:, :2], (1, 1)) - pts_o[keep_mask][:, :2],  # 体素中心-点 xy
    ], axis=1).astype(np.float32)
    # 特征顺序: x,y,z,i, dx_c,dy_c,dz_c, cx-p.x, cy-p.y  (9 维)
    vo[v_id, r] = feats
    voxels[:k] = vo
    num_points[:k] = np.minimum(counts, max_points)
    return voxels, coords, num_points


class NuScenes3D(Dataset):
    """3D 检测数据集
    mode: 'pillar' (PointPillars, 体素 z 维=1) 或 'voxel' (VoxelNet)
    """

    def __init__(self, split, mode="pillar",
                 pillar_size=(0.5, 0.5, 6.0), voxel_size=(0.5, 0.5, 1.0),
                 max_points=32, max_voxels=18000, augment=True):
        self.root = os.path.join(BASE, "nu3d", split)
        self.files = sorted(os.listdir(self.root))
        self.mode = mode
        self.max_points = max_points
        self.max_voxels = max_voxels
        self.augment = augment and split == "train"
        if mode == "pillar":
            self.voxel_size = pillar_size  # z 方向一个大柱体覆盖整个高度范围
        else:
            self.voxel_size = voxel_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.root, self.files[i]))
        points = d["points"].copy()          # (N,4)
        boxes = d["boxes"].copy()            # (M,8)

        if self.augment and np.random.rand() < 0.5:
            # 沿 x 轴翻转 (y -> -y)
            points[:, 1] *= -1
            boxes[:, 1] *= -1
            boxes[:, 6] *= -1

        voxels, coords, num_points = voxelize(
            points, self.voxel_size, self.max_points, self.max_voxels)
        return (torch.from_numpy(voxels), torch.from_numpy(coords),
                torch.from_numpy(num_points), torch.from_numpy(boxes))

    @staticmethod
    def collate(batch):
        v = torch.stack([b[0] for b in batch])
        c = torch.stack([b[1] for b in batch])
        n = torch.stack([b[2] for b in batch])
        gt = [b[3] for b in batch]
        return v, c, n, gt


def compute_anchor_stats(split="train"):
    """统计各类别平均尺寸与中心高度, 用于生成锚框"""
    root = os.path.join(BASE, "nu3d", split)
    dims = {c: [] for c in range(len(CLASS_NAMES))}
    zs = {c: [] for c in range(len(CLASS_NAMES))}
    for f in os.listdir(root):
        boxes = np.load(os.path.join(root, f))["boxes"]
        for b in boxes:
            c = int(b[7])
            dims[c].append(b[3:6])   # w,l,h
            zs[c].append(b[2])
    stats = {}
    for c in range(len(CLASS_NAMES)):
        if dims[c]:
            stats[c] = {"dwlh": np.mean(np.array(dims[c]), axis=0).tolist(),
                        "z": float(np.mean(zs[c])), "count": len(dims[c])}
        else:
            stats[c] = {"dwlh": [1.0, 1.0, 1.0], "z": -1.0, "count": 0}
    return stats
