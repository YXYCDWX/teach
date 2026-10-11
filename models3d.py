# -*- coding: utf-8 -*-
"""3D 检测模型: CenterPoint (纯 PyTorch 手写实现)
- 柱体骨干: PointPillars 风格 PFN -> 伪图像 -> 2D CNN -> 384 通道融合特征
- 无锚框检测头: BEV 热力图峰值定位物体 (CenterPoint, Yin et al., 2021)
  - 热力图目标: 官方高斯斑 (gaussian_radius + draw_gaussian_to_heatmap, OpenPCDet 口径)
  - 损失: penalty-reduced focal (官方 CenterNet/CenterPoint neg_loss_cornernet) + 正样本 L1
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nuscenes_common import CLASS_NAMES


def gaussian_radius(height, width, min_overlap=0.1):
    """CenterPoint/CornerNet 官方高斯半径反解
    逐字对齐 OpenPCDet `centernet_utils.gaussian_radius`(注意其 r1/r2 取正根, 与
    CornerNet 原文写法不同; 最终取 min, 实际由 r3 主导)。
    height/width: 物体在热力图两个方向占的格数。返回值调用方再 clamp_min(2)。
    """
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / 2.0
    a2, b2 = 4.0, 2.0 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))
    r2 = (b2 + sq2) / 2.0
    a3, b3 = 4.0 * min_overlap, -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2.0
    return min(r1, r2, r3)


def draw_gaussian_to_heatmap(heatmap, cx, cy, radius, k=1.0):
    """在单类别 2D 热力图 (A,B) 上以 (cx,cy) 为峰画各向同性二维高斯,
    与已有值逐元素取 max。对齐 OpenPCDet `centernet_utils.draw_gaussian_to_heatmap`
    (diameter=2r+1, sigma=diameter/6, 中心 1.0 向四周衰减)。
    本实现约定 dim0=ix(x 方向), dim1=iy(y 方向); cx/cy 为整数格序号。
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ax = torch.arange(-radius, radius + 1, device=heatmap.device, dtype=torch.float32)
    gauss = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2)
                      / (2.0 * sigma * sigma)) * k
    A, B = heatmap.shape
    left, right = min(cx, radius), min(A - cx, radius + 1)
    top, bottom = min(cy, radius), min(B - cy, radius + 1)
    win = heatmap[cx - left:cx + right, cy - top:cy + bottom]
    gw = gauss[radius - left:radius + right, radius - top:radius + bottom]
    if win.numel() > 0 and gw.numel() > 0:
        torch.maximum(win, gw, out=win)
    return heatmap


class CenterHead(nn.Module):
    """CenterPoint 风格无锚框检测头 (Yin et al., 2021)
    每类别一张 BEV 热力图找峰值定位物体, 8 维回归通道:
      0:2 胞内偏移 (x,y), 2: 高度 z, 3:6 尺寸 log(w,l,h), 6:8 朝向 sin/cos
    热力图目标: 官方高斯斑 (gaussian_radius + draw_gaussian_to_heatmap, OpenPCDet 口径)
    损失: penalty-reduced focal (CenterNet/CenterPoint 官方 neg_loss_cornernet) + 正样本 L1
    与锚框头范式对照: 无先验框、无锚框-GT 匹配, 中心点即预测
    """

    CELL = 0.5    # 米/格 (点云范围 100m / 200 格)
    X0 = -50.0    # 点云范围下界

    def __init__(self, in_channels=384, n_classes=len(CLASS_NAMES)):
        super().__init__()
        self.n_classes = n_classes
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.hm = nn.Conv2d(256, n_classes, 1)
        self.reg = nn.Conv2d(256, 8, 1)
        # 热力图偏置初始化: 初始低响应, focal 训练的标准稳定做法
        nn.init.constant_(self.hm.bias, -2.19)

    def forward(self, x):
        u = self.shared(x)
        return self.hm(u), self.reg(u)      # (B,C,X,Y), (B,8,X,Y)

    def loss(self, hm, reg_map, gt_boxes_list):
        B, C, X, Y = hm.shape
        device = hm.device
        hm_t = torch.zeros_like(hm)
        reg_t = torch.zeros(B, 8, X, Y, device=device)
        for b in range(B):
            gt = gt_boxes_list[b]           # (M,8) [cx,cy,cz,w,l,h,yaw,cls]
            for m in range(len(gt)):
                cx, cy = gt[m, 0], gt[m, 1]
                fx = (cx - self.X0) / self.CELL     # 格坐标 (小数)
                fy = (cy - self.X0) / self.CELL
                ix = int(torch.clamp(torch.floor(fx), 0, X - 1).item())
                iy = int(torch.clamp(torch.floor(fy), 0, Y - 1).item())
                c = int(gt[m, 7].item())
                # 官方高斯热力图目标 (对齐 OpenPCDet center_head.assign_target_of_single_head
                #   + centernet_utils.draw_gaussian_to_heatmap):
                #   radius 由物体在热力图两个方向的格数反解, 下限 2 格; 中心 1.0, 四周高斯衰减。
                radius = max(int(gaussian_radius(gt[m, 3].item() / self.CELL,
                                                 gt[m, 4].item() / self.CELL,
                                                 min_overlap=0.1)), 2)
                draw_gaussian_to_heatmap(hm_t[b, c], ix, iy, radius)
                # 胞内偏移目标 (0~1), 解码 x = (ix+off)*CELL + X0
                reg_t[b, 0, ix, iy] = fx - ix
                reg_t[b, 1, ix, iy] = fy - iy
                reg_t[b, 2, ix, iy] = gt[m, 2]      # z 直接回归
                reg_t[b, 3:6, ix, iy] = torch.log(torch.clamp(gt[m, 3:6], min=1e-3))
                reg_t[b, 6, ix, iy] = torch.sin(gt[m, 6])
                reg_t[b, 7, ix, iy] = torch.cos(gt[m, 6])

        # penalty-reduced focal, 官方 CenterNet/CenterPoint neg_loss_cornernet
        # (OpenPCDet loss_utils.FocalLossCenterNet, 逐字对齐):
        #   pos_inds = gt==1, neg_inds = gt<1, neg_weights = (1-gt)^4
        #   loss = -( log(p)*(1-p)^2*pos_inds + log(1-p)*p^2*neg_weights*neg_inds ).sum() / num_pos
        # 关键: 高斯目标含 0~1 软标签, 近中心的衰减格由 neg_weights=(1-gt)^4 降权。
        # 关键: 归一化用正样本数 num_pos, 且用 sum()/num_pos 而非 mean()。
        #   用 mean() 时负压被总格点数 N(≈10*200*200=40万) 稀释约 N/num_pos≈1.3万倍,
        #   假阳性无法压制 -> 热力图成片高响应 -> 每帧数百个乱框。
        p = hm.sigmoid()
        pos_inds = hm_t.eq(1).float()
        neg_inds = hm_t.lt(1).float()
        neg_weights = (1.0 - hm_t).pow(4)
        # log(p)=logsigmoid(hm), log(1-p)=logsigmoid(-hm): 数值稳定, 避免 log(0)=-inf
        pos_loss = F.logsigmoid(hm) * (1.0 - p).pow(2) * pos_inds
        neg_loss = F.logsigmoid(-hm) * p.pow(2) * neg_weights * neg_inds
        num_pos = pos_inds.sum()
        if num_pos > 0:
            cls_loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        else:
            cls_loss = -neg_loss.sum()

        # 正样本格点的回归 L1 (任意类别的正格点都参与, mask 对齐到回归通道维)
        pos_any = (hm_t == 1).any(dim=1).unsqueeze(1)       # (B,1,X,Y)
        pos_mask = pos_any.expand_as(reg_t)                  # (B,8,X,Y)
        n_pos = pos_any.sum().clamp(min=1).float()
        if pos_mask.any():
            reg_loss = F.l1_loss(reg_map, reg_t, reduction="none")[pos_mask].sum() / n_pos
        else:
            reg_loss = hm.new_zeros(())
        return cls_loss, reg_loss

    @torch.no_grad()
    def predict(self, hm, reg_map, score_thr=0.1, nms_dist=0.6, max_det=100):
        """热力图 3x3 峰值 NMS + top-K 解码 + 同类距离 NMS
        返回每样本 dict(cls ids, boxes (M,7), scores)
        """
        B, C, X, Y = hm.shape
        device = hm.device
        h = hm.sigmoid()
        hmax = F.max_pool2d(h, 3, stride=1, padding=1)   # 局部极大值
        keep = (h == hmax).float()
        results = []
        for b in range(B):
            flat = (h[b] * keep[b]).view(C, -1)          # (C, X*Y)
            k = min(500, flat.numel())
            scores, idx = flat.view(-1).topk(k)
            sel = scores > score_thr
            if sel.sum() == 0:
                results.append({"cls": np.zeros(0, dtype=int),
                                "boxes": np.zeros((0, 7), dtype=np.float32),
                                "scores": np.zeros(0)})
                continue
            scores, idx = scores[sel], idx[sel]
            cls_np = (idx // (X * Y)).cpu().numpy()
            pix = idx % (X * Y)
            ix_f = (pix // Y).float()                    # 行 -> X 方向格序号
            iy_f = (pix % Y).float()                     # 列 -> Y 方向格序号
            rg = reg_map[b][:, pix // Y, pix % Y]        # (8,N) 注意 dim0=X, dim1=Y
            x = (ix_f + rg[0]) * self.CELL + self.X0
            y = (iy_f + rg[1]) * self.CELL + self.X0
            z = rg[2]
            wlh = rg[3:6].exp()
            yaw = torch.atan2(rg[6], rg[7])
            boxes = torch.stack([x, y, z, wlh[0], wlh[1], wlh[2], yaw], dim=1).cpu().numpy()
            conf = scores.cpu().numpy()
            # 同类距离 NMS (贪心)
            order = np.argsort(-conf)
            final, suppressed = [], np.zeros(len(order), dtype=bool)
            for ii in range(len(order)):
                if suppressed[ii]:
                    continue
                i = order[ii]
                final.append(i)
                if len(final) >= max_det:
                    break
                d = np.linalg.norm(boxes[order[ii + 1:], :2] - boxes[i, :2], axis=1)
                same = cls_np[order[ii + 1:]] == cls_np[i]
                suppressed[ii + 1:] |= (d < nms_dist) & same
            final = final[:max_det]
            results.append({"cls": cls_np[final],
                            "boxes": boxes[final].astype(np.float32),
                            "scores": conf[final].astype(np.float32)})
        return results


class PillarBackbone(nn.Module):
    """CenterPoint 的柱体骨干: PFN -> 伪图像 -> 2D CNN -> 融合特征 -> CenterHead"""

    def __init__(self, in_feat=9, n_classes=len(CLASS_NAMES)):
        super().__init__()
        # PFN: 简化 PillarFeatureNet
        self.pfn = nn.Sequential(
            nn.Linear(in_feat, 64), nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True))
        # 2D 骨干: 3 个 stride-2 卷积块
        self.block1 = self._block(64, 64, 2)    # 200->100
        self.block2 = self._block(64, 128, 2)   # 100->50
        self.block3 = self._block(128, 256, 2)  # 50->25
        # 上采样对齐回 200x200
        self.up1 = nn.ConvTranspose2d(64, 128, 2, stride=2)          # 100->200
        self.up2 = nn.ConvTranspose2d(128, 128, 4, stride=4)         # 50->200
        self.up3 = nn.ConvTranspose2d(256, 128, 8, stride=8)         # 25->200
        self.head = CenterHead(384, n_classes)

    @staticmethod
    def _block(cin, cout, stride):
        return nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(cin, cout, 3, stride=stride), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))

    def forward(self, voxels, coords, num_points, grid=(200, 200)):
        """voxels: (B, V, P, 9)  coords: (B, V, 3)  num_points: (B, V)
        返回 CenterHead 输出 (hm, reg)
        """
        B, V, P, F_ = voxels.shape
        device = voxels.device
        # 只处理有点的体素
        mask = num_points > 0                     # (B,V)
        v = voxels[mask]                          # (N,P,9)
        n = num_points[mask]
        N = v.shape[0]
        feat = self.pfn(v.reshape(-1, F_)).reshape(N, P, -1)   # (N,P,64)
        # 掩码最大池化
        feat = feat * (torch.arange(P, device=device).view(1, P, 1) < n.view(-1, 1, 1)).float()
        feat = feat.max(dim=1).values             # (N,64)

        # 散射到伪图像
        img = torch.zeros(B, 64, grid[0], grid[1], device=device)
        ix, iy = coords[mask, 0].long(), coords[mask, 1].long()
        img[mask_coords(B, mask, device), :, ix, iy] = feat
        x1 = self.block1(img)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        u = torch.cat([self.up1(x1), self.up2(x2), self.up3(x3)], dim=1)  # 3*128=384
        return self.head(u)


def mask_coords(B, mask, device):
    """返回被 mask 选中的 batch 索引"""
    return torch.arange(B, device=device).view(-1, 1).expand_as(mask)[mask]
