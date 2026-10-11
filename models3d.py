# -*- coding: utf-8 -*-
"""3D 检测模型: PointPillars 与 VoxelNet (纯 PyTorch 手写实现)
两者共享同一套锚框检测头与损失函数 (SECOND 风格):
  - 分类: sigmoid focal loss
  - 回归: smooth L1 (7 维残差编码)
  - 朝向: 2 bins softmax 交叉熵
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nuscenes_common import CLASS_NAMES

# 点云范围与网格
POINT_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)


def build_anchors(anchor_stats, grid_x=200, grid_y=200):
    """为每个网格单元生成锚框
    每类别 2 个朝向 (yaw0, yaw0+pi/2)
    返回 anchors: (grid_x, grid_y, n_cls*2, 7) [cx,cy,cz,w,l,h,yaw]
    """
    n_anchor = len(CLASS_NAMES) * 2
    xs = torch.linspace(POINT_RANGE[0] + 0.25, POINT_RANGE[3] - 0.25, grid_x)
    ys = torch.linspace(POINT_RANGE[1] + 0.25, POINT_RANGE[4] - 0.25, grid_y)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    a = torch.zeros(grid_x, grid_y, n_anchor, 7)
    for c in range(len(CLASS_NAMES)):
        st = anchor_stats[c]
        w, l, h = st["dwlh"]
        z = st["z"]
        for r, yaw0 in enumerate([0.0, math.pi / 2]):
            k = c * 2 + r
            a[:, :, k, 0] = gx
            a[:, :, k, 1] = gy
            a[:, :, k, 2] = z
            a[:, :, k, 3] = w
            a[:, :, k, 4] = l
            a[:, :, k, 5] = h
            a[:, :, k, 6] = yaw0
    return a  # (X, Y, A, 7)


def encode_boxes(gt, anchors):
    """gt/anchors: (N,7) -> 残差 (N,7) + 朝向 bin (N,)"""
    dx = (gt[:, 0] - anchors[:, 0]) / anchors[:, 4]
    dy = (gt[:, 1] - anchors[:, 1]) / anchors[:, 3]
    dz = (gt[:, 2] - anchors[:, 2]) / anchors[:, 5]
    dw = torch.log(torch.clamp(gt[:, 3], min=1e-3) / anchors[:, 3])
    dl = torch.log(torch.clamp(gt[:, 4], min=1e-3) / anchors[:, 4])
    dh = torch.log(torch.clamp(gt[:, 5], min=1e-3) / anchors[:, 5])
    dyaw = torch.sin(gt[:, 6] - anchors[:, 6])
    return torch.stack([dx, dy, dz, dw, dl, dh, dyaw], dim=1)


def decode_boxes(res, anchors):
    """残差 (N,7) + anchors (N,7) -> 框 (N,8) 含朝向修正前"""
    x = res[:, 0] * anchors[:, 4] + anchors[:, 0]
    y = res[:, 1] * anchors[:, 3] + anchors[:, 1]
    z = res[:, 2] * anchors[:, 5] + anchors[:, 2]
    w = torch.exp(torch.clamp(res[:, 3], max=5)) * anchors[:, 3]
    l = torch.exp(torch.clamp(res[:, 4], max=5)) * anchors[:, 4]
    h = torch.exp(torch.clamp(res[:, 5], max=5)) * anchors[:, 5]
    # arcsin 恢复 yaw (与 anchor 差在 [-pi/2, pi/2])
    dyaw = torch.clamp(res[:, 6], -1 + 1e-6, 1 - 1e-6)
    yaw = torch.asin(dyaw) + anchors[:, 6]
    return torch.stack([x, y, z, w, l, h, yaw], dim=1)


def sigmoid_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(pred)
    ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * ((1 - p_t) ** gamma)
    loss = loss * (alpha * target + (1 - alpha) * (1 - target))
    return loss


class AnchorHead(nn.Module):
    """锚框检测头 (PointPillars / VoxelNet 共用)"""

    def __init__(self, in_channels=384, n_classes=len(CLASS_NAMES), n_anchor_per_loc=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_anchor = n_anchor_per_loc or n_classes * 2  # 每位置锚框数
        self.cls = nn.Conv2d(in_channels, self.n_anchor, 1)     # 每锚框 1 个前景分数 (按类别通道)
        self.reg = nn.Conv2d(in_channels, self.n_anchor * 7, 1)
        self.dir = nn.Conv2d(in_channels, self.n_anchor * 2, 1)

    def forward(self, x):
        """x: (B, C, X, Y) -> cls (B,A,X,Y), reg (B,A*7,X,Y), dir (B,A*2,X,Y)"""
        cls = self.cls(x)
        reg = self.reg(x)
        drc = self.dir(x)
        return cls, reg, drc

    def loss(self, cls_map, reg_map, dir_map, anchors, gt_boxes_list,
             pos_thr=2.0):
        """计算损失
        cls_map: (B,A,X,Y)  reg_map: (B,A*7,X,Y)  dir_map: (B,A*2,X,Y)
        anchors: (X,Y,A,7)  gt_boxes_list: 每样本 (M,8)
        注: 全体非正样本锚框 (cls_t==0) 均作为背景参与 focal 损失
        (与 SECOND/PointPillars 主流做法一致, 易负样本由 (1-p)^gamma 降权),
        不区分远近负样本
        """
        B, A, X, Y = cls_map.shape
        device = cls_map.device
        anc = anchors.view(-1, 7).to(device)            # (P,7) P=X*Y*A
        pos_thr = torch.as_tensor(pos_thr, device=device)

        cls_flat = cls_map.permute(0, 2, 3, 1).reshape(B, -1)          # (B,P) 每锚框 1 个前景分数
        reg_flat = reg_map.permute(0, 2, 3, 1).reshape(B, -1, 7)
        dir_flat = dir_map.permute(0, 2, 3, 1).reshape(B, -1, 2)
        # anchors 展平顺序为 (x,y,a): p = (x*Y+y)*A + a -> 通道类别 = (p % A) // 2
        anchor_cls = (torch.arange(A, device=device) // 2).repeat(X * Y)  # (P,)

        cls_targets, reg_targets, dir_targets = [], [], []
        pos_masks, reg_anchors = [], []
        for b in range(B):
            gt = gt_boxes_list[b].to(device)            # (M,8)
            M = len(gt)
            P = anc.shape[0]
            cls_t = torch.zeros(P, device=device)       # 每锚框 1 个前景目标
            reg_t = torch.zeros(P, 7, device=device)
            dir_t = torch.zeros(P, 2, device=device)
            pos_m = torch.zeros(P, dtype=torch.bool, device=device)
            if M > 0:
                # 锚框-GT 按类别 + BEV 中心距离匹配
                anc_xy = anc[:, :2]                     # (P,2)
                gt_xy = gt[:, :2]                       # (M,2)
                dist = torch.cdist(anc_xy, gt_xy)       # (P,M)
                for m in range(M):
                    c = int(gt[m, 7].item())
                    mask = (anchor_cls == c) & (dist[:, m] < pos_thr)
                    if mask.sum() == 0:
                        # 至少匹配最近的锚框
                        mask = torch.zeros(P, dtype=torch.bool, device=device)
                        mask[torch.argmin(dist[:, m])] = True
                    cls_t[mask] = 1.0
                    idx = torch.where(mask)[0]
                    res = encode_boxes(gt[m, :7].unsqueeze(0).expand(len(idx), 7), anc[idx])
                    reg_t[idx] = res
                    # 朝向 bin: 与锚框朝向差是否超过 90 度
                    ang_diff = gt[m, 6] - anc[idx, 6]
                    bin_t = (torch.cos(ang_diff) < 0).long()
                    dir_t[idx, 0] = (1 - bin_t).float()
                    dir_t[idx, 1] = bin_t.float()
                    pos_m |= mask
            cls_targets.append(cls_t)
            reg_targets.append(reg_t)
            dir_targets.append(dir_t)
            pos_masks.append(pos_m)
            reg_anchors.append(anc)

        cls_t = torch.stack(cls_targets)        # (B,P)
        reg_t = torch.stack(reg_targets)        # (B,P,7)
        dir_t = torch.stack(dir_targets)        # (B,P,2)
        pos_m = torch.stack(pos_masks)          # (B,P)

        # 分类损失: sigmoid focal + 官方 OpenPCDet 归一化口径
        #   (pcdet/models/dense_heads/anchor_head_template.py: get_cls_layer_loss)
        #   官方: pos_normalizer = 每样本正样本数 npos, 正负样本共用同一 1/npos,
        #         即 cls_loss = sum(focal) / npos / batch_size
        #   原实现: 正样本按类各自 sum/npos_c 取平均 + 负样本 focal.mean()
        #         (mean 除以全部约 80 万个负样本锚框), 负压被稀释约 N/npos 倍,
        #         与 CenterHead 修复前同类问题; 且按类归一化 ≠ 官方全局 npos。
        focal = sigmoid_focal_loss(cls_flat, cls_t)                    # (B,P)
        npos_b = (cls_t == 1).sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
        cls_loss = (focal / npos_b).sum() / B

        # 正样本回归与朝向损失
        n_pos = pos_m.sum().clamp(min=1).float()
        if pos_m.any():
            reg_loss = F.smooth_l1_loss(reg_flat[pos_m], reg_t[pos_m], reduction="sum") / n_pos
            dir_loss = (F.cross_entropy(dir_flat[pos_m], dir_t[pos_m], reduction="none")
                        .sum()) / n_pos
        else:
            reg_loss = torch.tensor(0.0, device=device)
            dir_loss = torch.tensor(0.0, device=device)
        return cls_loss, reg_loss, dir_loss

    @torch.no_grad()
    def predict(self, cls_map, reg_map, dir_map, anchors, score_thr=0.1,
                nms_dist=0.6, max_det=100, pre_nms_topk=2000):
        """解码预测框
        返回每样本: dict(cls ids (M,), boxes (M,7) numpy, scores (M,))
        注: NMS 前先取分数最高的 pre_nms_topk 个候选再做贪心 NMS。
        避免在数十万低分锚框上做纯 Python 循环 (否则 CPU 爆炸)。
        分数最高的候选不受此截断影响, 最终结果与全量 NMS 基本一致。
        """
        B, A, X, Y = cls_map.shape
        device = cls_map.device
        anc = anchors.view(-1, 7).to(device)
        cls_flat = torch.sigmoid(cls_map.permute(0, 2, 3, 1).reshape(B, -1))  # (B,P)
        reg_flat = reg_map.permute(0, 2, 3, 1).reshape(B, -1, 7)
        dir_flat = dir_map.permute(0, 2, 3, 1).reshape(B, -1, 2)
        # 与 loss 保持一致: anchors 展平顺序 (x,y,a) -> 类别 = (p % A) // 2
        anchor_cls = (torch.arange(A, device=device) // 2).repeat(X * Y)  # (P,)

        results = []
        for b in range(B):
            conf = cls_flat[b]                          # (P,)
            cls_id = anchor_cls
            # 先截断到分数最高的 pre_nms_topk 个候选 (GPU), 再做 NMS
            k = min(pre_nms_topk, conf.numel())
            top_conf, top_i = torch.topk(conf, k)
            sel = top_conf > score_thr
            if sel.sum() == 0:
                results.append({"cls": np.zeros(0, dtype=int),
                                "boxes": np.zeros((0, 7), dtype=np.float32),
                                "scores": np.zeros(0)})
                continue
            idx = top_i[sel]
            res = decode_boxes(reg_flat[b][idx], anc[idx])
            # 朝向修正: dir bin=1 时 yaw 加 pi
            drc = dir_flat[b][idx]
            flip = drc[:, 1] > drc[:, 0]
            yaw = res[:, 6] + flip.float() * math.pi
            boxes = torch.cat([res[:, :6], yaw.unsqueeze(1)], dim=1)

            conf_np = conf[idx].cpu().numpy()
            cls_np = cls_id[idx].cpu().numpy()
            box_np = boxes.cpu().numpy()
            # 按类别的距离 NMS (贪心)
            order = np.argsort(-conf_np)
            final = []
            suppressed = np.zeros(len(order), dtype=bool)
            for ii in range(len(order)):
                if suppressed[ii]:
                    continue
                i = order[ii]
                final.append(i)
                if len(final) >= max_det:
                    break
                d = np.linalg.norm(box_np[order[ii + 1:], :2] - box_np[i, :2], axis=1)
                same = cls_np[order[ii + 1:]] == cls_np[i]
                suppressed[ii + 1:] |= (d < nms_dist) & same
            final = final[:max_det]
            results.append({"cls": cls_np[final],
                            "boxes": box_np[final].astype(np.float32),
                            "scores": conf_np[final].astype(np.float32)})
        return results


class CenterHead(nn.Module):
    """CenterPoint 风格无锚框检测头 (Yin et al., 2021)
    每类别一张 BEV 热力图找峰值定位物体, 8 维回归通道:
      0:2 胞内偏移 (x,y), 2: 高度 z, 3:6 尺寸 log(w,l,h), 6:8 朝向 sin/cos
    损失: penalty-reduced focal (CenterFocal) + 正样本 L1
    与锚框头的对照: 无先验框、无锚框-GT 匹配, 中心点即预测
    """

    CELL = 0.5    # 米/格 (POINT_RANGE 100m / 200 格), 与 build_anchors 网格一致
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
                hm_t[b, c, ix, iy] = 1.0
                # 胞内偏移目标 (0~1), 解码 x = (ix+off)*CELL + X0
                reg_t[b, 0, ix, iy] = fx - ix
                reg_t[b, 1, ix, iy] = fy - iy
                reg_t[b, 2, ix, iy] = gt[m, 2]      # z 直接回归
                reg_t[b, 3:6, ix, iy] = torch.log(torch.clamp(gt[m, 3:6], min=1e-3))
                reg_t[b, 6, ix, iy] = torch.sin(gt[m, 6])
                reg_t[b, 7, ix, iy] = torch.cos(gt[m, 6])

        # CenterFocal (CenterNet 原始定义, Zhou et al. 2019):
        #   正样本 -log(p) 按 (1-p)^2 加权, 负样本 -log(1-p) 按 p^2 加权, 再除以正样本总数 npos
        # 关键: 负样本项必须用 sum()/npos, 不能用 mean()。
        #   用 mean() 时负压被总格点数 N(≈10*200*200=40万) 稀释约 N/npos≈1.3万倍,
        #   假阳性无法压制 -> 热力图成片高响应 -> 每帧数百个乱框。
        #   8 帧对照实验(仅改此处): 预测框命中率 21%->100%, 每帧框数 200(截满)->54(GT=56),
        #   召回 73%->96%。
        bce = F.binary_cross_entropy_with_logits(hm, hm_t, reduction="none")
        p = hm.sigmoid()
        pm_c = hm_t == 1
        nm_c = ~pm_c
        pos_loss = (bce * (1 - p).pow(2))[pm_c].sum()
        neg_loss = (bce * p.pow(2))[nm_c].sum()
        npos = pm_c.sum().clamp(min=1).float()
        cls_loss = (pos_loss + neg_loss) / npos

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
        返回每样本 dict(cls ids, boxes (M,7), scores), 与 AnchorHead.predict 同构
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
            # 同类距离 NMS (贪心, 与 AnchorHead.predict 一致)
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


class PointPillars(nn.Module):
    """PointPillars: PFN -> 伪图像 -> 2D CNN -> 检测头
    head="anchor": SECOND 风格锚框头 (默认); head="center": CenterPoint 风格无锚框头
    两者共用同一骨干, 便于课上对比锚框 vs 无锚框范式"""

    def __init__(self, anchor_stats, in_feat=9, n_classes=len(CLASS_NAMES), head="anchor"):
        super().__init__()
        self.n_anchor = n_classes * 2
        self.head_type = head
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
        self.head = (AnchorHead(384, n_classes) if head == "anchor"
                     else CenterHead(384, n_classes))

    @staticmethod
    def _block(cin, cout, stride):
        return nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(cin, cout, 3, stride=stride), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))

    def forward(self, voxels, coords, num_points, grid=(200, 200), return_feat=False):
        """voxels: (B, V, P, 9)  coords: (B, V, 3)  num_points: (B, V)
        返回 (cls, reg, dir) 特征图; return_feat=True 时额外返回 384 通道融合特征 u
        (供两阶段模型的 RoI 精修使用)
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
        x = img
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        u = torch.cat([self.up1(x1), self.up2(x2), self.up3(x3)], dim=1)  # 3*128=384
        if return_feat:
            cls, reg, drc = self.head(u)
            return cls, reg, drc, u
        return self.head(u)


def mask_coords(B, mask, device):
    """返回被 mask 选中的 batch 索引"""
    return torch.arange(B, device=device).view(-1, 1).expand_as(mask)[mask]


class VoxelNet(nn.Module):
    """VoxelNet: VFE -> 稀疏张量稠密化 -> 3D CNN -> 2D 锚框头"""

    def __init__(self, anchor_stats, in_feat=9, n_classes=len(CLASS_NAMES)):
        super().__init__()
        # VFE
        self.vfe = nn.Sequential(
            nn.Linear(in_feat, 64), nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True))
        # 3D 卷积骨干 (D=8 z, 200x200 xy)
        self.c1 = nn.Sequential(
            nn.Conv3d(64, 64, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))     # (64,8,100,100)
        self.c2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True))    # (128,4,50,50)
        self.c3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True))    # (256,2,25,25)
        # z 维收敛到 1
        self.z1 = nn.Conv3d(64, 64, (8, 1, 1))
        self.z2 = nn.Conv3d(128, 128, (4, 1, 1))
        self.z3 = nn.Conv3d(256, 256, (2, 1, 1))
        # 2D 上采样到 200x200
        self.up1 = nn.ConvTranspose2d(64, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 128, 4, stride=4)
        self.up3 = nn.ConvTranspose2d(256, 128, 8, stride=8)
        self.head = AnchorHead(384, n_classes)

    def forward(self, voxels, coords, num_points, grid=(200, 200, 8)):
        B, V, P, F_ = voxels.shape
        device = voxels.device
        mask = num_points > 0
        v = voxels[mask]
        n = num_points[mask]
        N = v.shape[0]
        feat = self.vfe(v.reshape(-1, F_)).reshape(N, P, -1)
        feat = feat * (torch.arange(P, device=device).view(1, P, 1) < n.view(-1, 1, 1)).float()
        feat = feat.max(dim=1).values               # (N,64)

        dense = torch.zeros(B, 64, grid[2], grid[0], grid[1], device=device)
        bx = mask_coords(B, mask, device)
        ix, iy, iz = coords[mask, 0].long(), coords[mask, 1].long(), coords[mask, 2].long()
        dense[bx, :, iz, ix, iy] = feat

        y1 = self.c1(dense)      # (B,64,8,100,100)
        y2 = self.c2(y1)         # (B,128,4,50,50)
        y3 = self.c3(y2)         # (B,256,2,25,25)
        z1 = self.z1(y1).squeeze(2)   # (B,64,100,100)
        z2 = self.z2(y2).squeeze(2)   # (B,128,50,50)
        z3 = self.z3(y3).squeeze(2)   # (B,256,25,25)
        u = torch.cat([self.up1(z1), self.up2(z2), self.up3(z3)], dim=1)  # 3*128=384
        return self.head(u)
