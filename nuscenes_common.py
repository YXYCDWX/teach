# -*- coding: utf-8 -*-
"""nuScenes v1.0-mini 公共解析模块
直接解析 JSON 表, 不依赖 nuscenes-devkit。
nuScenes 坐标系约定:
  - 全局坐标系 (global): z 轴向上
  - 传感器坐标系 -> 车体坐标系: calibrated_sensor (translation + rotation)
  - 车体坐标系 -> 全局坐标系: ego_pose (translation + rotation)
"""
import os
import json
import math
import numpy as np

# nuScenes 检测任务的 10 类, 统一类别顺序 (2D/3D 共用)
CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
CLSID = {n: i for i, n in enumerate(CLASS_NAMES)}

# nuScenes category 前缀 -> 统一类别名
CATEGORY_MAP = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}


class NuScenesTables:
    """加载并索引 v1.0-mini 的所有 JSON 表"""

    def __init__(self, dataroot):
        self.dataroot = dataroot
        table_dir = os.path.join(dataroot, "v1.0-mini")
        self.tables = {}
        for name in ["category", "attribute", "visibility", "scene", "log",
                     "sensor", "calibrated_sensor", "ego_pose", "sample",
                     "sample_data", "sample_annotation", "instance"]:
            with open(os.path.join(table_dir, name + ".json"), "r", encoding="utf-8") as f:
                self.tables[name] = json.load(f)
        # 按 token 建索引
        self.index = {}
        for name, rows in self.tables.items():
            self.index[name] = {r["token"]: r for r in rows}
        # 此版本 sample_annotation 无 category_name 字段, 通过 instance -> category 关联补齐
        inst_cat = {r["token"]: r["category_token"] for r in self.tables["instance"]}
        cat_name = {r["token"]: r["name"] for r in self.tables["category"]}
        for ann in self.tables["sample_annotation"]:
            ann["category_name"] = cat_name[inst_cat[ann["instance_token"]]]

    # ---------- 通用查询 ----------
    def get(self, table, token):
        return self.index[table][token]

    def list_sample_data(self, sample_token, sensor_channel, keyframe_only=True):
        """返回某 sample 下某传感器的 sample_data 记录
        注: 此版本 sample_data 表无 channel 字段, 用 filename 目录名匹配
        """
        out = []
        sd = self.tables["sample_data"]
        for r in sd:
            if r["sample_token"] != sample_token:
                continue
            folder = os.path.dirname(r["filename"]).replace("/", os.sep).split(os.sep)[-1]
            if folder != sensor_channel:
                continue
            if keyframe_only and not r["is_key_frame"]:
                continue
            out.append(r)
        return out

    def sample_annotations(self, sample_token):
        return [a for a in self.tables["sample_annotation"]
                if a["sample_token"] == sample_token]

    def scenes(self):
        return self.tables["scene"]

    def scene_samples(self, scene_token):
        """按顺序返回某 scene 的所有 keyframe sample token"""
        scene = self.get("scene", scene_token)
        tokens = []
        cur = scene["first_sample_token"]
        while cur:
            s = self.get("sample", cur)
            tokens.append(cur)
            cur = s["next"]
        return tokens


# ---------- 变换工具 ----------
def quaternion_to_matrix(q):
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵"""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def make_transform(translation, rotation):
    """translation + rotation -> 4x4 齐次变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_matrix(rotation)
    T[:3, 3] = translation
    return T


def transform_matrix(cs_record, ep_record):
    """传感器坐标系 -> 全局坐标系 的 4x4 矩阵"""
    T_sensor_ego = make_transform(cs_record["translation"], cs_record["rotation"])
    T_ego_global = make_transform(ep_record["translation"], ep_record["rotation"])
    return T_ego_global @ T_sensor_ego


def global_to_sensor(points, T_sensor_from_global):
    """全局点 (N,3) -> 传感器坐标系"""
    R = T_sensor_from_global[:3, :3]
    t = T_sensor_from_global[:3, 3]
    return points @ R.T + t


def box_corners(center, size, yaw):
    """中心 (3,), 尺寸 (w,l,h), 偏航角 yaw (绕 z 轴) -> 8 个角点 (8,3)
    nuScenes 约定: 长 l 沿局部 x 轴 (朝向方向), 宽 w 沿局部 y 轴, h 沿 z 轴
    返回角点顺序: 底面 4 点 (z-h/2) + 顶面 4 点 (z+h/2)
    """
    w, l, h = size
    # 局部角点 (x: ±l/2 沿朝向, y: ±w/2 沿横向)
    x_sign = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64)
    y_sign = np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float64)
    z_sign = np.array([-1] * 4 + [1] * 4, dtype=np.float64)
    corners = np.stack([x_sign * l / 2, y_sign * w / 2, z_sign * h / 2], axis=1)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return corners @ Rz.T + np.asarray(center)


def ann_box_in_frame(ann, T_sensor_from_global, T_global_from_ego, T_ego_from_sensor):
    """把 sample_annotation 的 3D 框变换到目标传感器坐标系
    返回 (center(3,), size(w,l,h), yaw, cls_id); 类别不在映射表内返回 None
    """
    cls_id = CLSID.get(CATEGORY_MAP.get(ann["category_name"], ""), None)
    if cls_id is None:
        return None
    # 注意: sample_annotation 的 translation/rotation 已是全局坐标系, 无需再变换
    center_global = np.asarray(ann["translation"], dtype=np.float64)
    R_global = quaternion_to_matrix(ann["rotation"])
    # 全局 -> 目标传感器系
    R_inv = T_sensor_from_global[:3, :3]
    t_inv = T_sensor_from_global[:3, 3]
    center = R_inv @ center_global + t_inv
    R_local = R_inv @ R_global
    # 提取绕 z 轴偏航角
    yaw = math.atan2(R_local[1, 0], R_local[0, 0])
    size = ann["size"]  # (w, l, h)
    return center, np.asarray(size, dtype=np.float64), yaw, cls_id


def frames_for_sample_data(tables, sd_record):
    """为 sample_data 构造: 全局->该传感器 的 4x4 矩阵 + 相机内参 + 车体系相关矩阵"""
    cs = tables.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    ep = tables.get("ego_pose", sd_record["ego_pose_token"])
    T_ego_from_sensor = make_transform(cs["translation"], cs["rotation"])
    T_global_from_ego = make_transform(ep["translation"], ep["rotation"])
    T_sensor_from_global = np.linalg.inv(T_global_from_ego @ T_ego_from_sensor)
    K = np.asarray(cs["camera_intrinsic"], dtype=np.float64) if cs["camera_intrinsic"] else None
    return {
        "T_sensor_from_global": T_sensor_from_global,
        "T_ego_from_sensor": T_ego_from_sensor,
        "T_global_from_ego": T_global_from_ego,
        "K": K,
        "cs": cs,
        "ep": ep,
    }
