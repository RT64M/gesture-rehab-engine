"""
三层特征描述符提取。

第一层：归一化坐标（42维，21个点 × 2）—— 完整保留几何信息
第二层：关节角度（15维）—— 医学可解释，对应 ROM
第三层：指间几何关系（11维 = 7个指尖距离 + 4个展开角）—— 捕捉手指配合

默认对外的"标准特征向量" = 第二层 + 第三层 = 26 维
这是评分引擎实际使用的特征维度（紧凑且医学可解释）。

如果需要做诊断或可视化，可以单独取出第一层。
"""
import numpy as np
from dataclasses import dataclass

from . import config
from .geometry import (
    normalize_landmarks,
    pca_align_2d,
    all_joint_angles,
    pairwise_distances,
    joint_angle,
)


@dataclass
class FeatureBundle:
    """单个样本的完整特征束（三层都保留，方便后续选择）。"""
    coords: np.ndarray         # (21, 2) 归一化+对齐后的坐标
    joint_angles: np.ndarray   # (15,)   关节角度（弧度）
    tip_distances: np.ndarray  # (7,)    指尖距离（已归一化）
    spread_angles: np.ndarray  # (4,)    展开角（弧度）

    @property
    def geometric_vector(self) -> np.ndarray:
        """对外的 26 维标准特征向量（第二层 + 第三层）。"""
        return np.concatenate([
            self.joint_angles,
            self.tip_distances,
            self.spread_angles,
        ])

    @property
    def full_vector(self) -> np.ndarray:
        """完整 68 维向量（包含坐标，仅供研究/调试用）。"""
        return np.concatenate([
            self.coords.flatten(),
            self.joint_angles,
            self.tip_distances,
            self.spread_angles,
        ])


def extract_features(
    raw_landmarks: np.ndarray,
    apply_pca_alignment: bool = True,
) -> FeatureBundle:
    """
    从一组原始关键点提取三层特征。

    Parameters
    ----------
    raw_landmarks : np.ndarray, shape (21, 2) 或 (21, 3)
        来自 HaGRIDv2 的图像归一化坐标，或来自 MediaPipe 的实时坐标。
    apply_pca_alignment : bool
        是否做 PCA 旋转对齐。开启后，左/右手、各种朝向的手都会被对齐到统一姿态。
        对评分有利；但如果想保留"手的朝向"作为特征，可以关闭。

    Returns
    -------
    FeatureBundle
    """
    # 第一步：尺度 + 平移归一化
    pts = normalize_landmarks(raw_landmarks)

    # 第二步：旋转对齐
    if apply_pca_alignment:
        pts = pca_align_2d(pts)

    # 第二层：关节角度
    angles = all_joint_angles(pts)

    # 第三层：指间几何
    tip_dist = pairwise_distances(pts, config.TIP_PAIRS)
    spread = np.array([
        joint_angle(pts, a, b, c)
        for (a, b, c) in config.SPREAD_TRIPLETS.values()
    ])

    return FeatureBundle(
        coords=pts[:, :2],  # 只保留 2D
        joint_angles=angles,
        tip_distances=tip_dist,
        spread_angles=spread,
    )


def batch_extract_geometric(
    landmarks_array: np.ndarray,
    apply_pca_alignment: bool = True,
) -> np.ndarray:
    """
    批量提取 26 维几何特征向量。

    Parameters
    ----------
    landmarks_array : np.ndarray, shape (N, 21, 2)

    Returns
    -------
    np.ndarray, shape (N, 26)
    """
    if len(landmarks_array) == 0:
        return np.empty((0, config.N_GEOMETRIC_FEATURES), dtype=np.float64)
    feats = np.zeros((len(landmarks_array), config.N_GEOMETRIC_FEATURES), dtype=np.float64)
    for i, lm in enumerate(landmarks_array):
        feats[i] = extract_features(lm, apply_pca_alignment).geometric_vector
    return feats


def feature_names() -> list[str]:
    """返回 26 维特征向量每一维的语义名称（用于可视化、报告）。"""
    return (
        list(config.JOINT_TRIPLETS.keys())
        + [f"dist_{k}" for k in config.TIP_PAIRS.keys()]
        + list(config.SPREAD_TRIPLETS.keys())
    )
