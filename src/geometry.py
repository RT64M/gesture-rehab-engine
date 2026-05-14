"""
几何工具：实现尺度/平移/旋转不变的关键点归一化与角度计算。

数学原理
--------
设原始关键点为 P = {p_0, p_1, ..., p_20}, 每个 p_i ∈ R^2 (或 R^3).

第一步 平移不变：以手腕 p_0 为原点
    P' = P - p_0

第二步 尺度不变：用参考骨长 ||p_9 - p_0|| 归一化
    s = ||p_9 - p_0||
    P'' = P' / s
    （手腕到中指 MCP 的距离是手掌内最稳定的骨段，几乎不随手指弯曲变化）

第三步 旋转不变（可选）：PCA 主轴对齐
    用 P'' 的协方差矩阵的主特征向量作为新坐标系的 y 轴
    这样让"手指方向"始终朝上，消除手在画面里的旋转

第四步 关节角度（自动满足上述三种不变性）：
    对三元组 (a, b, c), 计算 b 处的夹角：
        v1 = p_a - p_b
        v2 = p_c - p_b
        θ  = arccos( v1·v2 / (|v1||v2|) )
"""
import numpy as np
from . import config


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    对一组关键点做平移 + 尺度归一化。

    Parameters
    ----------
    landmarks : np.ndarray, shape (21, 2) 或 (21, 3)
        原始关键点坐标。

    Returns
    -------
    np.ndarray, shape 同输入
        归一化后的坐标，手腕在原点，参考骨长为 1。
    """
    pts = np.asarray(landmarks, dtype=np.float64).copy()

    # 平移：手腕到原点
    wrist = pts[config.WRIST]
    pts -= wrist

    # 尺度：参考骨长归一化
    ref_vec = pts[config.SCALE_REF_TO] - pts[config.SCALE_REF_FROM]
    scale = np.linalg.norm(ref_vec)
    if scale < 1e-8:
        # 异常情况：手腕和中指 MCP 重合（几乎不可能，但要防御）
        return pts
    pts /= scale
    return pts


def pca_align_2d(landmarks_2d: np.ndarray) -> np.ndarray:
    """
    用 PCA 对齐：让手指主方向朝上（+y 方向），消除画面内旋转。

    注意：这一步对 2D 关键点做。如果输入是 3D，仅对 (x, y) 做对齐，z 保持不变。

    Parameters
    ----------
    landmarks_2d : np.ndarray, shape (21, 2) 或 (21, 3)

    Returns
    -------
    np.ndarray, 同形状
    """
    pts = np.asarray(landmarks_2d, dtype=np.float64).copy()
    xy = pts[:, :2]

    # 对手指部分的关键点（排除手腕）做 PCA
    finger_pts = xy[1:]  # 20 个点
    centered = finger_pts - finger_pts.mean(axis=0)

    cov = np.cov(centered.T)  # 2x2
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh 返回升序，主轴是最后一列
    principal_axis = eigvecs[:, -1]  # shape (2,)

    # 我们希望主轴指向 "手指张开方向"，即从手腕指向中指 MCP 的方向
    target_dir = xy[config.MIDDLE_MCP] - xy[config.WRIST]
    if np.dot(principal_axis, target_dir) < 0:
        principal_axis = -principal_axis

    # 构造旋转矩阵：让 principal_axis 旋转到 +y 轴 (0, 1)
    # 当前主轴是 (cosθ, sinθ)，要旋转使其变为 (0, 1)
    # 即旋转角度为 π/2 - θ
    cos_t, sin_t = principal_axis[0], principal_axis[1]
    # 旋转矩阵 R 使 R @ principal_axis = (0, 1)
    # R = [[ sin_t, -cos_t],
    #      [ cos_t,  sin_t]]
    R = np.array([[ sin_t, -cos_t],
                  [ cos_t,  sin_t]])

    pts[:, :2] = xy @ R.T  # 应用旋转
    return pts


def joint_angle(landmarks: np.ndarray, a: int, b: int, c: int) -> float:
    """
    计算以 b 为顶点，向量 b->a 与 b->c 的夹角（弧度）。

    自动满足平移、旋转、尺度不变性，因为夹角是纯几何量。
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    v1 = pts[a] - pts[b]
    v2 = pts[c] - pts[b]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def all_joint_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    一次性计算所有 15 个关节角度，按 config.JOINT_NAMES 顺序排列。

    Returns
    -------
    np.ndarray, shape (15,), 弧度
    """
    angles = np.zeros(len(config.JOINT_TRIPLETS), dtype=np.float64)
    for i, (name, (a, b, c)) in enumerate(config.JOINT_TRIPLETS.items()):
        angles[i] = joint_angle(landmarks, a, b, c)
    return angles


def pairwise_distances(landmarks: np.ndarray, pairs: dict) -> np.ndarray:
    """
    计算指定关键点对之间的欧氏距离（已通过 normalize_landmarks 做过尺度归一化的话，
    返回值是相对于参考骨长的比值）。
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    out = np.zeros(len(pairs), dtype=np.float64)
    for i, (name, (p, q)) in enumerate(pairs.items()):
        out[i] = np.linalg.norm(pts[p] - pts[q])
    return out


def is_valid_landmarks(landmarks: np.ndarray,
                       min_valid_ratio: float = config.MIN_VALID_LANDMARK_RATIO) -> bool:
    """
    判断这组关键点是否足够可靠。
    HaGRIDv2 中 MediaPipe 检测失败的样本会出现大量 (0, 0) 坐标。
    """
    pts = np.asarray(landmarks)
    if pts.shape[0] != config.N_LANDMARKS:
        return False
    # 统计非零关键点比例
    norms = np.linalg.norm(pts, axis=1)
    valid_count = np.sum(norms > 1e-6)
    return valid_count / config.N_LANDMARKS >= min_valid_ratio
