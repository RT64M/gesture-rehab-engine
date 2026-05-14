"""
特征不变性的单元测试。

这些测试是论文/报告里的关键论证：我们的特征描述符在数学上严格满足
平移、尺度、旋转不变性。

运行: python -m pytest tests/test_geometry.py -v
也可以直接运行: python tests/test_geometry.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.features import extract_features


def make_synthetic_hand(seed: int = 0) -> np.ndarray:
    """造一只合成的"标准右手"用于测试。坐标是任意构造的，关键是关键点拓扑合理。"""
    rng = np.random.default_rng(seed)
    base = np.array([
        [0.50, 0.95],   # 0  WRIST
        [0.42, 0.85], [0.38, 0.78], [0.34, 0.70], [0.30, 0.62],   # 1-4 thumb
        [0.45, 0.75], [0.43, 0.60], [0.42, 0.50], [0.41, 0.42],   # 5-8 index
        [0.50, 0.74], [0.50, 0.58], [0.50, 0.46], [0.50, 0.36],   # 9-12 middle
        [0.55, 0.75], [0.57, 0.60], [0.58, 0.50], [0.59, 0.42],   # 13-16 ring
        [0.60, 0.78], [0.63, 0.66], [0.65, 0.58], [0.67, 0.50],   # 17-20 pinky
    ])
    # 加微小噪声
    return base + rng.normal(0, 0.002, base.shape)


def apply_transform(pts: np.ndarray, scale: float, theta_rad: float, t: np.ndarray) -> np.ndarray:
    """对关键点施加：先缩放 -> 再旋转 -> 再平移。"""
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad),  np.cos(theta_rad)]])
    return (pts * scale) @ R.T + t


def test_translation_invariance():
    """关节角度对平移完全不变。"""
    hand = make_synthetic_hand()
    f0 = extract_features(hand, apply_pca_alignment=True)

    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(50):
        t = rng.uniform(-10, 10, 2)
        f1 = extract_features(hand + t, apply_pca_alignment=True)
        err = np.max(np.abs(f0.geometric_vector - f1.geometric_vector))
        max_err = max(max_err, err)
    print(f"  [平移不变性]   max abs error = {max_err:.2e}")
    assert max_err < 1e-6, f"平移不变性破坏，误差={max_err}"


def test_scale_invariance():
    """关节角度和归一化距离对尺度不变。"""
    hand = make_synthetic_hand()
    f0 = extract_features(hand, apply_pca_alignment=True)

    max_err = 0.0
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        f1 = extract_features(hand * scale, apply_pca_alignment=True)
        err = np.max(np.abs(f0.geometric_vector - f1.geometric_vector))
        max_err = max(max_err, err)
    print(f"  [尺度不变性]   max abs error = {max_err:.2e}")
    assert max_err < 1e-6, f"尺度不变性破坏，误差={max_err}"


def test_rotation_invariance():
    """开启 PCA 对齐后，特征对旋转不变。"""
    hand = make_synthetic_hand()
    f0 = extract_features(hand, apply_pca_alignment=True)

    rng = np.random.default_rng(2)
    max_err = 0.0
    for _ in range(50):
        theta = rng.uniform(-np.pi, np.pi)
        rotated = apply_transform(hand, scale=1.0, theta_rad=theta, t=np.zeros(2))
        f1 = extract_features(rotated, apply_pca_alignment=True)
        err = np.max(np.abs(f0.geometric_vector - f1.geometric_vector))
        max_err = max(max_err, err)
    print(f"  [旋转不变性]   max abs error = {max_err:.2e}")
    # 旋转涉及浮点 sin/cos，容差略宽
    assert max_err < 1e-4, f"旋转不变性破坏，误差={max_err}"


def test_combined_transform():
    """同时施加平移 + 缩放 + 旋转，特征仍不变。"""
    hand = make_synthetic_hand()
    f0 = extract_features(hand, apply_pca_alignment=True)

    rng = np.random.default_rng(3)
    max_err = 0.0
    for _ in range(100):
        scale = rng.uniform(0.3, 3.0)
        theta = rng.uniform(-np.pi, np.pi)
        t = rng.uniform(-5, 5, 2)
        transformed = apply_transform(hand, scale, theta, t)
        f1 = extract_features(transformed, apply_pca_alignment=True)
        err = np.max(np.abs(f0.geometric_vector - f1.geometric_vector))
        max_err = max(max_err, err)
    print(f"  [组合变换不变性] max abs error = {max_err:.2e} (100次随机试验)")
    assert max_err < 1e-4, f"组合变换不变性破坏，误差={max_err}"


def test_joint_angle_range():
    """关节角度应在 [0, π] 范围内。"""
    hand = make_synthetic_hand()
    f = extract_features(hand)
    assert np.all(f.joint_angles >= 0)
    assert np.all(f.joint_angles <= np.pi)
    assert np.all(f.spread_angles >= 0)
    assert np.all(f.spread_angles <= np.pi)
    print(f"  [角度范围]     OK, 关节角范围 [{f.joint_angles.min():.2f}, {f.joint_angles.max():.2f}] rad")


if __name__ == "__main__":
    print("\n运行不变性测试...\n")
    test_translation_invariance()
    test_scale_invariance()
    test_rotation_invariance()
    test_combined_transform()
    test_joint_angle_range()
    print("\n全部通过 ✓")
