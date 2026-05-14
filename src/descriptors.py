"""
标准手势描述符 (Standard Gesture Descriptor)

每个手势 g 由一个二元组 D_g = (μ_g, Σ_g) 描述，其中
    μ_g ∈ R^d : 该手势的"平均特征向量"（理想姿态）
    Σ_g ∈ R^{d×d} : 该手势内部的协方差矩阵（描述哪些维度本来就严格、哪些宽松）

新观测 x 与该手势的相似度用马氏距离衡量：
    d_M(x; D_g) = sqrt( (x - μ_g)^T Σ_g^{-1} (x - μ_g) )

为什么用马氏而不是欧氏？
    - 不同特征量纲不同（弧度 vs 归一化距离），欧氏距离会偏向数值大的维度
    - 不同手势对不同关节"严格度"不同。比如握拳要求所有手指都弯曲（方差小，要求严），
      OK 手势的小指方向自由（方差大，要求宽）。马氏距离自动赋予不同维度合适的权重。

冷启动评分公式（论文/报告里要写的）：
    score(x; g) = exp( - d_M(x; D_g) / τ )    ∈ (0, 1]
    其中 τ 是温度参数，控制评分曲线的陡峭度。

后续的"自适应评分"会更新这个 D_g —— 但每个用户的初始 D_g 都来自这里。
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


@dataclass
class GestureDescriptor:
    """一个手势的标准数学描述符。"""
    gesture: str
    n_samples: int
    mu: np.ndarray              # (d,)
    sigma: np.ndarray           # (d, d)
    sigma_inv: np.ndarray       # (d, d) 预先求逆，加速评分
    feature_names: list[str]    # 每一维的名称

    def mahalanobis(self, x: np.ndarray) -> float:
        """计算 x 相对于该描述符的马氏距离。"""
        diff = x - self.mu
        return float(np.sqrt(diff @ self.sigma_inv @ diff))

    def score(self, x: np.ndarray, tau: float = 3.0) -> float:
        """
        冷启动评分：score ∈ (0, 1]。
        tau 越小，评分越严格（小偏差也会大幅扣分）。
        默认 tau=3.0 是个温和的起点，后面可以做敏感性分析。
        """
        d = self.mahalanobis(x)
        return float(np.exp(-d / tau))

    def to_dict(self) -> dict:
        """序列化（np.ndarray 转 list 以便 JSON 存盘）。"""
        return {
            "gesture": self.gesture,
            "n_samples": self.n_samples,
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "feature_names": self.feature_names,
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "GestureDescriptor":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        mu = np.array(d["mu"])
        sigma = np.array(d["sigma"])
        sigma_inv = np.linalg.inv(sigma)
        return cls(
            gesture=d["gesture"],
            n_samples=d["n_samples"],
            mu=mu,
            sigma=sigma,
            sigma_inv=sigma_inv,
            feature_names=d["feature_names"],
        )


def build_descriptor(
    gesture: str,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    shrinkage: float = 0.05,
) -> GestureDescriptor:
    """
    从一组样本特征拟合标准描述符。

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (N, d)
        某手势的所有样本特征向量。
    shrinkage : float ∈ [0, 1]
        协方差矩阵的收缩正则化（Ledoit-Wolf 风格的简化版）：
            Σ_reg = (1 - α) * Σ + α * (tr(Σ)/d) * I
        作用：
            1. 防止 Σ 奇异（当 N 小或维度间共线时 Σ 不可逆）
            2. 缓解小样本下协方差估计的过拟合
        默认 0.05 是一个温和的正则化强度。
    """
    if feature_matrix.shape[0] < 10:
        raise ValueError(f"样本数过少 (N={feature_matrix.shape[0]})，无法稳定估计协方差")

    mu = feature_matrix.mean(axis=0)
    sigma_raw = np.cov(feature_matrix, rowvar=False)

    # 正则化
    d = sigma_raw.shape[0]
    target = (np.trace(sigma_raw) / d) * np.eye(d)
    sigma = (1 - shrinkage) * sigma_raw + shrinkage * target

    sigma_inv = np.linalg.inv(sigma)

    return GestureDescriptor(
        gesture=gesture,
        n_samples=feature_matrix.shape[0],
        mu=mu,
        sigma=sigma,
        sigma_inv=sigma_inv,
        feature_names=feature_names,
    )


def classify(x: np.ndarray, descriptors: dict[str, GestureDescriptor]) -> tuple[str, dict]:
    """
    用一组描述符对单个观测做最近邻分类（基于马氏距离）。

    Returns
    -------
    best_gesture : str
    distances : dict[gesture -> mahalanobis distance]
    """
    distances = {g: d.mahalanobis(x) for g, d in descriptors.items()}
    best = min(distances, key=distances.get)
    return best, distances
