from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .descriptors import GestureDescriptor


def _safe_covariance(samples: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if len(samples) < 2:
        return fallback.copy()
    covariance = np.cov(samples, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]], dtype=np.float64)
    return covariance


@dataclass
class UserGestureModel:
    gesture: str
    base_descriptor: GestureDescriptor
    expected_distance: float = 1.0
    tau0: float = 5.0
    adaptive_window: int = 10
    n_transition: int = 30
    lambda_max: float = 0.8
    covariance_floor: float = 1e-4
    covariance_shrinkage: float = 0.1
    _samples: list[np.ndarray] = field(default_factory=list)
    _baseline_distances: list[float] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def blend_weight(self) -> float:
        if self.n_transition <= 0:
            return self.lambda_max
        return float(min(self.n_samples / self.n_transition, self.lambda_max))

    @property
    def adaptive_factor(self) -> float:
        if not self._baseline_distances or self.expected_distance <= 0:
            return 1.0
        window = self._baseline_distances[: self.adaptive_window]
        median_distance = float(np.median(window))
        factor = median_distance / self.expected_distance
        return float(np.clip(factor, 0.5, 2.0))

    @property
    def tau(self) -> float:
        return float(self.tau0 * self.adaptive_factor)

    @property
    def user_mean(self) -> np.ndarray:
        if not self._samples:
            return self.base_descriptor.mu.copy()
        return np.mean(np.vstack(self._samples), axis=0)

    @property
    def user_covariance(self) -> np.ndarray:
        if not self._samples:
            return self.base_descriptor.sigma.copy()
        samples = np.vstack(self._samples)
        covariance = _safe_covariance(samples, self.base_descriptor.sigma)
        covariance = (
            (1.0 - self.covariance_shrinkage) * covariance
            + self.covariance_shrinkage * self.base_descriptor.sigma
        )
        covariance = covariance + self.covariance_floor * np.eye(covariance.shape[0])
        return covariance

    def effective_descriptor(self) -> GestureDescriptor:
        weight = self.blend_weight
        if weight <= 0.0:
            return self.base_descriptor
        mu = (1.0 - weight) * self.base_descriptor.mu + weight * self.user_mean
        sigma = (1.0 - weight) * self.base_descriptor.sigma + weight * self.user_covariance
        sigma = sigma + self.covariance_floor * np.eye(sigma.shape[0])
        sigma_inv = np.linalg.inv(sigma)
        return GestureDescriptor(
            gesture=self.gesture,
            n_samples=max(self.base_descriptor.n_samples, self.n_samples),
            mu=mu,
            sigma=sigma,
            sigma_inv=sigma_inv,
            feature_names=list(self.base_descriptor.feature_names),
        )

    def record_attempt(self, feature_vector: np.ndarray) -> None:
        feature_vector = np.asarray(feature_vector, dtype=np.float64)
        self._samples.append(feature_vector.copy())
        self._baseline_distances.append(self.base_descriptor.mahalanobis(feature_vector))

