from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config
from src.demo_profiles import nearest_competitor
from src.descriptors import GestureDescriptor


CURVE_NAMES = [
    "linear_recovery",
    "s_curve_recovery",
    "plateau",
    "fatigue_dip",
]


def _stable_seed(*parts: object) -> int:
    total = 0
    for part in parts:
        for ch in str(part):
            total = (total * 131 + ord(ch)) % (2**32)
    return total


def curve_progress(curve: str, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if curve == "linear_recovery":
        value = t
    elif curve == "s_curve_recovery":
        value = 1.0 / (1.0 + np.exp(-8.0 * (t - 0.5)))
        value = (value - 0.01798620996209156) / (0.9820137900379085 - 0.01798620996209156)
    elif curve == "plateau":
        value = min(t * 1.65, 0.72)
    elif curve == "fatigue_dip":
        dip = np.exp(-((t - 0.72) ** 2) / 0.006) * 0.26
        recovery_boost = max(t - 0.84, 0.0) * 0.18
        value = t - dip + recovery_boost
    else:
        raise KeyError(f"Unknown recovery curve: {curve}")
    return float(np.clip(value, 0.0, 1.0))


@dataclass
class PatientAttempt:
    patient_id: str
    recovery_curve: str
    gesture: str
    session_index: int
    attempt_index: int
    base_ability: float
    realized_ability: float
    dominant_features: list[str]
    vector: np.ndarray


class VirtualPatient:
    """Synthetic patient with a latent recovery trajectory and stable deficits."""

    def __init__(
        self,
        patient_id: str,
        healthy_descriptors: dict[str, GestureDescriptor],
        recovery_curve: str,
        noise_level: float = 0.08,
        initial_impairment: float = 1.35,
        responsiveness: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.patient_id = patient_id
        self.healthy_descriptors = healthy_descriptors
        self.recovery_curve = recovery_curve
        self.noise_level = float(noise_level)
        self.initial_impairment = float(initial_impairment)
        self.responsiveness = float(responsiveness)
        self.seed = int(_stable_seed(patient_id, recovery_curve) if seed is None else seed)
        self._rng = np.random.default_rng(self.seed)
        self._training_offset = 0.0
        self._profiles = {
            gesture: self._build_gesture_profile(gesture)
            for gesture in healthy_descriptors
        }

    def _build_gesture_profile(self, gesture: str) -> dict:
        descriptor = self.healthy_descriptors[gesture]
        competitor = nearest_competitor(gesture, self.healthy_descriptors)
        competitor_descriptor = self.healthy_descriptors[competitor]
        sigma_diag = np.sqrt(np.clip(np.diag(descriptor.sigma), 1e-8, None))
        delta = competitor_descriptor.mu - descriptor.mu
        direction = np.sign(delta)
        direction[direction == 0.0] = 1.0

        weights = np.full(len(descriptor.feature_names), 0.12, dtype=np.float64)
        angle_indices = [descriptor.feature_names.index(name) for name in config.JOINT_NAMES]
        top_angle_indices = sorted(angle_indices, key=lambda idx: abs(delta[idx]), reverse=True)
        dominant_count = min(3, len(top_angle_indices))
        dominant_indices = top_angle_indices[:dominant_count]
        secondary_indices = top_angle_indices[dominant_count : dominant_count + 3]

        for index in dominant_indices:
            weights[index] = self._rng.uniform(0.95, 1.35)
        for index in secondary_indices:
            weights[index] = self._rng.uniform(0.35, 0.65)

        if len(weights) > config.N_JOINT_ANGLES:
            weights[config.N_JOINT_ANGLES :] *= 0.45

        deficit_vector = direction * sigma_diag * weights
        dominant_features = [descriptor.feature_names[index] for index in dominant_indices]
        return {
            "deficit_vector": deficit_vector,
            "dominant_features": dominant_features,
        }

    def clone(self, suffix: str | None = None) -> "VirtualPatient":
        patient_id = self.patient_id if suffix is None else f"{self.patient_id}:{suffix}"
        return VirtualPatient(
            patient_id=patient_id,
            healthy_descriptors=self.healthy_descriptors,
            recovery_curve=self.recovery_curve,
            noise_level=self.noise_level,
            initial_impairment=self.initial_impairment,
            responsiveness=self.responsiveness,
            seed=self.seed,
        )

    def base_ability_at(self, session_index: int, total_sessions: int) -> float:
        denominator = max(total_sessions - 1, 1)
        return curve_progress(self.recovery_curve, session_index / denominator)

    def ability_at(self, session_index: int, total_sessions: int) -> float:
        base = self.base_ability_at(session_index, total_sessions)
        return float(np.clip(base + self._training_offset, 0.0, 1.0))

    def apply_training_feedback(self, quality: float) -> None:
        step = float(np.clip(quality, -1.0, 1.0)) * self.responsiveness
        self._training_offset = float(np.clip(self._training_offset + step, -0.18, 0.25))

    def sample_attempt(
        self,
        gesture: str,
        session_index: int,
        total_sessions: int,
        attempt_index: int = 0,
    ) -> PatientAttempt:
        descriptor = self.healthy_descriptors[gesture]
        profile = self._profiles[gesture]
        base_ability = self.base_ability_at(session_index, total_sessions)
        realized_ability = self.ability_at(session_index, total_sessions)
        residual = self.initial_impairment * (1.0 - realized_ability)
        center = descriptor.mu + residual * profile["deficit_vector"]

        sigma_diag = np.sqrt(np.clip(np.diag(descriptor.sigma), 1e-8, None))
        noise = self._rng.normal(
            loc=0.0,
            scale=np.maximum(sigma_diag * self.noise_level, 1e-4),
            size=len(center),
        )
        vector = center + noise

        return PatientAttempt(
            patient_id=self.patient_id,
            recovery_curve=self.recovery_curve,
            gesture=gesture,
            session_index=int(session_index),
            attempt_index=int(attempt_index),
            base_ability=float(base_ability),
            realized_ability=float(realized_ability),
            dominant_features=list(profile["dominant_features"]),
            vector=np.asarray(vector, dtype=np.float64),
        )

    def perform_gesture(
        self,
        gesture: str,
        session_index: int,
        total_sessions: int,
        attempt_index: int = 0,
    ) -> np.ndarray:
        return self.sample_attempt(gesture, session_index, total_sessions, attempt_index).vector
