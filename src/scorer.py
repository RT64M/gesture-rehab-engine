from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config
from .descriptors import GestureDescriptor, classify
from .feedback import feedback_payload, generate_feedback, summarize_deviations
from .user_model import UserGestureModel


@dataclass
class ScoreResult:
    target_gesture: str
    predicted_gesture: str
    mahalanobis_distance: float
    raw_score: float
    display_score: int
    matched: bool
    feedback: list[str]
    tau: float
    blend_weight: float
    feature_deviations: list[dict[str, float | str]]
    population_distances: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def load_descriptors(
    descriptor_dir: Path,
    gestures: list[str] | tuple[str, ...] | None = None,
) -> dict[str, GestureDescriptor]:
    allowed = list(gestures) if gestures is not None else list(config.TARGET_GESTURES)
    descriptors: dict[str, GestureDescriptor] = {}
    for gesture in allowed:
        path = descriptor_dir / f"{gesture}.json"
        if not path.exists():
            continue
        descriptors[gesture] = GestureDescriptor.load_json(path)
    if not descriptors:
        raise FileNotFoundError(f"No configured descriptor json files found in {descriptor_dir}")
    return descriptors


def compute_expected_distances(
    descriptors: dict[str, GestureDescriptor],
    features_dir: Path | None = None,
) -> dict[str, float]:
    expected: dict[str, float] = {}
    if features_dir is None:
        return {gesture: 1.0 for gesture in descriptors}

    for gesture, descriptor in descriptors.items():
        feature_path = features_dir / f"{gesture}.npz"
        if not feature_path.exists():
            expected[gesture] = 1.0
            continue
        with np.load(feature_path) as data:
            train = np.asarray(data["train"], dtype=np.float64)
        if len(train) == 0:
            expected[gesture] = 1.0
            continue
        distances = np.array([descriptor.mahalanobis(row) for row in train], dtype=np.float64)
        expected[gesture] = float(np.median(distances))
    return expected


class ColdStartScorer:
    def __init__(
        self,
        descriptors: dict[str, GestureDescriptor],
        expected_distances: dict[str, float] | None = None,
        tau0: float = 5.0,
        gamma: float = 0.6,
        adaptive_window: int = 10,
        n_transition: int = 30,
        lambda_max: float = 0.8,
    ) -> None:
        self.descriptors = descriptors
        self.expected_distances = expected_distances or {gesture: 1.0 for gesture in descriptors}
        self.tau0 = tau0
        self.gamma = gamma
        self.adaptive_window = adaptive_window
        self.n_transition = n_transition
        self.lambda_max = lambda_max
        self._user_models: dict[str, UserGestureModel] = {}

    @classmethod
    def from_artifacts(
        cls,
        descriptor_dir: Path,
        features_dir: Path | None = None,
        **kwargs,
    ) -> "ColdStartScorer":
        descriptors = load_descriptors(descriptor_dir)
        expected_distances = compute_expected_distances(descriptors, features_dir)
        return cls(descriptors=descriptors, expected_distances=expected_distances, **kwargs)

    def get_user_model(self, gesture: str) -> UserGestureModel:
        if gesture not in self.descriptors:
            raise KeyError(f"Unknown gesture: {gesture}")
        if gesture not in self._user_models:
            self._user_models[gesture] = UserGestureModel(
                gesture=gesture,
                base_descriptor=self.descriptors[gesture],
                expected_distance=self.expected_distances.get(gesture, 1.0),
                tau0=self.tau0,
                adaptive_window=self.adaptive_window,
                n_transition=self.n_transition,
                lambda_max=self.lambda_max,
            )
        return self._user_models[gesture]

    def get_user_descriptor(self, gesture: str) -> GestureDescriptor:
        return self.get_user_model(gesture).effective_descriptor()

    def classify(self, feature_vector: np.ndarray) -> tuple[str, dict[str, float]]:
        feature_vector = np.asarray(feature_vector, dtype=np.float64)
        return classify(feature_vector, self.descriptors)

    def score(
        self,
        feature_vector: np.ndarray,
        target_gesture: str,
        update_user_model: bool = True,
        top_k_feedback: int = 3,
        tau_override: float | None = None,
    ) -> ScoreResult:
        feature_vector = np.asarray(feature_vector, dtype=np.float64)
        if target_gesture not in self.descriptors:
            raise KeyError(f"Unknown gesture: {target_gesture}")

        user_model = self.get_user_model(target_gesture)
        effective_descriptor = user_model.effective_descriptor()
        tau = max(float(tau_override) if tau_override is not None else user_model.tau, 1e-6)
        distance = effective_descriptor.mahalanobis(feature_vector)
        raw_score = float(np.exp(-distance / tau))
        display_score = int(np.clip(np.round((raw_score ** self.gamma) * 100.0), 0, 100))

        predicted_gesture, population_distances = self.classify(feature_vector)
        deviations = summarize_deviations(feature_vector, effective_descriptor, top_k=top_k_feedback)
        feedback = generate_feedback(
            feature_vector,
            effective_descriptor,
            target_gesture=target_gesture,
            predicted_gesture=predicted_gesture,
            top_k=top_k_feedback,
        )

        result = ScoreResult(
            target_gesture=target_gesture,
            predicted_gesture=predicted_gesture,
            mahalanobis_distance=float(distance),
            raw_score=raw_score,
            display_score=display_score,
            matched=predicted_gesture == target_gesture,
            feedback=feedback_payload(feedback),
            tau=float(tau),
            blend_weight=float(user_model.blend_weight),
            feature_deviations=deviations,
            population_distances={key: float(value) for key, value in population_distances.items()},
        )

        if update_user_model:
            user_model.record_attempt(feature_vector)
        return result
