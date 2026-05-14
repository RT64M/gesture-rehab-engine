from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.challenge_zone import get_challenge_zone
from src.scorer import ColdStartScorer, compute_expected_distances, load_descriptors
from src.threshold_manager import MomentumThresholdManager


@dataclass
class StrategyAttemptResult:
    strategy: str
    target_gesture: str
    predicted_gesture: str
    raw_score: float
    display_score: int
    matched: bool
    tau: float
    challenge_zone: str
    feedback: list[str] = field(default_factory=list)
    feedback_features: list[str] = field(default_factory=list)
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseStrategy:
    name = "base"

    def score_attempt(self, feature_vector: np.ndarray, target_gesture: str) -> StrategyAttemptResult:
        raise NotImplementedError


class FixedStandardizedStrategy(BaseStrategy):
    name = "fixed_standardized"

    def __init__(self, scorer: ColdStartScorer, tau0: float = 5.0) -> None:
        self.scorer = scorer
        self.tau0 = float(tau0)

    def score_attempt(self, feature_vector: np.ndarray, target_gesture: str) -> StrategyAttemptResult:
        result = self.scorer.score(
            feature_vector,
            target_gesture,
            update_user_model=False,
            tau_override=self.tau0,
        )
        return StrategyAttemptResult(
            strategy=self.name,
            target_gesture=target_gesture,
            predicted_gesture=result.predicted_gesture,
            raw_score=float(result.raw_score),
            display_score=int(result.display_score),
            matched=bool(result.matched),
            tau=self.tau0,
            challenge_zone=get_challenge_zone(result.display_score),
            feedback=list(result.feedback),
            feedback_features=[str(item["feature"]) for item in result.feature_deviations],
            metadata={"blend_weight": float(result.blend_weight)},
        )


@dataclass(frozen=True)
class HeuristicRule:
    feature: str
    comparator: str
    threshold: float
    tolerance: float
    weight: float = 1.0


HEURISTIC_RULES: dict[str, list[HeuristicRule]] = {
    "fist": [
        HeuristicRule("index_pip", "le", 2.20, 0.45),
        HeuristicRule("middle_pip", "le", 2.20, 0.45),
        HeuristicRule("ring_pip", "le", 2.20, 0.45),
        HeuristicRule("pinky_pip", "le", 2.20, 0.45),
        HeuristicRule("dist_thumb_index_tip", "ge", 0.32, 0.18, weight=0.7),
    ],
    "palm": [
        HeuristicRule("index_pip", "ge", 2.55, 0.40),
        HeuristicRule("middle_pip", "ge", 2.55, 0.40),
        HeuristicRule("ring_pip", "ge", 2.55, 0.40),
        HeuristicRule("pinky_pip", "ge", 2.55, 0.40),
        HeuristicRule("spread_index_middle", "ge", 0.18, 0.18, weight=0.7),
    ],
    "ok": [
        HeuristicRule("dist_thumb_index_tip", "le", 0.26, 0.18, weight=1.4),
        HeuristicRule("middle_pip", "ge", 2.45, 0.40),
        HeuristicRule("ring_pip", "ge", 2.45, 0.40),
        HeuristicRule("pinky_pip", "ge", 2.45, 0.40),
    ],
    "peace": [
        HeuristicRule("index_pip", "ge", 2.50, 0.40),
        HeuristicRule("middle_pip", "ge", 2.50, 0.40),
        HeuristicRule("ring_pip", "le", 2.20, 0.45),
        HeuristicRule("pinky_pip", "le", 2.20, 0.45),
        HeuristicRule("spread_index_middle", "ge", 0.28, 0.20, weight=0.8),
    ],
    "rock": [
        HeuristicRule("index_pip", "ge", 2.50, 0.40),
        HeuristicRule("middle_pip", "le", 2.20, 0.45),
        HeuristicRule("ring_pip", "le", 2.20, 0.45),
        HeuristicRule("pinky_pip", "ge", 2.40, 0.45),
        HeuristicRule("spread_ring_pinky", "ge", 0.18, 0.18, weight=0.7),
    ],
}


class HeuristicRuleStrategy(BaseStrategy):
    name = "heuristic_rule"

    def __init__(self, feature_names: list[str], descriptors: dict | None = None) -> None:
        self.feature_names = feature_names
        self.feature_index = {name: index for index, name in enumerate(feature_names)}
        self.descriptors = descriptors or {}

    def _rule_score(self, value: float, rule: HeuristicRule) -> float:
        if rule.comparator == "ge":
            penalty = max(0.0, rule.threshold - value)
        elif rule.comparator == "le":
            penalty = max(0.0, value - rule.threshold)
        else:
            raise KeyError(f"Unknown comparator: {rule.comparator}")
        return float(np.clip(1.0 - penalty / max(rule.tolerance, 1e-6), 0.0, 1.0))

    def _gesture_score(self, feature_vector: np.ndarray, gesture: str) -> tuple[float, list[tuple[HeuristicRule, float]]]:
        if gesture not in HEURISTIC_RULES:
            return self._descriptor_score(feature_vector, gesture)

        rows: list[tuple[HeuristicRule, float]] = []
        weighted = 0.0
        total_weight = 0.0
        for rule in HEURISTIC_RULES[gesture]:
            value = float(feature_vector[self.feature_index[rule.feature]])
            score = self._rule_score(value, rule)
            rows.append((rule, score))
            weighted += score * rule.weight
            total_weight += rule.weight
        return weighted / max(total_weight, 1e-6), rows

    def _descriptor_score(self, feature_vector: np.ndarray, gesture: str) -> tuple[float, list[tuple[HeuristicRule, float]]]:
        descriptor = self.descriptors.get(gesture)
        if descriptor is None:
            raise KeyError(f"No heuristic rules or descriptor available for gesture: {gesture}")

        sigma_diag = np.sqrt(np.clip(np.diag(descriptor.sigma), 1e-8, None))
        z = np.abs((feature_vector - descriptor.mu) / np.maximum(sigma_diag, 1e-6))
        raw_score = float(np.exp(-np.mean(np.clip(z, 0.0, 6.0)) / 2.5))
        weakest_indices = np.argsort(z)[-3:][::-1]
        rows = [
            (
                HeuristicRule(
                    feature=descriptor.feature_names[index],
                    comparator="near",
                    threshold=float(descriptor.mu[index]),
                    tolerance=float(sigma_diag[index]),
                ),
                float(np.clip(np.exp(-z[index] / 2.5), 0.0, 1.0)),
            )
            for index in weakest_indices
        ]
        return raw_score, rows

    def score_attempt(self, feature_vector: np.ndarray, target_gesture: str) -> StrategyAttemptResult:
        feature_vector = np.asarray(feature_vector, dtype=np.float64)
        gesture_scores = {
            gesture: self._gesture_score(feature_vector, gesture)
            for gesture in (self.descriptors.keys() or HEURISTIC_RULES.keys())
        }
        predicted_gesture = max(gesture_scores, key=lambda key: gesture_scores[key][0])
        raw_score, rule_rows = gesture_scores[target_gesture]
        display_score = int(np.clip(np.round(raw_score * 100.0), 0, 100))

        weakest = sorted(rule_rows, key=lambda item: item[1])[:3]
        feedback_features = [item[0].feature for item in weakest]
        feedback = [f"{item[0].feature} rule miss={1.0 - item[1]:.2f}" for item in weakest]

        return StrategyAttemptResult(
            strategy=self.name,
            target_gesture=target_gesture,
            predicted_gesture=predicted_gesture,
            raw_score=float(raw_score),
            display_score=display_score,
            matched=predicted_gesture == target_gesture,
            tau=0.0,
            challenge_zone=get_challenge_zone(display_score),
            feedback=feedback,
            feedback_features=feedback_features,
            metadata={},
        )


class AdaptiveScoringStrategy(BaseStrategy):
    name = "adaptive"

    def __init__(
        self,
        scorer: ColdStartScorer,
        tau_init: float = 5.0,
        round_size: int = 10,
    ) -> None:
        self.scorer = scorer
        self.threshold_manager = MomentumThresholdManager(tau_init=tau_init, round_size=round_size)

    def score_attempt(self, feature_vector: np.ndarray, target_gesture: str) -> StrategyAttemptResult:
        tau = self.threshold_manager.get_current_tau(target_gesture)
        score_result = self.scorer.score(
            feature_vector,
            target_gesture,
            update_user_model=True,
            tau_override=tau,
        )
        self.threshold_manager.register_attempt(
            target_gesture,
            score_result.display_score,
            score_result.raw_score,
        )

        round_index_before = self.threshold_manager.get_state(target_gesture).round_index
        update_reason = ""
        if self.threshold_manager.has_full_round(target_gesture):
            update = self.threshold_manager.on_round_end(target_gesture)
            update_reason = update.reason
        state = self.threshold_manager.get_state(target_gesture)

        return StrategyAttemptResult(
            strategy=self.name,
            target_gesture=target_gesture,
            predicted_gesture=score_result.predicted_gesture,
            raw_score=float(score_result.raw_score),
            display_score=int(score_result.display_score),
            matched=bool(score_result.matched),
            tau=float(score_result.tau),
            challenge_zone=get_challenge_zone(score_result.display_score),
            feedback=list(score_result.feedback),
            feedback_features=[str(item["feature"]) for item in score_result.feature_deviations],
            metadata={
                "blend_weight": float(score_result.blend_weight),
                "round_index_before": int(round_index_before),
                "round_index_after": int(state.round_index),
                "update_reason": update_reason,
            },
        )


def load_strategy_artifacts(
    descriptor_dir: Path,
    features_dir: Path | None = None,
) -> tuple[dict, dict[str, float], list[str]]:
    descriptors = load_descriptors(descriptor_dir)
    expected_distances = compute_expected_distances(descriptors, features_dir)
    feature_names = list(next(iter(descriptors.values())).feature_names)
    return descriptors, expected_distances, feature_names


def build_strategy(
    strategy_name: str,
    descriptor_dir: Path,
    features_dir: Path | None = None,
) -> BaseStrategy:
    descriptors, expected_distances, feature_names = load_strategy_artifacts(descriptor_dir, features_dir)
    if strategy_name == FixedStandardizedStrategy.name:
        return FixedStandardizedStrategy(ColdStartScorer(descriptors, expected_distances))
    if strategy_name == HeuristicRuleStrategy.name:
        return HeuristicRuleStrategy(feature_names, descriptors)
    if strategy_name == AdaptiveScoringStrategy.name:
        return AdaptiveScoringStrategy(ColdStartScorer(descriptors, expected_distances))
    raise KeyError(f"Unknown strategy: {strategy_name}")
