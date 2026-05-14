from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.scorer import ColdStartScorer

from .strategies import (
    AdaptiveScoringStrategy,
    FixedStandardizedStrategy,
    HeuristicRuleStrategy,
    load_strategy_artifacts,
)
from .virtual_patient import CURVE_NAMES, VirtualPatient, _stable_seed


@dataclass
class ExperimentAttemptLog:
    patient_id: str
    recovery_curve: str
    strategy: str
    gesture: str
    session_index: int
    attempt_index: int
    base_ability: float
    realized_ability: float
    display_score: int
    raw_score: float
    predicted_gesture: str
    matched: bool
    tau: float
    challenge_zone: str
    feedback_features: list[str]
    true_dominant_features: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentSessionLog:
    patient_id: str
    recovery_curve: str
    strategy: str
    session_index: int
    mean_score: float
    quality: float
    pre_ability: float
    post_ability: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    attempts: list[ExperimentAttemptLog]
    sessions: list[ExperimentSessionLog]
    config: dict

    def to_dict(self) -> dict:
        return {
            "attempts": [row.to_dict() for row in self.attempts],
            "sessions": [row.to_dict() for row in self.sessions],
            "config": self.config,
        }


def session_quality(mean_score: float) -> float:
    if mean_score < 40.0:
        return -0.75
    if mean_score < 60.0:
        return -0.30
    if mean_score <= 85.0:
        return 1.00
    return 0.20


class ExperimentRunner:
    """Run simulated patients through multiple scoring strategies."""

    def __init__(
        self,
        descriptor_dir: Path,
        features_dir: Path | None = None,
        patients_per_curve: int = 4,
        n_sessions: int = 12,
        attempts_per_session: int = 10,
        strategy_names: list[str] | None = None,
        recovery_curves: list[str] | None = None,
        gestures: list[str] | None = None,
        seed: int = 20260422,
    ) -> None:
        self.descriptor_dir = Path(descriptor_dir)
        self.features_dir = None if features_dir is None else Path(features_dir)
        self.patients_per_curve = int(patients_per_curve)
        self.n_sessions = int(n_sessions)
        self.attempts_per_session = int(attempts_per_session)
        self.strategy_names = strategy_names or [
            FixedStandardizedStrategy.name,
            HeuristicRuleStrategy.name,
            AdaptiveScoringStrategy.name,
        ]
        self.recovery_curves = recovery_curves or list(CURVE_NAMES)
        self.seed = int(seed)
        self.descriptors, self.expected_distances, self.feature_names = load_strategy_artifacts(
            self.descriptor_dir,
            self.features_dir,
        )
        if gestures is not None:
            unknown = sorted(set(gestures) - set(self.descriptors))
            if unknown:
                raise KeyError(f"Unsupported gestures: {unknown}")
            self.descriptors = {gesture: self.descriptors[gesture] for gesture in gestures}
            self.expected_distances = {gesture: self.expected_distances[gesture] for gesture in gestures}
        self.gestures = list(self.descriptors)

    def _make_patient_specs(self) -> list[dict]:
        specs: list[dict] = []
        for curve in self.recovery_curves:
            for index in range(self.patients_per_curve):
                patient_id = f"{curve}-p{index + 1:02d}"
                specs.append(
                    {
                        "patient_id": patient_id,
                        "recovery_curve": curve,
                        "noise_level": 0.06 + 0.01 * (index % 4),
                        "initial_impairment": 1.20 + 0.08 * (index % 5),
                        "responsiveness": 0.035 + 0.01 * (index % 3),
                        "seed": _stable_seed(self.seed, patient_id, curve),
                    }
                )
        return specs

    def run(self) -> ExperimentResult:
        attempts: list[ExperimentAttemptLog] = []
        sessions: list[ExperimentSessionLog] = []
        patient_specs = self._make_patient_specs()

        for strategy_name in self.strategy_names:
            for spec in patient_specs:
                patient = VirtualPatient(
                    patient_id=spec["patient_id"],
                    healthy_descriptors=self.descriptors,
                    recovery_curve=spec["recovery_curve"],
                    noise_level=spec["noise_level"],
                    initial_impairment=spec["initial_impairment"],
                    responsiveness=spec["responsiveness"],
                    seed=spec["seed"],
                )

                if strategy_name == FixedStandardizedStrategy.name:
                    strategy = FixedStandardizedStrategy(
                        ColdStartScorer(self.descriptors, self.expected_distances)
                    )
                elif strategy_name == HeuristicRuleStrategy.name:
                    strategy = HeuristicRuleStrategy(self.feature_names, self.descriptors)
                elif strategy_name == AdaptiveScoringStrategy.name:
                    strategy = AdaptiveScoringStrategy(
                        scorer=ColdStartScorer(self.descriptors, self.expected_distances)
                    )
                else:
                    raise KeyError(f"Unknown strategy: {strategy_name}")

                for session_index in range(self.n_sessions):
                    pre_ability = patient.ability_at(session_index, self.n_sessions)
                    session_scores: list[float] = []

                    for gesture in self.gestures:
                        for attempt_index in range(self.attempts_per_session):
                            patient_attempt = patient.sample_attempt(
                                gesture=gesture,
                                session_index=session_index,
                                total_sessions=self.n_sessions,
                                attempt_index=attempt_index,
                            )
                            strategy_result = strategy.score_attempt(patient_attempt.vector, gesture)
                            session_scores.append(strategy_result.display_score)
                            attempts.append(
                                ExperimentAttemptLog(
                                    patient_id=patient_attempt.patient_id,
                                    recovery_curve=patient_attempt.recovery_curve,
                                    strategy=strategy_result.strategy,
                                    gesture=gesture,
                                    session_index=session_index,
                                    attempt_index=attempt_index,
                                    base_ability=patient_attempt.base_ability,
                                    realized_ability=patient_attempt.realized_ability,
                                    display_score=strategy_result.display_score,
                                    raw_score=strategy_result.raw_score,
                                    predicted_gesture=strategy_result.predicted_gesture,
                                    matched=strategy_result.matched,
                                    tau=strategy_result.tau,
                                    challenge_zone=strategy_result.challenge_zone,
                                    feedback_features=list(strategy_result.feedback_features),
                                    true_dominant_features=list(patient_attempt.dominant_features),
                                )
                            )

                    mean_score = float(np.mean(session_scores)) if session_scores else 0.0
                    quality = session_quality(mean_score)
                    patient.apply_training_feedback(quality)
                    post_ability = patient.ability_at(min(session_index + 1, self.n_sessions - 1), self.n_sessions)
                    sessions.append(
                        ExperimentSessionLog(
                            patient_id=patient.patient_id,
                            recovery_curve=patient.recovery_curve,
                            strategy=strategy_name,
                            session_index=session_index,
                            mean_score=mean_score,
                            quality=quality,
                            pre_ability=pre_ability,
                            post_ability=post_ability,
                        )
                    )

        return ExperimentResult(
            attempts=attempts,
            sessions=sessions,
            config={
                "patients_per_curve": self.patients_per_curve,
                "n_sessions": self.n_sessions,
                "attempts_per_session": self.attempts_per_session,
                "strategy_names": list(self.strategy_names),
                "recovery_curves": list(self.recovery_curves),
                "gestures": list(self.gestures),
            },
        )
