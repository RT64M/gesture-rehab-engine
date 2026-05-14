from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from src import config
from src.descriptors import GestureDescriptor
from src.progress_analyzer import ProgressAnalyzer, ProgressReport, RoundRecord, SessionData


def _feature_names() -> list[str]:
    distance_names = [f"dist_{name}" for name in config.TIP_PAIRS]
    spread_names = list(config.SPREAD_TRIPLETS.keys())
    return list(config.JOINT_NAMES) + distance_names + spread_names


def _healthy_descriptors() -> dict[str, GestureDescriptor]:
    feature_names = _feature_names()
    mu = np.zeros(len(feature_names), dtype=np.float64)
    sigma = np.eye(len(feature_names), dtype=np.float64)
    descriptor = GestureDescriptor(
        gesture="fist",
        n_samples=100,
        mu=mu,
        sigma=sigma,
        sigma_inv=np.linalg.inv(sigma),
        feature_names=feature_names,
    )
    return {"fist": descriptor}


def _record(
    day_index: int,
    tau: float,
    user_mean: np.ndarray,
    healthy_mean: np.ndarray,
    mean_score: float = 75.0,
    gesture: str = "fist",
    baseline: float = 70.0,
    momentum: float = 0.0,
    challenge_zone: str = "in_zone",
) -> RoundRecord:
    timestamp = (date(2026, 1, 1) + timedelta(days=day_index)).isoformat()
    return RoundRecord(
        timestamp=timestamp,
        gesture=gesture,
        round_index=day_index + 1,
        tau=tau,
        baseline=baseline,
        momentum=momentum,
        mean_score=mean_score,
        challenge_zone=challenge_zone,
        user_mean=user_mean.tolist(),
        healthy_mean=healthy_mean.tolist(),
    )


def _analyzer() -> ProgressAnalyzer:
    return ProgressAnalyzer(
        healthy_descriptors=_healthy_descriptors(),
        initial_taus={"fist": 5.0},
    )


def test_progression_index_tracks_sustained_improvement():
    healthy = np.zeros(len(_feature_names()), dtype=np.float64)
    records = []
    for day_index, tau in enumerate(np.linspace(5.0, 2.0, 14)):
        user_mean = np.linspace(1.0, 0.1, len(_feature_names()))
        records.append(_record(day_index, tau=tau, user_mean=user_mean, healthy_mean=healthy))

    report = _analyzer().update(SessionData(records=records))
    progression_values = [row["progression_index"] for row in report.history_summary["daily_progression"]]
    correlation = np.corrcoef(np.arange(len(progression_values)), progression_values)[0, 1]
    assert correlation > 0.95


def test_plateau_warning_triggers_after_stagnation_window():
    healthy = np.zeros(len(_feature_names()), dtype=np.float64)
    records = [
        _record(day_index, tau=4.0, user_mean=np.ones(len(_feature_names())), healthy_mean=healthy)
        for day_index in range(14)
    ]

    report = _analyzer().update(SessionData(records=records))
    assert any("plateaued" in warning for warning in report.warnings)


def test_regression_warning_triggers_after_consecutive_decline():
    healthy = np.zeros(len(_feature_names()), dtype=np.float64)
    taus = [2.0, 2.4, 2.9, 3.5, 4.2, 4.8]
    records = [
        _record(day_index, tau=tau, user_mean=np.ones(len(_feature_names())), healthy_mean=healthy)
        for day_index, tau in enumerate(taus)
    ]

    report = _analyzer().update(SessionData(records=records))
    assert any("Regression" in warning for warning in report.warnings)


def test_joint_recovery_identifies_lagging_joint():
    healthy = np.zeros(len(_feature_names()), dtype=np.float64)
    initial = np.ones(len(_feature_names()), dtype=np.float64)
    latest = np.zeros(len(_feature_names()), dtype=np.float64)
    latest[config.JOINT_NAMES.index("index_pip")] = 0.85

    report = _analyzer().update(
        SessionData(
            records=[
                _record(0, tau=5.0, user_mean=initial, healthy_mean=healthy),
                _record(1, tau=4.0, user_mean=latest, healthy_mean=healthy),
            ]
        )
    )

    joint_details = report.joint_details["fist"]
    assert joint_details["index_pip"] < 0.2
    assert joint_details["thumb_cmc"] > 0.9


def test_progression_index_uses_descriptor_recovery_not_only_tau():
    healthy = np.zeros(len(_feature_names()), dtype=np.float64)
    initial = np.ones(len(_feature_names()), dtype=np.float64)
    latest = np.full(len(_feature_names()), 0.2, dtype=np.float64)

    report = _analyzer().update(
        SessionData(
            records=[
                _record(
                    0,
                    tau=5.0,
                    user_mean=initial,
                    healthy_mean=healthy,
                    mean_score=50.0,
                    challenge_zone="too_hard",
                ),
                _record(
                    1,
                    tau=5.0,
                    user_mean=latest,
                    healthy_mean=healthy,
                    mean_score=50.0,
                    challenge_zone="too_hard",
                ),
            ]
        )
    )

    rows = report.history_summary["daily_progression"]
    assert rows[-1]["strictness_progression"] == 0.0
    assert rows[-1]["descriptor_recovery"] > 0.7
    assert rows[-1]["progression_index"] > rows[0]["progression_index"]
    assert "progression_component_weights" in report.history_summary


def test_recovery_score_and_joint_ratios_are_clipped():
    healthy = np.zeros(len(_feature_names()), dtype=np.float64)
    initial = np.ones(len(_feature_names()), dtype=np.float64)
    latest = np.full(len(_feature_names()), 3.0, dtype=np.float64)

    report = _analyzer().update(
        SessionData(
            records=[
                _record(0, tau=5.0, user_mean=initial, healthy_mean=healthy),
                _record(1, tau=4.5, user_mean=latest, healthy_mean=healthy),
            ]
        )
    )

    assert 0.0 <= report.recovery_score <= 100.0
    for value in report.joint_details["fist"].values():
        assert 0.0 <= value <= 1.0


def test_report_export_returns_cached_report():
    healthy = np.zeros(len(_feature_names()), dtype=np.float64)
    report = _analyzer().update(
        SessionData(
            records=[
                _record(0, tau=5.0, user_mean=np.ones(len(_feature_names())), healthy_mean=healthy),
                _record(1, tau=4.0, user_mean=np.zeros(len(_feature_names())), healthy_mean=healthy),
            ]
        )
    )

    exported = _analyzer()
    exported.update(
        SessionData(
            records=[
                _record(0, tau=5.0, user_mean=np.ones(len(_feature_names())), healthy_mean=healthy),
                _record(1, tau=4.0, user_mean=np.zeros(len(_feature_names())), healthy_mean=healthy),
            ]
        )
    )
    assert isinstance(exported.export_report(), ProgressReport)
    assert report.progression_index == exported.export_report().progression_index
