from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from itertools import product

import numpy as np

from .runner import ExperimentResult


METRIC_DEFINITIONS = {
    "ability_gain": "Final post-session latent ability minus initial pre-session latent ability, averaged per virtual subject.",
    "challenge_ratio": "Fraction of attempts whose displayed score falls inside the effective challenge zone [60, 85].",
    "feedback_accuracy": "Fraction of feedback-bearing attempts where the top feedback feature overlaps the virtual subject's simulated dominant deficit.",
    "frustration_risk": "Fraction of virtual-subject/gesture streams containing at least three consecutive attempts below 40.",
    "score_ability_correlation": "Pearson correlation between displayed score and the virtual subject's latent realized ability.",
    "scenario_variance": "Variance of challenge-zone ratios across recovery-curve scenarios; lower values indicate more even behavior.",
}


@dataclass
class StrategyMetrics:
    ability_gain: float
    challenge_ratio: float
    feedback_accuracy: float
    frustration_risk: float
    score_ability_correlation: float
    scenario_variance: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StatisticalReport:
    strategy_metrics: dict[str, StrategyMetrics]
    pairwise_effect_sizes: dict[str, float]
    confidence_intervals: dict[str, dict[str, dict[str, float | int | str]]]
    pairwise_permutation_tests: dict[str, dict[str, float | int | str]]
    scenario_metrics: dict[str, dict[str, dict[str, float]]]
    metric_definitions: dict[str, str]
    narrative: dict[str, str | list[str]]

    def to_dict(self) -> dict:
        return {
            "strategy_metrics": {key: value.to_dict() for key, value in self.strategy_metrics.items()},
            "pairwise_effect_sizes": self.pairwise_effect_sizes,
            "confidence_intervals": self.confidence_intervals,
            "pairwise_permutation_tests": self.pairwise_permutation_tests,
            "scenario_metrics": self.scenario_metrics,
            "metric_definitions": self.metric_definitions,
            "narrative": self.narrative,
        }


def _cohen_d(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    pooled = np.sqrt(((left.var(ddof=1) + right.var(ddof=1)) / 2.0))
    if pooled < 1e-8:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def _bootstrap_stat_interval(
    values: np.ndarray,
    statistic,
    rng: np.random.Generator,
    iterations: int = 2000,
    level: float = 0.95,
) -> dict[str, float | int | str]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"low": 0.0, "high": 0.0, "level": level, "method": "bootstrap", "n": 0}
    if len(values) == 1:
        value = float(statistic(values))
        return {"low": value, "high": value, "level": level, "method": "bootstrap", "n": 1}

    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        estimates[index] = float(statistic(sample))
    alpha = (1.0 - level) / 2.0
    return {
        "low": float(np.quantile(estimates, alpha)),
        "high": float(np.quantile(estimates, 1.0 - alpha)),
        "level": level,
        "method": "bootstrap",
        "n": int(len(values)),
    }


def _bootstrap_correlation_interval(
    scores: np.ndarray,
    abilities: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 2000,
    level: float = 0.95,
) -> dict[str, float | int | str]:
    scores = np.asarray(scores, dtype=np.float64)
    abilities = np.asarray(abilities, dtype=np.float64)
    if len(scores) < 3 or np.std(scores) <= 1e-8 or np.std(abilities) <= 1e-8:
        return {"low": 0.0, "high": 0.0, "level": level, "method": "bootstrap", "n": int(len(scores))}

    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample_indices = rng.integers(0, len(scores), size=len(scores))
        sample_scores = scores[sample_indices]
        sample_abilities = abilities[sample_indices]
        if np.std(sample_scores) <= 1e-8 or np.std(sample_abilities) <= 1e-8:
            estimates[index] = 0.0
        else:
            estimates[index] = float(np.corrcoef(sample_scores, sample_abilities)[0, 1])
    alpha = (1.0 - level) / 2.0
    return {
        "low": float(np.quantile(estimates, alpha)),
        "high": float(np.quantile(estimates, 1.0 - alpha)),
        "level": level,
        "method": "bootstrap",
        "n": int(len(scores)),
    }


def _patient_ability_gains(sessions: list) -> dict[str, float]:
    rows_by_patient: dict[str, list] = {}
    for row in sessions:
        rows_by_patient.setdefault(row.patient_id, []).append(row)

    gains: dict[str, float] = {}
    for patient_id, rows in rows_by_patient.items():
        rows = sorted(rows, key=lambda item: item.session_index)
        gains[patient_id] = float(rows[-1].post_ability - rows[0].pre_ability)
    return gains


def _frustration_hits(attempts: list) -> np.ndarray:
    stream_scores: dict[tuple[str, str], list[int]] = {}
    for row in attempts:
        stream_scores.setdefault((row.patient_id, row.gesture), []).append(row.display_score)

    hits = []
    for values in stream_scores.values():
        hits.append(
            any(
                values[index] < 40 and values[index + 1] < 40 and values[index + 2] < 40
                for index in range(len(values) - 2)
            )
        )
    return np.asarray(hits, dtype=np.float64)


def _feedback_hits(attempts: list) -> np.ndarray:
    aligned = [
        bool(set(row.feedback_features[:1]) & set(row.true_dominant_features))
        for row in attempts
        if row.feedback_features
    ]
    return np.asarray(aligned, dtype=np.float64)


def _paired_permutation_test(
    left: dict[str, float],
    right: dict[str, float],
    rng: np.random.Generator,
    max_exact_pairs: int = 14,
    iterations: int = 5000,
) -> dict[str, float | int | str]:
    common = sorted(set(left) & set(right))
    if not common:
        return {
            "metric": "ability_gain",
            "observed_difference": 0.0,
            "p_value": 1.0,
            "n_pairs": 0,
            "method": "paired permutation",
        }

    diffs = np.asarray([left[key] - right[key] for key in common], dtype=np.float64)
    observed = float(np.mean(diffs))
    if len(diffs) <= max_exact_pairs:
        estimates = []
        for signs in product((-1.0, 1.0), repeat=len(diffs)):
            estimates.append(float(np.mean(diffs * np.asarray(signs, dtype=np.float64))))
        estimates = np.asarray(estimates, dtype=np.float64)
        method = "exact paired sign-flip permutation"
    else:
        signs = rng.choice([-1.0, 1.0], size=(iterations, len(diffs)), replace=True)
        estimates = np.mean(signs * diffs, axis=1)
        method = "sampled paired sign-flip permutation"
    p_value = float((np.sum(np.abs(estimates) >= abs(observed)) + 1.0) / (len(estimates) + 1.0))
    return {
        "metric": "ability_gain",
        "observed_difference": observed,
        "p_value": p_value,
        "n_pairs": int(len(diffs)),
        "method": method,
    }


def _build_narrative(strategy_metrics: dict[str, StrategyMetrics]) -> dict[str, str | list[str]]:
    best_gain = max(strategy_metrics, key=lambda key: strategy_metrics[key].ability_gain)
    best_challenge = max(strategy_metrics, key=lambda key: strategy_metrics[key].challenge_ratio)
    best_feedback = max(strategy_metrics, key=lambda key: strategy_metrics[key].feedback_accuracy)
    lowest_frustration = min(strategy_metrics, key=lambda key: strategy_metrics[key].frustration_risk)
    return {
        "phase4_relation": (
            "Phase 4 reverse-infers simulated progress from scoring-formula variables such as tau, baseline, momentum, "
            "challenge-zone state, and descriptor history; Phase 5 reuses the same simulated rehabilitation setting "
            "to compare scoring policies as experimental interventions."
        ),
        "key_findings": [
            f"{best_gain} has the highest mean latent ability gain.",
            f"{best_challenge} keeps the largest share of attempts inside the effective challenge zone.",
            f"{best_feedback} produces the most aligned top-feature feedback in the simulation.",
            f"{lowest_frustration} has the lowest estimated frustration risk.",
        ],
        "interpretation_boundary": (
            "These are simulation-backed paper-prototype results, not clinical efficacy claims. "
            "No real patient logs are used or planned for this project; future work stays within simulated histories "
            "or public non-patient gesture benchmarks."
        ),
    }


class ExperimentAnalyzer:
    def analyze(self, result: ExperimentResult) -> StatisticalReport:
        strategy_metrics: dict[str, StrategyMetrics] = {}
        confidence_intervals: dict[str, dict[str, dict[str, float | int | str]]] = {}
        scenario_metrics: dict[str, dict[str, dict[str, float]]] = {}
        attempts_by_strategy: dict[str, list] = {}
        sessions_by_strategy: dict[str, list] = {}
        ability_gains_by_strategy: dict[str, dict[str, float]] = {}
        rng = np.random.default_rng(20260510)

        for attempt in result.attempts:
            attempts_by_strategy.setdefault(attempt.strategy, []).append(attempt)
        for session in result.sessions:
            sessions_by_strategy.setdefault(session.strategy, []).append(session)

        for strategy, attempts in attempts_by_strategy.items():
            sessions = sessions_by_strategy.get(strategy, [])
            scores = np.asarray([row.display_score for row in attempts], dtype=np.float64)
            abilities = np.asarray([row.realized_ability for row in attempts], dtype=np.float64)
            challenge_hits = ((scores >= 60.0) & (scores <= 85.0)).astype(np.float64)
            challenge_ratio = float(np.mean(challenge_hits)) if len(challenge_hits) else 0.0

            feedback_hits = _feedback_hits(attempts)
            feedback_accuracy = float(np.mean(feedback_hits)) if len(feedback_hits) else 0.0

            correlation = 0.0
            if len(scores) >= 2 and np.std(scores) > 1e-8 and np.std(abilities) > 1e-8:
                correlation = float(np.corrcoef(scores, abilities)[0, 1])

            frustration_hits = _frustration_hits(attempts)
            frustration_risk = float(np.mean(frustration_hits)) if len(frustration_hits) else 0.0

            gains = _patient_ability_gains(sessions)
            ability_gains_by_strategy[strategy] = gains
            gain_values = np.asarray(list(gains.values()), dtype=np.float64)
            ability_gain = float(np.mean(gain_values)) if len(gain_values) else 0.0

            curve_ratios: list[float] = []
            scenario_metrics[strategy] = {}
            for curve in sorted({row.recovery_curve for row in attempts}):
                curve_attempts = [row for row in attempts if row.recovery_curve == curve]
                curve_scores = np.asarray([row.display_score for row in curve_attempts], dtype=np.float64)
                if len(curve_scores):
                    curve_challenge = float(np.mean((curve_scores >= 60.0) & (curve_scores <= 85.0)))
                    curve_ratios.append(curve_challenge)
                    scenario_metrics[strategy][curve] = {
                        "mean_score": float(np.mean(curve_scores)),
                        "challenge_ratio": curve_challenge,
                        "feedback_accuracy": float(np.mean(_feedback_hits(curve_attempts))) if curve_attempts else 0.0,
                    }
            scenario_variance = float(np.var(curve_ratios)) if curve_ratios else 0.0

            strategy_metrics[strategy] = StrategyMetrics(
                ability_gain=ability_gain,
                challenge_ratio=challenge_ratio,
                feedback_accuracy=feedback_accuracy,
                frustration_risk=frustration_risk,
                score_ability_correlation=correlation,
                scenario_variance=scenario_variance,
            )
            confidence_intervals[strategy] = {
                "ability_gain": _bootstrap_stat_interval(gain_values, np.mean, rng),
                "challenge_ratio": _bootstrap_stat_interval(challenge_hits, np.mean, rng),
                "feedback_accuracy": _bootstrap_stat_interval(feedback_hits, np.mean, rng),
                "frustration_risk": _bootstrap_stat_interval(frustration_hits, np.mean, rng),
                "score_ability_correlation": _bootstrap_correlation_interval(scores, abilities, rng),
                "scenario_variance": _bootstrap_stat_interval(np.asarray(curve_ratios, dtype=np.float64), np.var, rng),
            }

        pairwise_effect_sizes: dict[str, float] = {}
        pairwise_permutation_tests: dict[str, dict[str, float | int | str]] = {}
        for left, right in combinations(sorted(sessions_by_strategy), 2):
            left_gains = ability_gains_by_strategy[left]
            right_gains = ability_gains_by_strategy[right]
            pairwise_effect_sizes[f"{left}__vs__{right}"] = _cohen_d(
                np.asarray(list(left_gains.values()), dtype=np.float64),
                np.asarray(list(right_gains.values()), dtype=np.float64),
            )
            pairwise_permutation_tests[f"{left}__vs__{right}"] = _paired_permutation_test(left_gains, right_gains, rng)

        return StatisticalReport(
            strategy_metrics=strategy_metrics,
            pairwise_effect_sizes=pairwise_effect_sizes,
            confidence_intervals=confidence_intervals,
            pairwise_permutation_tests=pairwise_permutation_tests,
            scenario_metrics=scenario_metrics,
            metric_definitions=dict(METRIC_DEFINITIONS),
            narrative=_build_narrative(strategy_metrics),
        )
