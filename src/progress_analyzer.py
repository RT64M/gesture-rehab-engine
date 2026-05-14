from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from . import config
from .descriptors import GestureDescriptor


PROGRESSION_COMPONENT_WEIGHTS = {
    "strictness_progression": 0.40,
    "descriptor_recovery": 0.30,
    "baseline_margin": 0.15,
    "momentum_trend": 0.10,
    "challenge_zone_alignment": 0.05,
}


def _sort_timestamp(value: str) -> tuple[int, str]:
    try:
        return (0, datetime.fromisoformat(value).isoformat())
    except ValueError:
        return (1, value)


def _safe_float_list(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(item) for item in array.tolist()]


@dataclass
class RoundRecord:
    timestamp: str
    gesture: str
    round_index: int
    tau: float
    baseline: float
    momentum: float
    mean_score: float
    challenge_zone: str
    user_mean: list[float]
    healthy_mean: list[float]

    def __post_init__(self) -> None:
        self.round_index = int(self.round_index)
        self.tau = float(self.tau)
        self.baseline = float(self.baseline)
        self.momentum = float(self.momentum)
        self.mean_score = float(self.mean_score)
        self.user_mean = _safe_float_list(self.user_mean)
        self.healthy_mean = _safe_float_list(self.healthy_mean)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionData:
    records: list[RoundRecord] = field(default_factory=list)

    def sorted_records(self) -> list[RoundRecord]:
        return sorted(self.records, key=lambda row: (_sort_timestamp(row.timestamp), row.gesture, row.round_index))

    def to_dict(self) -> dict:
        return {"records": [record.to_dict() for record in self.sorted_records()]}


@dataclass
class ProgressReport:
    progression_index: float
    recovery_score: float
    recovery_velocity: float
    joint_details: dict[str, dict[str, float]]
    warnings: list[str]
    recommendation: str
    per_gesture_progression: dict[str, float]
    history_summary: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ProgressAnalyzer:
    def __init__(
        self,
        healthy_descriptors: dict[str, GestureDescriptor],
        initial_taus: dict[str, float],
        velocity_window: int = 7,
        plateau_window: int = 7,
        plateau_epsilon: float = 0.02,
        regression_window: int = 3,
        regression_threshold: float = -0.03,
        joint_weights: dict[str, float] | None = None,
        progression_weights: dict[str, float] | None = None,
    ) -> None:
        self.healthy_descriptors = healthy_descriptors
        self.initial_taus = {gesture: float(value) for gesture, value in initial_taus.items()}
        self.velocity_window = int(velocity_window)
        self.plateau_window = int(plateau_window)
        self.plateau_epsilon = float(plateau_epsilon)
        self.regression_window = int(regression_window)
        self.regression_threshold = float(regression_threshold)
        self.angle_feature_names = list(config.JOINT_NAMES)
        self._angle_indices = self._resolve_angle_indices()
        self._joint_weights = self._normalize_joint_weights(joint_weights)
        self._progression_weights = self._normalize_progression_weights(progression_weights)
        self._latest_session: SessionData | None = None
        self._latest_report: ProgressReport | None = None
        self._latest_daily_series: list[dict] = []

    def _resolve_angle_indices(self) -> dict[str, int]:
        if not self.healthy_descriptors:
            return {name: index for index, name in enumerate(self.angle_feature_names)}

        sample_descriptor = next(iter(self.healthy_descriptors.values()))
        indices: dict[str, int] = {}
        for name in self.angle_feature_names:
            if name not in sample_descriptor.feature_names:
                raise KeyError(f"Joint feature '{name}' is missing from descriptor feature names")
            indices[name] = sample_descriptor.feature_names.index(name)
        return indices

    def _normalize_joint_weights(self, joint_weights: dict[str, float] | None) -> dict[str, float]:
        if joint_weights is None:
            weight = 1.0 / len(self.angle_feature_names)
            return {name: weight for name in self.angle_feature_names}

        normalized = {name: float(max(joint_weights.get(name, 0.0), 0.0)) for name in self.angle_feature_names}
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("joint_weights must contain at least one positive weight")
        return {name: value / total for name, value in normalized.items()}

    def _normalize_progression_weights(self, weights: dict[str, float] | None) -> dict[str, float]:
        source = PROGRESSION_COMPONENT_WEIGHTS if weights is None else weights
        normalized = {
            name: float(max(source.get(name, 0.0), 0.0))
            for name in PROGRESSION_COMPONENT_WEIGHTS
        }
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("progression_weights must contain at least one positive weight")
        return {name: value / total for name, value in normalized.items()}

    def _tau0_for(self, gesture: str, fallback: float) -> float:
        tau0 = float(self.initial_taus.get(gesture, fallback))
        return max(tau0, 1e-6)

    def _progression_for_record(self, record: RoundRecord) -> float:
        tau0 = self._tau0_for(record.gesture, record.tau)
        return float(np.clip(1.0 - record.tau / tau0, 0.0, 1.0))

    def _descriptor_recovery_for_record(self, initial_record: RoundRecord, record: RoundRecord) -> float:
        initial_user = np.asarray(initial_record.user_mean, dtype=np.float64)
        current_user = np.asarray(record.user_mean, dtype=np.float64)
        healthy = np.asarray(record.healthy_mean, dtype=np.float64)
        initial_gap = float(np.linalg.norm(initial_user - healthy))
        current_gap = float(np.linalg.norm(current_user - healthy))
        if initial_gap < 1e-6:
            return 1.0 if current_gap < 1e-6 else 0.0
        return float(np.clip(1.0 - current_gap / initial_gap, 0.0, 1.0))

    def _baseline_margin_for_record(self, record: RoundRecord) -> float:
        margin = (float(record.mean_score) - float(record.baseline)) / 25.0
        return float(np.clip(margin, 0.0, 1.0))

    def _momentum_trend_for_record(self, record: RoundRecord) -> float:
        return float(np.clip(float(record.momentum) / 10.0, 0.0, 1.0))

    def _challenge_alignment_for_record(self, record: RoundRecord) -> float:
        if record.challenge_zone == "in_zone":
            return 1.0
        if record.challenge_zone == "too_easy":
            return 0.5
        return 0.0

    def _progress_components_for_record(
        self,
        initial_record: RoundRecord,
        record: RoundRecord,
    ) -> dict[str, float]:
        return {
            "strictness_progression": self._progression_for_record(record),
            "descriptor_recovery": self._descriptor_recovery_for_record(initial_record, record),
            "baseline_margin": self._baseline_margin_for_record(record),
            "momentum_trend": self._momentum_trend_for_record(record),
            "challenge_zone_alignment": self._challenge_alignment_for_record(record),
        }

    def _combine_progress_components(self, components: dict[str, float]) -> float:
        return float(
            np.clip(
                sum(components[name] * self._progression_weights[name] for name in self._progression_weights),
                0.0,
                1.0,
            )
        )

    def _progression_for_record_with_initial(
        self,
        initial_record: RoundRecord,
        record: RoundRecord,
    ) -> float:
        return self._combine_progress_components(self._progress_components_for_record(initial_record, record))

    def _compute_daily_progression(
        self,
        records: list[RoundRecord],
        initial_by_gesture: dict[str, RoundRecord],
    ) -> list[dict]:
        timestamps = sorted({record.timestamp for record in records}, key=_sort_timestamp)
        latest_by_gesture: dict[str, RoundRecord] = {}
        daily_rows: list[dict] = []

        for timestamp in timestamps:
            for record in records:
                if record.timestamp != timestamp:
                    continue
                latest_by_gesture[record.gesture] = record

            component_rows = [
                self._progress_components_for_record(initial_by_gesture[record.gesture], record)
                for record in latest_by_gesture.values()
            ]
            progressions = [self._combine_progress_components(components) for components in component_rows]
            progression_index = float(np.mean(progressions)) if progressions else 0.0
            row = {
                "timestamp": timestamp,
                "progression_index": progression_index,
            }
            for name in PROGRESSION_COMPONENT_WEIGHTS:
                values = [components[name] for components in component_rows]
                row[name] = float(np.mean(values)) if values else 0.0
            daily_rows.append(row)
        return daily_rows

    def _compute_velocities(self, progression_values: list[float]) -> list[float]:
        velocities: list[float] = []
        for index in range(len(progression_values)):
            start = max(0, index - self.velocity_window + 1)
            window = np.asarray(progression_values[start : index + 1], dtype=np.float64)
            if len(window) < 2:
                velocities.append(0.0)
                continue
            x = np.arange(len(window), dtype=np.float64)
            slope = float(np.polyfit(x, window, deg=1)[0])
            velocities.append(slope)
        return velocities

    def _compute_joint_recovery(
        self,
        initial_record: RoundRecord,
        latest_record: RoundRecord,
    ) -> dict[str, float]:
        ratios: dict[str, float] = {}
        initial_user = np.asarray(initial_record.user_mean, dtype=np.float64)
        latest_user = np.asarray(latest_record.user_mean, dtype=np.float64)
        healthy = np.asarray(latest_record.healthy_mean, dtype=np.float64)

        for name, index in self._angle_indices.items():
            baseline_gap = abs(initial_user[index] - healthy[index])
            current_gap = abs(latest_user[index] - healthy[index])
            ratio = 1.0 - current_gap / max(baseline_gap, 1e-6)
            ratios[name] = float(np.clip(ratio, 0.0, 1.0))
        return ratios

    def _aggregate_joint_recovery(self, joint_details: dict[str, dict[str, float]]) -> dict[str, float]:
        if not joint_details:
            return {name: 0.0 for name in self.angle_feature_names}

        aggregated: dict[str, float] = {}
        for name in self.angle_feature_names:
            values = [details[name] for details in joint_details.values() if name in details]
            aggregated[name] = float(np.mean(values)) if values else 0.0
        return aggregated

    def _compute_recovery_score(self, aggregate_joint: dict[str, float]) -> float:
        weighted = sum(aggregate_joint[name] * self._joint_weights[name] for name in self.angle_feature_names)
        return float(np.clip(weighted * 100.0, 0.0, 100.0))

    def detect_plateau(self) -> str | None:
        if len(self._latest_daily_series) < self.plateau_window:
            return None
        recent = self._latest_daily_series[-self.plateau_window :]
        if all(abs(float(row["recovery_velocity"])) < self.plateau_epsilon for row in recent):
            return f"Progress has plateaued for {self.plateau_window} consecutive days; adjust the training plan."
        return None

    def detect_regression(self) -> str | None:
        if len(self._latest_daily_series) < self.regression_window:
            return None
        recent = self._latest_daily_series[-self.regression_window :]
        if all(float(row["recovery_velocity"]) < self.regression_threshold for row in recent):
            return f"Regression has persisted for {self.regression_window} consecutive days; consult a clinician or reduce intensity."
        return None

    def _build_recommendation(self, warnings: list[str], aggregate_joint: dict[str, float]) -> str:
        slowest = sorted(aggregate_joint.items(), key=lambda item: item[1])[:3]
        slowest_names = ", ".join(name for name, _ in slowest)

        if any("Regression" in warning for warning in warnings):
            prefix = "Consult a clinician or reduce training intensity"
        elif any("plateaued" in warning for warning in warnings):
            prefix = "Adjust the training plan"
        else:
            prefix = "Continue the current plan"

        if slowest_names:
            return f"{prefix}; focus on: {slowest_names}"
        return prefix

    def get_joint_recovery(self, gesture: str) -> dict[str, float]:
        if self._latest_report is None:
            raise RuntimeError("No progress report available. Call update() first.")
        return dict(self._latest_report.joint_details.get(gesture, {}))

    def export_report(self) -> ProgressReport:
        if self._latest_report is None:
            raise RuntimeError("No progress report available. Call update() first.")
        return self._latest_report

    def update(self, session_data: SessionData) -> ProgressReport:
        records = session_data.sorted_records()
        if not records:
            raise ValueError("session_data.records must not be empty")

        by_gesture: dict[str, list[RoundRecord]] = {}
        for record in records:
            by_gesture.setdefault(record.gesture, []).append(record)

        latest_by_gesture = {gesture: gesture_records[-1] for gesture, gesture_records in by_gesture.items()}
        initial_by_gesture = {gesture: gesture_records[0] for gesture, gesture_records in by_gesture.items()}

        per_gesture_progression = {
            gesture: self._progression_for_record_with_initial(initial_by_gesture[gesture], record)
            for gesture, record in latest_by_gesture.items()
        }
        progression_index = float(np.mean(list(per_gesture_progression.values())))

        joint_details = {
            gesture: self._compute_joint_recovery(initial_by_gesture[gesture], latest_record)
            for gesture, latest_record in latest_by_gesture.items()
        }
        aggregate_joint = self._aggregate_joint_recovery(joint_details)
        recovery_score = self._compute_recovery_score(aggregate_joint)

        daily_progression = self._compute_daily_progression(records, initial_by_gesture)
        velocities = self._compute_velocities([row["progression_index"] for row in daily_progression])
        for row, velocity in zip(daily_progression, velocities):
            row["recovery_velocity"] = float(velocity)
        recovery_velocity = float(velocities[-1] if velocities else 0.0)
        self._latest_daily_series = daily_progression

        warnings = []
        plateau_warning = self.detect_plateau()
        if plateau_warning:
            warnings.append(plateau_warning)
        regression_warning = self.detect_regression()
        if regression_warning:
            warnings.append(regression_warning)

        history_summary = {
            "n_records": len(records),
            "n_days": len(daily_progression),
            "date_start": daily_progression[0]["timestamp"],
            "date_end": daily_progression[-1]["timestamp"],
            "daily_progression": daily_progression,
            "progression_component_weights": dict(self._progression_weights),
            "per_gesture_tau_series": {
                gesture: [
                    {
                        "timestamp": record.timestamp,
                        "tau": float(record.tau),
                        "round_index": int(record.round_index),
                    }
                    for record in gesture_records
                ]
                for gesture, gesture_records in by_gesture.items()
            },
            "overall_joint_recovery": aggregate_joint,
        }

        report = ProgressReport(
            progression_index=progression_index,
            recovery_score=recovery_score,
            recovery_velocity=recovery_velocity,
            joint_details=joint_details,
            warnings=warnings,
            recommendation=self._build_recommendation(warnings, aggregate_joint),
            per_gesture_progression=per_gesture_progression,
            history_summary=history_summary,
        )
        self._latest_session = session_data
        self._latest_report = report
        return report
