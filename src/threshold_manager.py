from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .challenge_zone import IN_ZONE, challenge_zone_status


@dataclass
class ThresholdState:
    gesture: str
    tau: float
    baseline: float
    momentum: float
    round_index: int
    challenge_zone: str
    pending_attempts: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThresholdUpdate:
    tau_old: float
    tau_new: float
    momentum: float
    baseline: float
    direction: str
    reason: str
    challenge_zone: str
    mean_score: float
    round_index: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _GestureRuntime:
    gesture: str
    tau: float
    baseline: float | None = None
    momentum: float = 0.0
    round_index: int = 0
    pending_scores: list[float] = field(default_factory=list)
    pending_raw_scores: list[float | None] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    last_mean_score: float | None = None
    challenge_zone: str = IN_ZONE


class MomentumThresholdManager:
    def __init__(
        self,
        tau_init: float = 5.0,
        beta: float = 0.9,
        eta: float = 0.1,
        theta_activate: float = 0.05,
        tau_min: float = 1.0,
        tau_max: float = 10.0,
        alpha_baseline: float = 0.95,
        eta_relax: float = 0.03,
        round_size: int = 10,
        challenge_low: float = 60.0,
        challenge_high: float = 85.0,
    ) -> None:
        self.tau_init = float(tau_init)
        self.beta = float(beta)
        self.eta = float(eta)
        self.theta_activate = float(theta_activate)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.alpha_baseline = float(alpha_baseline)
        self.eta_relax = float(eta_relax)
        self.round_size = int(round_size)
        self.challenge_low = float(challenge_low)
        self.challenge_high = float(challenge_high)
        self._gestures: dict[str, _GestureRuntime] = {}

    def _runtime(self, gesture: str) -> _GestureRuntime:
        runtime = self._gestures.get(gesture)
        if runtime is None:
            runtime = _GestureRuntime(gesture=gesture, tau=self.tau_init)
            self._gestures[gesture] = runtime
        return runtime

    def register_attempt(self, gesture: str, display_score: float, raw_score: float | None = None) -> None:
        runtime = self._runtime(gesture)
        runtime.pending_scores.append(float(display_score))
        runtime.pending_raw_scores.append(None if raw_score is None else float(raw_score))

    def has_full_round(self, gesture: str) -> bool:
        return len(self._runtime(gesture).pending_scores) >= self.round_size

    def on_round_end(self, gesture: str) -> ThresholdUpdate:
        runtime = self._runtime(gesture)
        if not runtime.pending_scores:
            raise ValueError(f"No pending attempts for gesture '{gesture}'")

        round_scores = runtime.pending_scores[: self.round_size]
        runtime.pending_scores = runtime.pending_scores[self.round_size :]
        runtime.pending_raw_scores = runtime.pending_raw_scores[self.round_size :]

        mean_score = float(np.mean(round_scores))
        tau_old = float(runtime.tau)
        challenge = challenge_zone_status(mean_score, low=self.challenge_low, high=self.challenge_high)

        if runtime.baseline is None:
            runtime.baseline = mean_score
            runtime.last_mean_score = mean_score
            runtime.challenge_zone = challenge.code
            runtime.round_index += 1
            update = ThresholdUpdate(
                tau_old=tau_old,
                tau_new=tau_old,
                momentum=runtime.momentum,
                baseline=float(runtime.baseline),
                direction="hold",
                reason="initialize_baseline",
                challenge_zone=challenge.code,
                mean_score=mean_score,
                round_index=runtime.round_index,
            )
            runtime.history.append(update.to_dict() | {"gesture": gesture})
            return update

        performance_signal = mean_score - runtime.baseline
        momentum = self.beta * runtime.momentum + (1.0 - self.beta) * performance_signal

        tau_new = tau_old
        direction = "hold"
        reason = "momentum_below_activation"

        if mean_score < runtime.baseline * 0.7:
            tau_new = min(tau_old + self.eta_relax * abs(performance_signal), self.tau_max)
            direction = "relax"
            reason = "fatigue_protection"
        elif momentum > self.theta_activate:
            tau_new = max(tau_old - self.eta * momentum, self.tau_min)
            direction = "tighten"
            reason = "sustained_improvement"

        baseline_new = self.alpha_baseline * runtime.baseline + (1.0 - self.alpha_baseline) * mean_score

        runtime.tau = float(np.clip(tau_new, self.tau_min, self.tau_max))
        runtime.baseline = float(baseline_new)
        runtime.momentum = float(momentum)
        runtime.round_index += 1
        runtime.last_mean_score = mean_score
        runtime.challenge_zone = challenge.code

        update = ThresholdUpdate(
            tau_old=tau_old,
            tau_new=runtime.tau,
            momentum=runtime.momentum,
            baseline=runtime.baseline,
            direction=direction,
            reason=reason,
            challenge_zone=challenge.code,
            mean_score=mean_score,
            round_index=runtime.round_index,
        )
        runtime.history.append(update.to_dict() | {"gesture": gesture})
        return update

    def get_state(self, gesture: str) -> ThresholdState:
        runtime = self._runtime(gesture)
        baseline = runtime.baseline if runtime.baseline is not None else 70.0
        return ThresholdState(
            gesture=gesture,
            tau=float(runtime.tau),
            baseline=float(baseline),
            momentum=float(runtime.momentum),
            round_index=int(runtime.round_index),
            challenge_zone=runtime.challenge_zone,
            pending_attempts=len(runtime.pending_scores),
        )

    def get_current_tau(self, gesture: str) -> float:
        return self.get_state(gesture).tau

    def export_history(self) -> list[dict]:
        rows: list[dict] = []
        for runtime in self._gestures.values():
            rows.extend(runtime.history)
        return rows

