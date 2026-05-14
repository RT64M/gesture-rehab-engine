from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import SCENARIO_DESCRIPTIONS, SCENARIO_LABELS
from .threshold_manager import MomentumThresholdManager


SCENARIO_ORDER = [
    "linear_recovery",
    "s_curve_recovery",
    "plateau",
    "fatigue_dip",
]


DEFAULT_STAGE3_CONFIG = {
    "tau_init": 5.0,
    "beta": 0.9,
    "eta": 0.1,
    "theta_activate": 0.05,
    "tau_min": 1.0,
    "tau_max": 10.0,
    "alpha_baseline": 0.95,
    "eta_relax": 0.03,
    "round_size": 10,
    "challenge_low": 60.0,
    "challenge_high": 85.0,
}


@dataclass
class SimulationTrace:
    gesture: str
    scenario: str
    rounds: list[dict]

    def to_dict(self) -> dict:
        return {
            "gesture": self.gesture,
            "scenario": self.scenario,
            "label": SCENARIO_LABELS[self.scenario],
            "description": SCENARIO_DESCRIPTIONS[self.scenario],
            "rounds": self.rounds,
        }


def build_round_means(
    scenario: str,
    num_rounds: int,
    recovery_speed: float,
) -> np.ndarray:
    num_rounds = max(int(num_rounds), 1)
    recovery_speed = float(np.clip(recovery_speed, 0.2, 2.5))
    x = np.linspace(0.0, 1.0, num_rounds)
    start = 55.0
    end = min(92.0, 55.0 + 28.0 * recovery_speed)

    if scenario == "linear_recovery":
        means = start + (end - start) * x
    elif scenario == "s_curve_recovery":
        sigmoid = 1.0 / (1.0 + np.exp(-8.0 * (x - 0.5)))
        sigmoid = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
        means = start + (end - start) * sigmoid
    elif scenario == "plateau":
        progress = np.minimum(x * (1.6 * recovery_speed), 0.72)
        means = start + (end - start) * progress
    elif scenario == "fatigue_dip":
        base = start + (end - start) * x
        dip_center = 0.7
        dip = np.exp(-((x - dip_center) ** 2) / 0.003) * (42.0 + 10.0 / recovery_speed)
        recovery_boost = np.maximum(x - 0.82, 0.0) * 14.0
        means = base - dip + recovery_boost
    else:
        raise KeyError(f"Unknown scenario: {scenario}")
    return np.clip(means, 25.0, 97.0)


def _scenario_seed(scenario: str, num_rounds: int, noise: float, recovery_speed: float) -> int:
    return abs(hash((scenario, num_rounds, round(noise, 2), round(recovery_speed, 2)))) % (2**32)


def simulate_progression_trace(
    gesture: str,
    scenario: str,
    num_rounds: int = 12,
    noise: float = 4.0,
    recovery_speed: float = 1.0,
    seed: int | None = None,
) -> SimulationTrace:
    means = build_round_means(scenario, num_rounds=num_rounds, recovery_speed=recovery_speed)
    rng = np.random.default_rng(_scenario_seed(scenario, num_rounds, noise, recovery_speed) if seed is None else seed)
    manager = MomentumThresholdManager(**DEFAULT_STAGE3_CONFIG)
    rounds: list[dict] = []

    for target_mean in means:
        scores = np.clip(
            rng.normal(loc=float(target_mean), scale=float(max(noise, 0.1)), size=DEFAULT_STAGE3_CONFIG["round_size"]),
            0.0,
            100.0,
        )
        for score in scores:
            manager.register_attempt(gesture, float(score))
        update = manager.on_round_end(gesture)
        rounds.append(
            {
                "round_index": update.round_index,
                "scores": [round(float(value), 2) for value in scores],
                "mean_score": round(update.mean_score, 2),
                "tau": round(update.tau_new, 4),
                "baseline": round(update.baseline, 4),
                "momentum": round(update.momentum, 4),
                "direction": update.direction,
                "reason": update.reason,
                "challenge_zone": update.challenge_zone,
            }
        )

    return SimulationTrace(gesture=gesture, scenario=scenario, rounds=rounds)
