import math

from src.challenge_zone import IN_ZONE, TOO_EASY, TOO_HARD, get_challenge_zone
from src.threshold_manager import MomentumThresholdManager


def finalize_round(manager: MomentumThresholdManager, gesture: str, scores: list[float]):
    for score in scores:
        manager.register_attempt(gesture, score)
    return manager.on_round_end(gesture)


def test_challenge_zone_thresholds():
    assert get_challenge_zone(45.0) == TOO_HARD
    assert get_challenge_zone(72.0) == IN_ZONE
    assert get_challenge_zone(92.0) == TOO_EASY


def test_threshold_tightens_under_sustained_improvement():
    manager = MomentumThresholdManager(round_size=5, beta=0.0, eta=0.1, theta_activate=0.05)
    finalize_round(manager, "fist", [60, 60, 60, 60, 60])
    update = finalize_round(manager, "fist", [90, 90, 90, 90, 90])
    assert update.direction == "tighten"
    assert update.tau_new < update.tau_old


def test_threshold_holds_under_small_fluctuation():
    manager = MomentumThresholdManager(round_size=5, beta=0.0, eta=0.1, theta_activate=5.0)
    finalize_round(manager, "fist", [70, 70, 70, 70, 70])
    update = finalize_round(manager, "fist", [72, 71, 69, 70, 68])
    assert update.direction == "hold"
    assert math.isclose(update.tau_new, update.tau_old)


def test_fatigue_protection_relaxes_tau():
    manager = MomentumThresholdManager(round_size=5, beta=0.0, eta_relax=0.03, tau_init=5.0)
    finalize_round(manager, "fist", [80, 80, 80, 80, 80])
    update = finalize_round(manager, "fist", [40, 42, 39, 41, 38])
    assert update.direction == "relax"
    assert update.reason == "fatigue_protection"
    assert update.tau_new > update.tau_old


def test_baseline_ema_updates_correctly():
    manager = MomentumThresholdManager(round_size=5, alpha_baseline=0.8)
    first = finalize_round(manager, "fist", [60, 60, 60, 60, 60])
    second = finalize_round(manager, "fist", [80, 80, 80, 80, 80])
    expected = 0.8 * first.baseline + 0.2 * 80.0
    assert math.isclose(second.baseline, expected)


def test_tau_is_clamped_to_bounds():
    manager = MomentumThresholdManager(round_size=5, beta=0.0, eta=10.0, tau_min=2.5, tau_max=6.0)
    finalize_round(manager, "fist", [60, 60, 60, 60, 60])
    update = finalize_round(manager, "fist", [100, 100, 100, 100, 100])
    assert update.tau_new >= 2.5

    manager = MomentumThresholdManager(round_size=5, beta=0.0, eta_relax=10.0, tau_init=5.5, tau_max=6.0)
    finalize_round(manager, "fist", [80, 80, 80, 80, 80])
    update = finalize_round(manager, "fist", [10, 10, 10, 10, 10])
    assert update.tau_new <= 6.0
