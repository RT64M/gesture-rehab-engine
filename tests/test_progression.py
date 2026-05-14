from src.progression import simulate_progression_trace


def test_linear_recovery_trace_generally_tightens_tau():
    trace = simulate_progression_trace("fist", "linear_recovery", num_rounds=10, noise=2.0, recovery_speed=1.0)
    taus = [row["tau"] for row in trace.rounds]
    assert taus[-1] < taus[0]


def test_plateau_trace_stabilizes_tau_late_in_training():
    trace = simulate_progression_trace("fist", "plateau", num_rounds=12, noise=1.5, recovery_speed=1.0)
    tail = [row["tau"] for row in trace.rounds[-3:]]
    assert max(tail) - min(tail) < 0.2


def test_fatigue_dip_trace_triggers_relaxation():
    trace = simulate_progression_trace("fist", "fatigue_dip", num_rounds=12, noise=2.0, recovery_speed=1.0)
    directions = [row["direction"] for row in trace.rounds]
    assert "relax" in directions
