import json
from pathlib import Path

from experiments.analysis import ExperimentAnalyzer
from experiments.plot_phase5 import generate_phase5_figures
from experiments.runner import ExperimentRunner
from experiments.strategies import AdaptiveScoringStrategy, FixedStandardizedStrategy, HeuristicRuleStrategy
from experiments.virtual_patient import VirtualPatient
from src import config
from src.scorer import ColdStartScorer, load_descriptors, compute_expected_distances


DESCRIPTOR_DIR = config.DATA_PROCESSED_DIR / "descriptors"
FEATURES_DIR = config.DATA_PROCESSED_DIR / "features"


def make_scorer() -> ColdStartScorer:
    descriptors = load_descriptors(DESCRIPTOR_DIR)
    expected = compute_expected_distances(descriptors, FEATURES_DIR)
    return ColdStartScorer(descriptors, expected)


def test_virtual_patient_progress_and_training_feedback():
    scorer = make_scorer()
    patient = VirtualPatient(
        patient_id="linear-p01",
        healthy_descriptors=scorer.descriptors,
        recovery_curve="linear_recovery",
        seed=123,
    )

    early = patient.ability_at(0, 10)
    late = patient.ability_at(9, 10)
    patient.apply_training_feedback(1.0)
    boosted = patient.ability_at(9, 10)

    assert late > early
    assert boosted >= late


def test_strategies_return_valid_scores_and_adaptive_round_progress():
    scorer = make_scorer()
    feature_names = list(next(iter(scorer.descriptors.values())).feature_names)
    patient = VirtualPatient(
        patient_id="curve-p01",
        healthy_descriptors=scorer.descriptors,
        recovery_curve="s_curve_recovery",
        seed=321,
    )
    vector = patient.perform_gesture("fist", session_index=1, total_sessions=6)

    fixed = FixedStandardizedStrategy(make_scorer())
    heuristic = HeuristicRuleStrategy(feature_names)
    adaptive = AdaptiveScoringStrategy(make_scorer(), round_size=10)

    fixed_result = fixed.score_attempt(vector, "fist")
    heuristic_result = heuristic.score_attempt(vector, "fist")
    for _ in range(10):
        adaptive_result = adaptive.score_attempt(vector, "fist")

    assert 0 <= fixed_result.display_score <= 100
    assert 0 <= heuristic_result.display_score <= 100
    assert 0 <= adaptive_result.display_score <= 100
    assert adaptive.threshold_manager.get_state("fist").round_index >= 1


def test_experiment_runner_and_analyzer_produce_strategy_metrics():
    runner = ExperimentRunner(
        descriptor_dir=DESCRIPTOR_DIR,
        features_dir=FEATURES_DIR,
        patients_per_curve=1,
        n_sessions=3,
        attempts_per_session=3,
        strategy_names=[
            FixedStandardizedStrategy.name,
            AdaptiveScoringStrategy.name,
        ],
        recovery_curves=["linear_recovery", "plateau"],
        seed=42,
    )
    result = runner.run()
    report = ExperimentAnalyzer().analyze(result)

    assert len(result.attempts) > 0
    assert len(result.sessions) > 0
    assert set(report.strategy_metrics.keys()) == {
        FixedStandardizedStrategy.name,
        AdaptiveScoringStrategy.name,
    }
    payload = report.to_dict()
    assert payload["metric_definitions"]["ability_gain"].startswith("Final post-session")
    assert "phase4_relation" in payload["narrative"]
    for metrics in report.strategy_metrics.values():
        assert 0.0 <= metrics.challenge_ratio <= 1.0
        assert 0.0 <= metrics.feedback_accuracy <= 1.0
        assert 0.0 <= metrics.frustration_risk <= 1.0
    for strategy in payload["strategy_metrics"]:
        assert "ability_gain" in payload["confidence_intervals"][strategy]


def test_phase5_figure_generator_writes_expected_pngs(tmp_path):
    runner = ExperimentRunner(
        descriptor_dir=DESCRIPTOR_DIR,
        features_dir=FEATURES_DIR,
        patients_per_curve=1,
        n_sessions=3,
        attempts_per_session=2,
        strategy_names=[
            FixedStandardizedStrategy.name,
            AdaptiveScoringStrategy.name,
        ],
        recovery_curves=["linear_recovery", "plateau"],
        seed=7,
    )
    result = runner.run()
    report = ExperimentAnalyzer().analyze(result)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False), encoding="utf-8")

    outputs = generate_phase5_figures(summary_path, tmp_path)

    assert set(outputs) == {
        "phase5_strategy_metrics",
        "phase5_scenario_heatmap",
        "phase5_effect_sizes",
    }
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
