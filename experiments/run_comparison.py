from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

from experiments.analysis import ExperimentAnalyzer
from experiments.plot_phase5 import generate_phase5_figures
from experiments.runner import ExperimentRunner
from experiments.strategies import AdaptiveScoringStrategy, FixedStandardizedStrategy, HeuristicRuleStrategy
from experiments.virtual_patient import CURVE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-5 strategy comparison preparation.")
    parser.add_argument(
        "--patients-per-curve",
        type=int,
        default=3,
        help="Number of virtual patients generated for each recovery curve.",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=8,
        help="Number of simulated sessions per patient-strategy pair.",
    )
    parser.add_argument(
        "--attempts-per-session",
        type=int,
        default=10,
        help="Number of attempts per gesture within a session.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR / "phase5_prep",
        help="Directory for summary and logs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=config.REPORT_FIGURES_DIR,
        help="Directory where phase-5 paper figures will be written.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip phase-5 figure generation.",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=None,
        help="Explicit gesture keys. Default uses the configured ten target gestures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    active_gestures = list(args.gestures) if args.gestures is not None else list(config.TARGET_GESTURES)

    runner = ExperimentRunner(
        descriptor_dir=config.DATA_PROCESSED_DIR / "descriptors",
        features_dir=config.DATA_PROCESSED_DIR / "features",
        patients_per_curve=args.patients_per_curve,
        n_sessions=args.sessions,
        attempts_per_session=args.attempts_per_session,
        strategy_names=[
            FixedStandardizedStrategy.name,
            HeuristicRuleStrategy.name,
            AdaptiveScoringStrategy.name,
        ],
        recovery_curves=list(CURVE_NAMES),
        gestures=active_gestures,
    )
    result = runner.run()
    report = ExperimentAnalyzer().analyze(result)
    summary_payload = report.to_dict()
    summary_payload["experiment_config"] = {
        "gestures": list(active_gestures),
        "patients_per_curve": args.patients_per_curve,
        "sessions": args.sessions,
        "attempts_per_session": args.attempts_per_session,
        "recovery_curves": list(CURVE_NAMES),
        "virtual_subjects_only": True,
    }

    (output_dir / "attempt_logs.json").write_text(
        json.dumps([row.to_dict() for row in result.attempts], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "session_logs.json").write_text(
        json.dumps([row.to_dict() for row in result.sessions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figure_outputs = {}
    if not args.no_figures:
        figure_outputs = generate_phase5_figures(summary_path, args.figures_dir)

    print(f"Curves: {', '.join(CURVE_NAMES)}")
    print(f"Gestures: {', '.join(active_gestures)}")
    print(f"Patients per curve: {args.patients_per_curve}")
    print(f"Sessions: {args.sessions}")
    print(f"Attempts per session: {args.attempts_per_session}")
    print("")
    for strategy, metrics in report.strategy_metrics.items():
        print(strategy)
        print(f"  ability_gain: {metrics.ability_gain:.4f}")
        print(f"  challenge_ratio: {metrics.challenge_ratio:.4f}")
        print(f"  feedback_accuracy: {metrics.feedback_accuracy:.4f}")
        print(f"  frustration_risk: {metrics.frustration_risk:.4f}")
        print(f"  score_ability_correlation: {metrics.score_ability_correlation:.4f}")
        print(f"  scenario_variance: {metrics.scenario_variance:.6f}")
    print("")
    print(f"Summary written to: {summary_path}")
    for key, path in figure_outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
