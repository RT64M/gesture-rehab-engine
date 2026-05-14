from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.demo_profiles import nearest_competitor
from src.progress_analyzer import ProgressAnalyzer, RoundRecord, SessionData
from src.progression import DEFAULT_STAGE3_CONFIG, SCENARIO_ORDER, build_round_means
from src.report_generator import generate_progress_report
from src.scorer import ColdStartScorer
from src.threshold_manager import MomentumThresholdManager


def _build_direction(gesture: str, scorer: ColdStartScorer) -> tuple[np.ndarray, np.ndarray]:
    descriptor = scorer.descriptors[gesture]
    competitor = nearest_competitor(gesture, scorer.descriptors)
    delta = scorer.descriptors[competitor].mu - descriptor.mu
    direction = np.sign(delta)
    direction[direction == 0.0] = 1.0
    std = np.sqrt(np.clip(np.diag(descriptor.sigma), 1e-8, None))
    return direction, std


def _severity_from_target_score(score: float) -> float:
    severity = np.interp(float(score), [25.0, 97.0], [1.8, 0.05])
    return float(np.clip(severity, 0.05, 1.8))


def _sample_round_vectors(
    rng: np.random.Generator,
    descriptor_mu: np.ndarray,
    direction: np.ndarray,
    std: np.ndarray,
    target_score: float,
    round_size: int,
) -> np.ndarray:
    severity = _severity_from_target_score(target_score)
    center = descriptor_mu + severity * std * direction
    noise_scale = np.clip(0.14 + severity * 0.05, 0.08, 0.24)
    covariance = np.diag(np.square(std * noise_scale)) + 1e-6 * np.eye(len(std))
    return rng.multivariate_normal(center, covariance, size=round_size)


def build_session_data(
    scorer: ColdStartScorer,
    scenario: str,
    days: int,
    gestures: list[str] | None = None,
    seed: int | None = None,
) -> SessionData:
    rng = np.random.default_rng(seed)
    manager = MomentumThresholdManager(**DEFAULT_STAGE3_CONFIG)
    records: list[RoundRecord] = []
    means = build_round_means(scenario, num_rounds=days, recovery_speed=1.0)
    start_day = date(2026, 1, 1)

    active_gestures = list(gestures) if gestures is not None else list(scorer.descriptors)
    directions = {
        gesture: _build_direction(gesture, scorer)
        for gesture in active_gestures
    }

    for day_index, target_mean in enumerate(means):
        timestamp = (start_day + timedelta(days=day_index)).isoformat()
        for gesture in active_gestures:
            descriptor = scorer.descriptors[gesture]
            direction, std = directions[gesture]
            samples = _sample_round_vectors(
                rng=rng,
                descriptor_mu=descriptor.mu,
                direction=direction,
                std=std,
                target_score=float(target_mean),
                round_size=DEFAULT_STAGE3_CONFIG["round_size"],
            )

            for vector in samples:
                tau_override = manager.get_current_tau(gesture)
                result = scorer.score(
                    vector,
                    gesture,
                    update_user_model=True,
                    tau_override=tau_override,
                )
                manager.register_attempt(gesture, result.display_score, result.raw_score)

            update = manager.on_round_end(gesture)
            user_mean = scorer.get_user_model(gesture).user_mean
            records.append(
                RoundRecord(
                    timestamp=timestamp,
                    gesture=gesture,
                    round_index=update.round_index,
                    tau=update.tau_new,
                    baseline=update.baseline,
                    momentum=update.momentum,
                    mean_score=update.mean_score,
                    challenge_zone=update.challenge_zone,
                    user_mean=user_mean.tolist(),
                    healthy_mean=descriptor.mu.tolist(),
                )
            )

    return SessionData(records=records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a phase-4 offline progress report demo.")
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_ORDER,
        default="linear_recovery",
        help="Simulated recovery scenario.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of simulated days / round-end snapshots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260422,
        help="Random seed for reproducible simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR / "phase4_demo",
        help="Directory used for report.json, html, and plots.",
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
    descriptor_dir = config.DATA_PROCESSED_DIR / "descriptors"
    features_dir = config.DATA_PROCESSED_DIR / "features"
    scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir)
    active_gestures = list(args.gestures) if args.gestures is not None else list(config.TARGET_GESTURES)
    unknown = sorted(set(active_gestures) - set(scorer.descriptors))
    if unknown:
        raise KeyError(f"Unsupported gestures: {unknown}")

    session_data = build_session_data(
        scorer=scorer,
        scenario=args.scenario,
        days=max(int(args.days), 1),
        gestures=active_gestures,
        seed=args.seed,
    )

    analyzer = ProgressAnalyzer(
        healthy_descriptors={gesture: scorer.descriptors[gesture] for gesture in active_gestures},
        initial_taus={gesture: DEFAULT_STAGE3_CONFIG["tau_init"] for gesture in active_gestures},
    )
    report = analyzer.update(session_data)
    artifacts = generate_progress_report(report, session_data, args.output_dir)

    print(f"Scenario: {args.scenario}")
    print(f"Gestures: {', '.join(active_gestures)}")
    print(f"Days: {args.days}")
    print(f"Progression index: {report.progression_index:.3f}")
    print(f"Recovery score: {report.recovery_score:.1f}")
    print(f"Recovery velocity: {report.recovery_velocity:.4f}")
    print(f"Warnings: {report.warnings or ['No warnings']}")
    print(f"Recommendation: {report.recommendation}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
