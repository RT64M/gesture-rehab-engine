from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.demo_profiles import PROFILE_ORDER, build_demo_profiles
from src.scorer import ColdStartScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stage 2 cold-start scoring demo.")
    parser.add_argument(
        "--gesture",
        choices=config.TARGET_GESTURES,
        default="fist",
        help="Target gesture to score.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_ORDER,
        default="standard",
        help="Synthetic sample profile to simulate.",
    )
    parser.add_argument(
        "--descriptor-dir",
        type=Path,
        default=config.DATA_PROCESSED_DIR / "descriptors",
        help="Directory containing descriptor json files.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=config.DATA_PROCESSED_DIR / "features",
        help="Directory containing train/test npz files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scorer = ColdStartScorer.from_artifacts(args.descriptor_dir, args.features_dir)
    profiles = build_demo_profiles(args.gesture, scorer.descriptors)
    profile = profiles[args.profile]

    print("=" * 64)
    print("Stage 2 Cold-Start Scoring Demo")
    print("=" * 64)
    print(f"target gesture : {args.gesture}")
    print(f"profile        : {profile.name} ({profile.label})")
    print(f"reference drift: toward {profile.competitor_gesture}")
    print("")

    for step, vector in enumerate(profile.vectors, start=1):
        result = scorer.score(vector, args.gesture, update_user_model=True)
        print(f"[step {step}] score={result.display_score:3d}  raw={result.raw_score:.4f}  "
              f"distance={result.mahalanobis_distance:.3f}  tau={result.tau:.3f}  "
              f"predicted={result.predicted_gesture}  blend={result.blend_weight:.2f}")
        for line in result.feedback:
            print(f"  - {line}")
        print("")


if __name__ == "__main__":
    main()

