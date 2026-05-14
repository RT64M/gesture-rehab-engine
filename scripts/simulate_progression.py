from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.labels import SCENARIO_LABELS
from src.progression import SCENARIO_ORDER, simulate_progression_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate Stage 3 threshold progression.")
    parser.add_argument("--gesture", default="fist", help="Gesture key to simulate.")
    parser.add_argument("--scenario", choices=SCENARIO_ORDER, default="linear_recovery")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--noise", type=float, default=4.0)
    parser.add_argument("--recovery-speed", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print full JSON trace.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace = simulate_progression_trace(
        gesture=args.gesture,
        scenario=args.scenario,
        num_rounds=args.rounds,
        noise=args.noise,
        recovery_speed=args.recovery_speed,
    )
    payload = trace.to_dict()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print(f"Stage 3 simulation: {SCENARIO_LABELS[args.scenario]} / gesture={args.gesture}")
    print("=" * 72)
    for row in payload["rounds"]:
        print(
            f"round={row['round_index']:>2}  mean={row['mean_score']:>5.1f}  tau={row['tau']:.3f}  "
            f"baseline={row['baseline']:.2f}  momentum={row['momentum']:.3f}  "
            f"zone={row['challenge_zone']}  {row['direction']}"
        )


if __name__ == "__main__":
    main()

