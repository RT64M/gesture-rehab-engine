from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.plugin_api import DEFAULT_DESCRIPTOR_DIR, DEFAULT_FEATURES_DIR, score_plugin_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one plugin JSON request.")
    parser.add_argument("--input", type=Path, required=True, help="Path to plugin request JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Path where response JSON will be written.")
    parser.add_argument(
        "--state",
        type=Path,
        default=config.ARTIFACTS_DIR / "plugin_state.json",
        help="Path to persistent plugin state JSON. Use --no-state to disable persistence.",
    )
    parser.add_argument("--no-state", action="store_true", help="Do not read or write persistent user state.")
    parser.add_argument(
        "--descriptor-dir",
        type=Path,
        default=DEFAULT_DESCRIPTOR_DIR,
        help="Directory containing descriptor JSON files.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help="Directory containing feature NPZ files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    response = score_plugin_request(
        payload,
        descriptor_dir=args.descriptor_dir,
        features_dir=args.features_dir,
        state_path=None if args.no_state else args.state,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Plugin score response written to: {args.output}")
    print(f"gesture={response['target_gesture']} score={response['display_score']} predicted={response['predicted_gesture']}")


if __name__ == "__main__":
    main()
