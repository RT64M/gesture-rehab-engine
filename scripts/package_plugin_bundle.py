from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.plugin_api import PLUGIN_API_VERSION


BUNDLE_FILES = [
    "README.md",
    "INTERFACE.md",
    "pyproject.toml",
    "uv.lock",
    "src/plugin_api.py",
    "src/config.py",
    "src/descriptors.py",
    "src/features.py",
    "src/geometry.py",
    "src/scorer.py",
    "src/user_model.py",
    "src/feedback.py",
    "src/challenge_zone.py",
    "scripts/plugin_score.py",
    "examples/plugin_integration/README.md",
    "examples/plugin_integration/request_feature_vector.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the stable plugin JSON adapter bundle.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "exports",
        help="Directory where the plugin bundle zip will be written.",
    )
    parser.add_argument(
        "--archive-name",
        type=str,
        default=None,
        help="Optional archive filename. Defaults to gesture_plugin_bundle_<timestamp>.zip.",
    )
    parser.add_argument(
        "--include-descriptors",
        action="store_true",
        help="Include processed descriptor JSON files in the bundle.",
    )
    return parser.parse_args()


def manifest(include_descriptors: bool) -> dict:
    return {
        "name": "gesture-scoring-plugin-adapter",
        "plugin_api_version": PLUGIN_API_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_gestures": list(config.TARGET_GESTURES),
        "feature_dimension": config.N_GEOMETRIC_FEATURES,
        "entrypoint": "scripts/plugin_score.py",
        "request_examples": ["examples/plugin_integration/request_feature_vector.json"],
        "includes_descriptors": include_descriptors,
        "public_contract": "CLI and JSON files only; Python internal classes are not stable plugin API.",
    }


def collect_files(include_descriptors: bool) -> list[Path]:
    files = [PROJECT_ROOT / name for name in BUNDLE_FILES if (PROJECT_ROOT / name).exists()]
    if include_descriptors:
        files.extend(sorted((PROJECT_ROOT / "data" / "processed" / "descriptors").glob("*.json")))
    return sorted({path.resolve() for path in files})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = args.archive_name or f"gesture_plugin_bundle_{timestamp}.zip"
    archive_path = args.output_dir / archive_name
    bundle_files = collect_files(args.include_descriptors)

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("plugin_manifest.json", json.dumps(manifest(args.include_descriptors), ensure_ascii=False, indent=2))
        for path in bundle_files:
            archive.write(path, arcname=path.relative_to(PROJECT_ROOT))

    print(f"Plugin bundle created: {archive_path}")
    print(f"Packed files: {len(bundle_files) + 1}")


if __name__ == "__main__":
    main()
