from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INCLUDE_DIRS = [
    "src",
    "scripts",
    "tests",
    "experiments",
    "web_demo",
    "notebooks",
]

DEFAULT_INCLUDE_FILES = [
    "README.md",
    "PROJECT_PLAN.md",
    "requirements.txt",
]

DEFAULT_EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    ".uv-cache",
    ".pytest_cache",
    ".tmp-tests",
    "__pycache__",
    "artifacts",
    "downloads",
    "reports",
}

DEFAULT_EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
}

DEFAULT_EXCLUDE_PATH_FRAGMENTS = [
    "data/raw/",
    "data/processed/features/",
    "data/processed/plots/",
    "web_demo/models/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package non-data project code into a zip archive while retaining descriptor artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "exports",
        help="Directory where the zip archive will be written.",
    )
    parser.add_argument(
        "--archive-name",
        type=str,
        default=None,
        help="Optional archive filename. Defaults to code_snapshot_<timestamp>.zip",
    )
    parser.add_argument(
        "--no-descriptors",
        action="store_true",
        help="Exclude data/processed/descriptors/*.json from the archive.",
    )
    return parser.parse_args()


def should_skip(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT).as_posix()

    for part in path.parts:
        if part in DEFAULT_EXCLUDE_DIR_NAMES:
            return True

    if path.suffix.lower() in DEFAULT_EXCLUDE_SUFFIXES:
        return True

    for fragment in DEFAULT_EXCLUDE_PATH_FRAGMENTS:
        if relative.startswith(fragment):
            return True

    return False


def iter_files_under(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and not should_skip(path):
            files.append(path)
    return files


def build_file_list(include_descriptors: bool) -> list[Path]:
    files: list[Path] = []

    for name in DEFAULT_INCLUDE_DIRS:
        files.extend(iter_files_under(PROJECT_ROOT / name))

    for name in DEFAULT_INCLUDE_FILES:
        path = PROJECT_ROOT / name
        if path.exists() and path.is_file():
            files.append(path)

    if include_descriptors:
        descriptor_dir = PROJECT_ROOT / "data" / "processed" / "descriptors"
        if descriptor_dir.exists():
            files.extend(sorted(descriptor_dir.glob("*.json")))

    deduped = sorted({path.resolve() for path in files})
    return [Path(path) for path in deduped]


def create_archive(output_path: Path, files: list[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(PROJECT_ROOT))


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = args.archive_name or f"code_snapshot_{timestamp}.zip"
    output_path = args.output_dir / archive_name

    files = build_file_list(include_descriptors=not args.no_descriptors)
    create_archive(output_path, files)

    print(f"Archive created: {output_path}")
    print(f"Packed files: {len(files)}")
    if not args.no_descriptors:
        print("Included descriptor artifacts: data/processed/descriptors/*.json")
    else:
        print("Descriptor artifacts were excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
