"""
Download HaGRID annotations from the official source and extract the gesture
JSON files needed by phase 1.

By default this script downloads the official annotations archive and extracts
the configured target gestures for train/val/test into:

    data/raw/<split>/<gesture>.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config


DEFAULT_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract HaGRID annotations needed for phase 1."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Official archive URL.")
    parser.add_argument(
        "--save-root",
        type=Path,
        default=config.DATA_RAW_DIR,
        help="Root directory where extracted JSON files should be stored.",
    )
    parser.add_argument(
        "--archive-name",
        default="annotations.zip",
        help="Local filename for the downloaded archive.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Annotation split(s) to extract.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(config.TARGET_GESTURES),
        help="Gesture labels to extract.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the archive even if it already exists locally.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip after extraction.",
    )
    return parser.parse_args(argv)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading annotations archive from:\n  {url}")
    print(f"Saving to:\n  {destination}")
    urllib.request.urlretrieve(url, destination)


def candidate_members(split: str, gesture: str) -> list[str]:
    return [
        f"hagrid_annotations/{split}/{gesture}.json",
        f"annotations/{split}/{gesture}.json",
        f"{split}/{gesture}.json",
    ]


def extract_requested_files(
    archive_path: Path,
    save_root: Path,
    splits: list[str],
    targets: list[str],
) -> list[Path]:
    extracted: list[Path] = []
    with ZipFile(archive_path) as zf:
        members = set(zf.namelist())
        for split in splits:
            for gesture in targets:
                matched_member = None
                for candidate in candidate_members(split, gesture):
                    if candidate in members:
                        matched_member = candidate
                        break
                if matched_member is None:
                    print(f"[skip] missing in archive: {split}/{gesture}.json")
                    continue

                target_path = save_root / split / f"{gesture}.json"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(matched_member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target_path)
                print(f"[ok] extracted {matched_member} -> {target_path}")
    return extracted


def main() -> int:
    args = parse_args()
    save_root = args.save_root.resolve()
    archive_path = save_root / "downloads" / args.archive_name

    if args.force_download or not archive_path.exists():
        download_file(args.url, archive_path)
    else:
        print(f"Using existing archive: {archive_path}")

    extracted = extract_requested_files(
        archive_path=archive_path,
        save_root=save_root,
        splits=args.splits,
        targets=args.targets,
    )

    if not args.keep_archive and archive_path.exists():
        archive_path.unlink()
        print(f"Removed archive: {archive_path}")

    print(f"\nExtracted {len(extracted)} annotation files.")
    if not extracted:
        print("No requested files were extracted.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
