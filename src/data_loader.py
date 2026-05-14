"""HaGRIDv2 annotation loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from . import config
from .geometry import is_valid_landmarks


KNOWN_SPLITS = ("train", "val", "test")


def load_gesture_annotations(json_path: Path) -> dict:
    """Load one gesture annotation JSON file."""
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_landmarks_list(annotation: dict) -> list:
    """Support both legacy and current HaGRID landmark field names."""
    for key in ("landmarks", "hand_landmarks"):
        value = annotation.get(key)
        if value:
            return value
    return []


def iter_gesture_landmarks(
    json_path: Path,
    target_label: str,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield valid 21x2 landmarks for the requested gesture label."""
    data = load_gesture_annotations(json_path)
    for sample_id, annotation in data.items():
        labels = annotation.get("labels", [])
        landmarks_list = _get_landmarks_list(annotation)
        for label, landmarks in zip(labels, landmarks_list):
            if label != target_label or not landmarks:
                continue
            array = np.asarray(landmarks, dtype=np.float64)
            if array.shape != (config.N_LANDMARKS, 2):
                continue
            if not is_valid_landmarks(array):
                continue
            yield sample_id, array
            break


def collect_landmarks_for_gesture(
    json_path: Path,
    target_label: str,
    max_samples: int | None = None,
) -> np.ndarray:
    """Collect valid landmarks from one JSON file into an (N, 21, 2) array."""
    samples: list[np.ndarray] = []
    for _, landmarks in iter_gesture_landmarks(json_path, target_label):
        samples.append(landmarks)
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        return np.empty((0, config.N_LANDMARKS, 2), dtype=np.float64)
    return np.stack(samples, axis=0)


def collect_landmarks_from_files(
    json_paths: list[Path],
    target_label: str,
    max_samples: int | None = None,
) -> np.ndarray:
    """Collect valid landmarks across multiple annotation JSON files."""
    samples: list[np.ndarray] = []
    for json_path in json_paths:
        for _, landmarks in iter_gesture_landmarks(json_path, target_label):
            samples.append(landmarks)
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break
    if not samples:
        return np.empty((0, config.N_LANDMARKS, 2), dtype=np.float64)
    return np.stack(samples, axis=0)


def _candidate_paths_for_split(raw_dir: Path, gesture: str, split: str) -> list[Path]:
    return [
        raw_dir / split / f"{gesture}.json",
        raw_dir / "hagrid_annotations" / split / f"{gesture}.json",
        raw_dir / "annotations" / split / f"{gesture}.json",
    ]


def find_annotation_file(raw_dir: Path, gesture: str) -> Path | None:
    """Find one annotation file for a gesture using legacy search behavior."""
    candidates = [
        raw_dir / "ann_subsample" / f"{gesture}.json",
        raw_dir / f"{gesture}.json",
        raw_dir / "train" / f"{gesture}.json",
        raw_dir / "annotations" / f"{gesture}.json",
        raw_dir / "hagrid_annotations" / "train" / f"{gesture}.json",
        raw_dir / "hagrid_annotations" / "val" / f"{gesture}.json",
        raw_dir / "hagrid_annotations" / "test" / f"{gesture}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(raw_dir.rglob(f"{gesture}.json"))
    return matches[0] if matches else None


def find_annotation_files(
    raw_dir: Path,
    gesture: str,
    splits: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Find all annotation files for a gesture across the requested splits."""
    requested_splits = list(KNOWN_SPLITS if splits is None else splits)
    matches: list[Path] = []
    seen: set[Path] = set()

    for split in requested_splits:
        for candidate in _candidate_paths_for_split(raw_dir, gesture, split):
            if candidate.exists() and candidate not in seen:
                matches.append(candidate)
                seen.add(candidate)

    if matches:
        return matches

    if splits is not None:
        return []

    fallback_matches = sorted(raw_dir.rglob(f"{gesture}.json"))
    return [path for path in fallback_matches if path not in seen]


def collect_landmarks_for_splits(
    raw_dir: Path,
    gesture: str,
    splits: list[str] | tuple[str, ...],
    max_samples: int | None = None,
) -> tuple[np.ndarray, list[Path]]:
    """Collect valid landmarks for one gesture across the requested splits."""
    paths = find_annotation_files(raw_dir, gesture, splits=splits)
    if not paths:
        return np.empty((0, config.N_LANDMARKS, 2), dtype=np.float64), []
    landmarks = collect_landmarks_from_files(paths, target_label=gesture, max_samples=max_samples)
    return landmarks, paths
