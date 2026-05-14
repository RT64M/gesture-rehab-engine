import json
from pathlib import Path
import shutil

import numpy as np

from scripts.build_descriptors import build_descriptors_from_raw
from scripts.download_hagrid_annotations import parse_args as parse_download_args
from src import config


def make_valid_landmarks(offset: float = 0.0) -> list[list[float]]:
    return [[offset + i * 0.01, offset + i * 0.01 + 0.005] for i in range(config.N_LANDMARKS)]


def write_annotation_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def make_local_temp_dir(name: str) -> Path:
    root = Path.cwd() / ".tmp-tests"
    root.mkdir(exist_ok=True)
    path = root / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def build_payload(gesture: str, count: int, base_offset: float) -> dict:
    return {
        f"{gesture}_{index}": {
            "labels": [gesture],
            "hand_landmarks": [make_valid_landmarks(base_offset + index * 0.001)],
        }
        for index in range(count)
    }


def test_download_parse_args_defaults_to_all_official_splits():
    args = parse_download_args([])
    assert args.splits == ["train", "val", "test"]
    assert args.targets == list(config.TARGET_GESTURES)


def test_build_descriptors_uses_official_test_when_present():
    temp_dir = make_local_temp_dir("build-official-test")
    raw_dir = temp_dir / "raw"
    descriptor_dir = temp_dir / "processed" / "descriptors"
    features_dir = temp_dir / "processed" / "features"
    try:
        write_annotation_file(raw_dir / "train" / "fist.json", build_payload("fist", 12, 0.00))
        write_annotation_file(raw_dir / "val" / "fist.json", build_payload("fist", 6, 0.20))
        write_annotation_file(raw_dir / "test" / "fist.json", build_payload("fist", 5, 0.40))

        summary = build_descriptors_from_raw(raw_dir, descriptor_dir, features_dir, seed=7)

        assert summary["split_counts"]["fist"]["used_official_test"] is True
        assert len(summary["train_features"]["fist"]) == 18
        assert len(summary["test_features"]["fist"]) == 5

        with np.load(features_dir / "fist.npz") as data:
            assert len(data["train"]) == 18
            assert len(data["test"]) == 5
            assert int(data["train_count"]) == 12
            assert int(data["val_count"]) == 6
            assert int(data["test_count"]) == 5
            assert int(data["used_official_test"]) == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_descriptors_falls_back_to_random_split_when_test_missing():
    temp_dir = make_local_temp_dir("build-fallback-test")
    raw_dir = temp_dir / "raw"
    descriptor_dir = temp_dir / "processed" / "descriptors"
    features_dir = temp_dir / "processed" / "features"
    try:
        write_annotation_file(raw_dir / "train" / "peace.json", build_payload("peace", 12, 0.00))
        write_annotation_file(raw_dir / "val" / "peace.json", build_payload("peace", 8, 0.20))

        summary = build_descriptors_from_raw(raw_dir, descriptor_dir, features_dir, test_ratio=0.25, seed=11)

        assert summary["split_counts"]["peace"]["used_official_test"] is False
        train_len = len(summary["train_features"]["peace"])
        test_len = len(summary["test_features"]["peace"])
        assert train_len + test_len == 20
        assert test_len == 5

        with np.load(features_dir / "peace.npz") as data:
            assert len(data["train"]) == train_len
            assert len(data["test"]) == test_len
            assert int(data["test_count"]) == 0
            assert int(data["used_official_test"]) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
