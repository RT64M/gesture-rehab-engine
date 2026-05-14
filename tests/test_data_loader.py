import json
from pathlib import Path
import shutil

from src import config
from src.data_loader import (
    collect_landmarks_for_gesture,
    collect_landmarks_for_splits,
    find_annotation_file,
    find_annotation_files,
    iter_gesture_landmarks,
)


def make_valid_landmarks(offset: float = 0.0) -> list[list[float]]:
    return [[offset + i * 0.01, offset + i * 0.01 + 0.005] for i in range(config.N_LANDMARKS)]


def write_annotation_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def make_local_temp_dir(name: str) -> Path:
    root = Path.cwd() / ".tmp-tests"
    root.mkdir(exist_ok=True)
    path = root / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_iter_gesture_landmarks_filters_invalid_entries():
    temp_dir = make_local_temp_dir("loader-filter")
    payload = {
        "sample_ok": {
            "labels": ["fist"],
            "landmarks": [make_valid_landmarks()],
        },
        "sample_wrong_label": {
            "labels": ["palm"],
            "landmarks": [make_valid_landmarks(0.1)],
        },
        "sample_zeroed": {
            "labels": ["fist"],
            "landmarks": [[[0.0, 0.0] for _ in range(config.N_LANDMARKS)]],
        },
        "sample_bad_shape": {
            "labels": ["fist"],
            "landmarks": [[[0.1, 0.2] for _ in range(10)]],
        },
    }
    json_path = temp_dir / "fist.json"
    try:
        write_annotation_file(json_path, payload)

        rows = list(iter_gesture_landmarks(json_path, target_label="fist"))

        assert len(rows) == 1
        sample_id, landmarks = rows[0]
        assert sample_id == "sample_ok"
        assert landmarks.shape == (config.N_LANDMARKS, 2)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_collect_landmarks_respects_max_samples():
    temp_dir = make_local_temp_dir("loader-collect")
    payload = {
        f"sample_{i}": {
            "labels": ["ok"],
            "landmarks": [make_valid_landmarks(i * 0.01)],
        }
        for i in range(5)
    }
    json_path = temp_dir / "ok.json"
    try:
        write_annotation_file(json_path, payload)

        arr = collect_landmarks_for_gesture(json_path, target_label="ok", max_samples=3)

        assert arr.shape == (3, config.N_LANDMARKS, 2)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_iter_gesture_landmarks_supports_hand_landmarks_alias():
    temp_dir = make_local_temp_dir("loader-hand-landmarks")
    payload = {
        "sample_ok": {
            "labels": ["rock"],
            "hand_landmarks": [make_valid_landmarks(0.2)],
        }
    }
    json_path = temp_dir / "rock.json"
    try:
        write_annotation_file(json_path, payload)

        rows = list(iter_gesture_landmarks(json_path, target_label="rock"))

        assert len(rows) == 1
        _, landmarks = rows[0]
        assert landmarks.shape == (config.N_LANDMARKS, 2)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_find_annotation_file_prefers_known_locations():
    temp_dir = make_local_temp_dir("loader-find-known")
    raw_dir = temp_dir / "raw"

    ann_subsample = raw_dir / "ann_subsample" / "fist.json"
    nested = raw_dir / "nested" / "fist.json"
    root = raw_dir / "palm.json"

    try:
        write_annotation_file(ann_subsample, {})
        write_annotation_file(nested, {})
        write_annotation_file(root, {})

        assert find_annotation_file(raw_dir, "fist") == ann_subsample
        assert find_annotation_file(raw_dir, "palm") == root
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_find_annotation_file_falls_back_to_recursive_search():
    temp_dir = make_local_temp_dir("loader-find-fallback")
    raw_dir = temp_dir / "raw"
    fallback = raw_dir / "deep" / "tree" / "peace.json"
    try:
        write_annotation_file(fallback, {})

        assert find_annotation_file(raw_dir, "peace") == fallback
        assert find_annotation_file(raw_dir, "missing") is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_find_annotation_file_supports_hagrid_annotations_layout():
    temp_dir = make_local_temp_dir("loader-find-hagrid-layout")
    raw_dir = temp_dir / "raw"
    path = raw_dir / "hagrid_annotations" / "train" / "fist.json"
    try:
        write_annotation_file(path, {})
        assert find_annotation_file(raw_dir, "fist") == path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_find_annotation_files_returns_multiple_requested_splits():
    temp_dir = make_local_temp_dir("loader-find-splits")
    raw_dir = temp_dir / "raw"
    train_path = raw_dir / "train" / "fist.json"
    val_path = raw_dir / "val" / "fist.json"
    test_path = raw_dir / "test" / "fist.json"
    try:
        write_annotation_file(train_path, {})
        write_annotation_file(val_path, {})
        write_annotation_file(test_path, {})

        matches = find_annotation_files(raw_dir, "fist", splits=["train", "val", "test"])

        assert matches == [train_path, val_path, test_path]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_collect_landmarks_for_splits_aggregates_multiple_files():
    temp_dir = make_local_temp_dir("loader-collect-splits")
    raw_dir = temp_dir / "raw"
    train_payload = {
        "sample_train": {
            "labels": ["ok"],
            "landmarks": [make_valid_landmarks(0.0)],
        }
    }
    val_payload = {
        "sample_val": {
            "labels": ["ok"],
            "hand_landmarks": [make_valid_landmarks(0.2)],
        }
    }
    try:
        write_annotation_file(raw_dir / "train" / "ok.json", train_payload)
        write_annotation_file(raw_dir / "val" / "ok.json", val_payload)

        rows, paths = collect_landmarks_for_splits(raw_dir, "ok", splits=["train", "val"])

        assert rows.shape == (2, config.N_LANDMARKS, 2)
        assert len(paths) == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
