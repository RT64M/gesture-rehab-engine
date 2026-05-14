"""Build gesture descriptors from HaGRIDv2 annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import collect_landmarks_for_splits, find_annotation_files
from src.descriptors import build_descriptor, classify
from src.features import batch_extract_geometric, feature_names
from src.labels import gesture_label


DEFAULT_TRAIN_SPLITS = ("train", "val")
DEFAULT_TEST_SPLITS = ("test",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target gesture descriptors from HaGRIDv2 splits.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=config.DATA_RAW_DIR,
        help="Root directory containing raw annotation JSON files.",
    )
    parser.add_argument(
        "--descriptor-dir",
        type=Path,
        default=config.DATA_PROCESSED_DIR / "descriptors",
        help="Output directory for descriptor JSON files.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=config.DATA_PROCESSED_DIR / "features",
        help="Output directory for feature NPZ files.",
    )
    parser.add_argument(
        "--max-samples-per-gesture",
        type=int,
        default=None,
        help="Optional cap for debugging. Default uses all available samples.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fallback random split ratio used only when official test data is missing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fallback splitting.",
    )
    return parser.parse_args(argv)


def _split_paths(raw_dir: Path, gesture: str, split: str) -> list[Path]:
    return find_annotation_files(raw_dir, gesture, splits=[split])


def _split_landmarks(
    raw_dir: Path,
    gesture: str,
    split: str,
    max_samples_per_gesture: int | None,
) -> tuple[np.ndarray, list[Path]]:
    return collect_landmarks_for_splits(
        raw_dir=raw_dir,
        gesture=gesture,
        splits=[split],
        max_samples=max_samples_per_gesture,
    )


def _safe_stack(chunks: list[np.ndarray]) -> np.ndarray:
    non_empty = [chunk for chunk in chunks if len(chunk)]
    if not non_empty:
        return np.empty((0, config.N_LANDMARKS, 2), dtype=np.float64)
    return np.concatenate(non_empty, axis=0)


def _fallback_train_test_split(
    features: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < 12:
        raise ValueError("Need at least 12 samples to perform fallback random splitting")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    n_test = max(1, int(round(len(features) * test_ratio)))
    n_test = min(n_test, len(features) - 10)
    if n_test <= 0:
        raise ValueError("Not enough samples left for descriptor fitting after fallback split")

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return features[train_idx], features[test_idx]


def build_descriptors_from_raw(
    raw_dir: Path,
    descriptor_dir: Path,
    features_dir: Path,
    max_samples_per_gesture: int | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    fnames = feature_names()
    descriptors: dict[str, object] = {}
    train_features: dict[str, np.ndarray] = {}
    test_features: dict[str, np.ndarray] = {}
    split_counts: dict[str, dict[str, int | bool]] = {}
    global_train_landmarks: dict[str, np.ndarray] = {}

    print("=" * 72)
    print(f"HaGRIDv2 {len(config.TARGET_GESTURES)} 类目标手势描述符构建")
    print("=" * 72)
    print(f"训练口径: {' + '.join(DEFAULT_TRAIN_SPLITS)}")
    print(f"评估口径: {' + '.join(DEFAULT_TEST_SPLITS)}")
    print(f"每类样本上限: {'全量' if max_samples_per_gesture is None else max_samples_per_gesture}")
    print("")

    for gesture in config.TARGET_GESTURES:
        train_landmarks, train_paths = _split_landmarks(raw_dir, gesture, "train", max_samples_per_gesture)
        val_landmarks, val_paths = _split_landmarks(raw_dir, gesture, "val", max_samples_per_gesture)
        test_landmarks, test_paths = _split_landmarks(raw_dir, gesture, "test", max_samples_per_gesture)

        combined_train_landmarks = _safe_stack([train_landmarks, val_landmarks])

        print(f"[{gesture_label(gesture):>4}]")
        print(f"  train paths: {len(train_paths)}  valid samples: {len(train_landmarks)}")
        print(f"  val paths:   {len(val_paths)}  valid samples: {len(val_landmarks)}")
        print(f"  test paths:  {len(test_paths)}  valid samples: {len(test_landmarks)}")
        print(f"  merged train+val samples: {len(combined_train_landmarks)}")

        if len(combined_train_landmarks) < 10:
            print("  [skip] insufficient train+val samples for descriptor fitting")
            continue

        train_pool_features = batch_extract_geometric(combined_train_landmarks, apply_pca_alignment=True)
        global_train_landmarks[gesture] = combined_train_landmarks

        used_official_test = len(test_landmarks) > 0
        if used_official_test:
            effective_train_features = train_pool_features
            effective_test_features = batch_extract_geometric(test_landmarks, apply_pca_alignment=True)
        else:
            effective_train_features, effective_test_features = _fallback_train_test_split(
                train_pool_features,
                test_ratio=test_ratio,
                seed=seed,
            )
            print(f"  [fallback] missing official test split, random split with test_ratio={test_ratio}")

        train_features[gesture] = effective_train_features
        test_features[gesture] = effective_test_features
        split_counts[gesture] = {
            "train_count": int(len(train_landmarks)),
            "val_count": int(len(val_landmarks)),
            "test_count": int(len(test_landmarks)),
            "train_val_count": int(len(combined_train_landmarks)),
            "used_official_test": bool(used_official_test),
        }

        np.savez(
            features_dir / f"{gesture}.npz",
            train=effective_train_features,
            test=effective_test_features,
            feature_names=np.asarray(fnames),
            train_count=np.asarray(int(len(train_landmarks))),
            val_count=np.asarray(int(len(val_landmarks))),
            test_count=np.asarray(int(len(test_landmarks))),
            train_val_count=np.asarray(int(len(combined_train_landmarks))),
            used_official_test=np.asarray(int(used_official_test)),
        )

        descriptor = build_descriptor(gesture, effective_train_features, fnames)
        descriptors[gesture] = descriptor
        descriptor.save_json(descriptor_dir / f"{gesture}.json")

        print(f"  final train features: {len(effective_train_features)}")
        print(f"  final test features:  {len(effective_test_features)}")
        print("")

    if not descriptors:
        raise RuntimeError("No gesture descriptors could be built from the available raw data")

    gestures = list(descriptors.keys())
    confusion = np.zeros((len(gestures), len(gestures)), dtype=int)
    total = 0
    total_correct = 0

    for true_gesture, features in test_features.items():
        true_index = gestures.index(true_gesture)
        for vector in features:
            predicted, _ = classify(vector, descriptors)
            predicted_index = gestures.index(predicted)
            confusion[true_index, predicted_index] += 1
            total += 1
            total_correct += int(predicted == true_gesture)

    print("--- 官方 test 评估 ---")
    header = "true\\pred " + " ".join(f"{gesture:>6}" for gesture in gestures)
    print(header)
    for row_index, gesture in enumerate(gestures):
        row = " ".join(f"{confusion[row_index, col]:>6d}" for col in range(len(gestures)))
        recall = confusion[row_index, row_index] / max(confusion[row_index].sum(), 1) * 100.0
        print(f"{gesture:>8} {row}    recall={recall:.1f}%")

    overall_accuracy = total_correct / max(total, 1) * 100.0
    print(f"\nOverall accuracy: {total_correct}/{total} = {overall_accuracy:.2f}%")
    print(f"Descriptors saved to: {descriptor_dir}")
    print(f"Feature matrices saved to: {features_dir}")

    return {
        "gestures": gestures,
        "descriptors": descriptors,
        "train_features": train_features,
        "test_features": test_features,
        "split_counts": split_counts,
        "confusion_matrix": confusion,
        "overall_accuracy": overall_accuracy,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_descriptors_from_raw(
        raw_dir=args.raw_dir,
        descriptor_dir=args.descriptor_dir,
        features_dir=args.features_dir,
        max_samples_per_gesture=args.max_samples_per_gesture,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
