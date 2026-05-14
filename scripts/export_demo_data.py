from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import collect_landmarks_for_splits
from src.demo_profiles import PROFILE_ORDER, build_demo_profiles
from src.labels import GESTURE_LABELS, SCENARIO_DESCRIPTIONS, SCENARIO_LABELS
from src.progress_analyzer import PROGRESSION_COMPONENT_WEIGHTS
from src.progression import DEFAULT_STAGE3_CONFIG, SCENARIO_ORDER, simulate_progression_trace
from src.scorer import ColdStartScorer


CLASSIFIER_PROFILES = [
    {
        "id": "reference",
        "strength": 0.0,
        "wobble": 0.0,
    },
    {
        "id": "mild_offset",
        "strength": 0.72,
        "wobble": 0.04,
    },
    {
        "id": "near_boundary",
        "strength": 1.05,
        "wobble": 0.08,
    },
    {
        "id": "severe_offset",
        "strength": 1.45,
        "wobble": 0.1,
    },
]


COLD_START_PRESETS = [
    {
        "id": "steady_recovery",
        "strengths": [1.05, 0.95, 0.85, 0.78, 0.70, 0.62, 0.55, 0.48, 0.44, 0.40, 0.36, 0.33, 0.31, 0.29, 0.27, 0.25],
        "wobble": 0.05,
    },
    {
        "id": "mild_stabilization",
        "strengths": [0.82, 0.78, 0.75, 0.72, 0.69, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.54, 0.53, 0.52, 0.50, 0.49],
        "wobble": 0.035,
    },
    {
        "id": "severe_start_recovery",
        "strengths": [2.40, 2.32, 2.24, 2.16, 2.08, 2.00, 1.94, 1.88, 1.82, 1.76, 1.70, 1.64, 1.58, 1.52, 1.48, 1.44],
        "wobble": 0.075,
    },
    {
        "id": "fatigue_plateau",
        "strengths": [1.05, 0.95, 0.86, 0.78, 0.70, 0.64, 0.60, 0.58, 0.82, 0.92, 0.86, 0.76, 0.68, 0.63, 0.60, 0.58],
        "wobble": 0.06,
    },
]


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def _feature_stats(descriptor, global_mean: np.ndarray) -> list[dict[str, float | str]]:
    sigma_diag = np.sqrt(np.clip(np.diag(descriptor.sigma), 1e-8, None))
    rows = []
    for name, mean_value, std_value, global_value in zip(
        descriptor.feature_names,
        descriptor.mu,
        sigma_diag,
        global_mean,
    ):
        rows.append(
            {
                "name": name,
                "mean": float(mean_value),
                "std": float(std_value),
                "global_delta": float(mean_value - global_value),
            }
        )
    return rows


def _confusion_matrix(scorer: ColdStartScorer, features_dir: Path) -> tuple[list[str], np.ndarray]:
    gestures = [gesture for gesture in config.TARGET_GESTURES if (features_dir / f"{gesture}.npz").exists()]
    matrix = np.zeros((len(gestures), len(gestures)), dtype=int)
    for row_index, gesture in enumerate(gestures):
        with np.load(features_dir / f"{gesture}.npz") as data:
            test_rows = np.asarray(data["test"], dtype=np.float64)
        for vector in test_rows:
            predicted, _ = scorer.classify(vector)
            col_index = gestures.index(predicted)
            matrix[row_index, col_index] += 1
    return gestures, matrix


def _gesture_direction(
    gesture: str,
    scorer: ColdStartScorer,
) -> tuple[np.ndarray, np.ndarray, str]:
    target = scorer.descriptors[gesture]
    distances = {
        other_gesture: target.mahalanobis(other_descriptor.mu)
        for other_gesture, other_descriptor in scorer.descriptors.items()
        if other_gesture != gesture
    }
    competitor = min(distances, key=distances.get)
    delta = scorer.descriptors[competitor].mu - target.mu
    std = np.sqrt(np.clip(np.diag(target.sigma), 1e-8, None))
    direction = np.sign(delta)
    direction[direction == 0] = 1.0
    return direction, std, competitor


def _wobble_pattern(dimension: int, seed: int) -> np.ndarray:
    index = np.arange(dimension, dtype=np.float64)
    pattern = np.sin((index + 1.0) * (seed + 1.0) * 0.73) + 0.45 * np.cos((index + 1.0) * 0.37)
    max_abs = float(np.max(np.abs(pattern)))
    if max_abs <= 1e-8:
        return np.ones(dimension, dtype=np.float64)
    return pattern / max_abs


def _offset_vector(
    gesture: str,
    scorer: ColdStartScorer,
    strength: float,
    wobble: float,
    step_index: int,
) -> np.ndarray:
    descriptor = scorer.descriptors[gesture]
    direction, std, _ = _gesture_direction(gesture, scorer)
    pattern = _wobble_pattern(len(descriptor.mu), step_index + len(gesture))
    return descriptor.mu + float(strength) * std * direction + float(wobble) * std * pattern


def _classify_payload(scorer: ColdStartScorer, gesture: str) -> list[dict]:
    rows = []
    _, _, competitor = _gesture_direction(gesture, scorer)
    for index, profile in enumerate(CLASSIFIER_PROFILES):
        vector = _offset_vector(
            gesture,
            scorer,
            strength=float(profile["strength"]),
            wobble=float(profile["wobble"]),
            step_index=index,
        )
        predicted, distances = scorer.classify(vector)
        target_distance = float(distances[gesture])
        sorted_distances = sorted((key, float(value)) for key, value in distances.items())
        top_distances = sorted(sorted_distances, key=lambda item: item[1])[:5]
        second_distance = top_distances[1][1] if len(top_distances) > 1 else target_distance
        rows.append(
            {
                "id": profile["id"],
                "strength": float(profile["strength"]),
                "competitor_gesture": competitor,
                "predicted_gesture": predicted,
                "target_distance": target_distance,
                "confidence_margin": float(second_distance - top_distances[0][1]),
                "top_distances": [
                    {
                        "gesture": candidate,
                        "distance": distance,
                    }
                    for candidate, distance in top_distances
                ],
                "vector": vector.tolist(),
            }
        )
    return rows


def _normalize_skeleton(points: np.ndarray) -> list[list[float]]:
    pts = np.asarray(points, dtype=np.float64).copy()
    if pts.shape != (config.N_LANDMARKS, 2):
        return []
    pts -= pts[config.WRIST]
    scale = np.linalg.norm(pts[config.SCALE_REF_TO] - pts[config.SCALE_REF_FROM])
    if scale > 1e-8:
        pts /= scale
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-8)
    normalized = (pts - min_xy) / span
    normalized[:, 1] = 1.0 - normalized[:, 1]
    return normalized.tolist()


def _mean_skeleton_payload(raw_dir: Path, gesture: str) -> dict:
    if gesture == "palm":
        return {
            "standard": _canonical_palm_skeleton(),
            "samples": 0,
            "source": "canonical_demo_template",
        }
    landmarks, _ = collect_landmarks_for_splits(
        raw_dir,
        gesture,
        splits=("train", "val"),
        max_samples=240,
    )
    if len(landmarks) == 0:
        return {
            "standard": [],
            "samples": 0,
        }
    mean_landmarks = np.mean(np.asarray(landmarks, dtype=np.float64), axis=0)
    return {
        "standard": _normalize_skeleton(mean_landmarks),
        "samples": int(len(landmarks)),
        "source": "hagrid_train_val_mean",
    }


def _canonical_palm_skeleton() -> list[list[float]]:
    """Palm target for the demo: wrist low, five fingers naturally upright."""
    return [
        [0.50, 0.94],
        [0.43, 0.82],
        [0.36, 0.70],
        [0.30, 0.59],
        [0.24, 0.50],
        [0.40, 0.61],
        [0.39, 0.42],
        [0.39, 0.24],
        [0.39, 0.07],
        [0.50, 0.60],
        [0.50, 0.39],
        [0.50, 0.20],
        [0.50, 0.04],
        [0.60, 0.61],
        [0.61, 0.42],
        [0.61, 0.24],
        [0.61, 0.08],
        [0.70, 0.64],
        [0.72, 0.48],
        [0.73, 0.32],
        [0.74, 0.16],
    ]


def build_demo_payload(descriptor_dir: Path, features_dir: Path) -> dict:
    scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir)
    gestures = list(scorer.descriptors.keys())
    global_mean = np.mean(np.vstack([descriptor.mu for descriptor in scorer.descriptors.values()]), axis=0)
    confusion_gestures, confusion = _confusion_matrix(scorer, features_dir)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))

    recalls = {}
    per_gesture_samples = {}
    for row_index, gesture in enumerate(confusion_gestures):
        row_total = int(confusion[row_index].sum())
        recalls[gesture] = round((confusion[row_index, row_index] / max(row_total, 1)) * 100.0, 1)
        with np.load(features_dir / f"{gesture}.npz") as data:
            per_gesture_samples[gesture] = {
                "train": int(len(data["train"])),
                "test": int(len(data["test"])),
            }

    descriptor_summary = {}
    sample_scores = {}
    profile_catalog = {}
    classifier_samples = {}
    gesture_skeletons = {}

    for gesture in gestures:
        descriptor = scorer.descriptors[gesture]
        feature_stats = _feature_stats(descriptor, global_mean)
        top_mean_features = sorted(feature_stats, key=lambda row: abs(float(row["global_delta"])), reverse=True)[:5]
        top_low_variance = sorted(feature_stats, key=lambda row: float(row["std"]))[:5]

        descriptor_summary[gesture] = {
            "n_samples": int(descriptor.n_samples),
            "top_mean_features": top_mean_features,
            "top_low_variance_features": top_low_variance,
            "all_features": feature_stats,
        }
        classifier_samples[gesture] = _classify_payload(scorer, gesture)
        gesture_skeletons[gesture] = _mean_skeleton_payload(config.DATA_RAW_DIR, gesture)

        profiles = build_demo_profiles(gesture, scorer.descriptors)
        sample_scores[gesture] = {}
        for profile_name in PROFILE_ORDER:
            profile = profiles[profile_name]
            profile_catalog[profile_name] = {
                "label": profile.label,
                "description": profile.description,
                "kind": profile.kind,
            }

            profile_scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir)
            results = []
            for vector in profile.vectors:
                result = profile_scorer.score(vector, gesture, update_user_model=True)
                results.append(result.to_dict())
            sample_scores[gesture][profile_name] = {
                "competitor_gesture": profile.competitor_gesture,
                "results": results,
            }

    return {
        "phase1_summary": {
            "gestures": gestures,
            "overall_accuracy": round((correct / max(total, 1)) * 100.0, 1),
            "total_test_samples": total,
            "per_gesture_samples": per_gesture_samples,
            "recalls": recalls,
        },
        "gesture_descriptors_summary": descriptor_summary,
        "confusion_matrix": {
            "gestures": confusion_gestures,
            "matrix": confusion.tolist(),
        },
        "sample_profiles": profile_catalog,
        "sample_scores": sample_scores,
        "classifier_samples": classifier_samples,
        "cold_start_presets": COLD_START_PRESETS,
        "gesture_skeletons": gesture_skeletons,
    }


def build_runtime_payload(descriptor_dir: Path, features_dir: Path) -> dict:
    scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir)
    default_traces = {
        scenario: simulate_progression_trace("fist", scenario).to_dict()
        for scenario in SCENARIO_ORDER
    }
    descriptors = {}
    for gesture, descriptor in scorer.descriptors.items():
        descriptors[gesture] = {
            "gesture": gesture,
            "label": GESTURE_LABELS.get(gesture, gesture),
            "n_samples": int(descriptor.n_samples),
            "mu": descriptor.mu.tolist(),
            "sigma": descriptor.sigma.tolist(),
            "sigma_inv": descriptor.sigma_inv.tolist(),
            "sigma_diag": np.sqrt(np.clip(np.diag(descriptor.sigma), 1e-8, None)).tolist(),
            "feature_names": list(descriptor.feature_names),
            "expected_distance": float(scorer.expected_distances.get(gesture, 1.0)),
        }

    joint_triplets = [
        {"name": name, "points": list(points)}
        for name, points in config.JOINT_TRIPLETS.items()
    ]
    tip_pairs = [
        {"name": f"dist_{name}", "points": list(points)}
        for name, points in config.TIP_PAIRS.items()
    ]
    spread_triplets = [
        {"name": name, "points": list(points)}
        for name, points in config.SPREAD_TRIPLETS.items()
    ]

    return {
        "gesture_labels": GESTURE_LABELS,
        "landmark_config": {
            "wrist": config.WRIST,
            "scale_ref_from": config.SCALE_REF_FROM,
            "scale_ref_to": config.SCALE_REF_TO,
            "connections": [list(pair) for pair in HAND_CONNECTIONS],
            "joint_triplets": joint_triplets,
            "tip_pairs": tip_pairs,
            "spread_triplets": spread_triplets,
        },
        "camera_classifier": {
            "num_hands": 1,
            "min_hand_detection_confidence": 0.55,
            "min_hand_presence_confidence": 0.55,
            "min_tracking_confidence": 0.55,
            "vision_module_url": "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm",
            "model_asset_url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "vision_wasm_url": "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
        },
        "stage2": {
            "tau0": 5.0,
            "gamma": 0.6,
            "adaptive_window": 10,
            "n_transition": 30,
            "lambda_max": 0.8,
            "descriptors": descriptors,
        },
        "stage3": {
            "defaults": DEFAULT_STAGE3_CONFIG,
            "scenarios": {
                scenario: {
                    "label": SCENARIO_LABELS[scenario],
                    "description": SCENARIO_DESCRIPTIONS[scenario],
                    "default_trace": default_traces[scenario],
                }
                for scenario in SCENARIO_ORDER
            },
        },
        "cold_start_demo": {
            "round_size": 4,
            "challenge_low": DEFAULT_STAGE3_CONFIG["challenge_low"],
            "challenge_high": DEFAULT_STAGE3_CONFIG["challenge_high"],
            "tau_init": DEFAULT_STAGE3_CONFIG["tau_init"],
            "tau_min": DEFAULT_STAGE3_CONFIG["tau_min"],
            "tau_max": DEFAULT_STAGE3_CONFIG["tau_max"],
            "presets": {
                preset["id"]: {
                    "attempts": len(preset["strengths"]),
                    "round_size": 4,
                }
                for preset in COLD_START_PRESETS
            },
        },
        "inverse_progress": {
            "component_weights": PROGRESSION_COMPONENT_WEIGHTS,
        },
    }


def main() -> None:
    descriptor_dir = config.DATA_PROCESSED_DIR / "descriptors"
    features_dir = config.DATA_PROCESSED_DIR / "features"
    demo_output = PROJECT_ROOT / "web_demo" / "demo_data.json"
    runtime_output = PROJECT_ROOT / "web_demo" / "runtime_config.json"

    demo_payload = build_demo_payload(descriptor_dir, features_dir)
    runtime_payload = build_runtime_payload(descriptor_dir, features_dir)

    demo_output.parent.mkdir(parents=True, exist_ok=True)
    with open(demo_output, "w", encoding="utf-8") as handle:
        json.dump(demo_payload, handle, ensure_ascii=False, indent=2)
    with open(runtime_output, "w", encoding="utf-8") as handle:
        json.dump(runtime_payload, handle, ensure_ascii=False, indent=2)

    print(f"Demo data written to: {demo_output}")
    print(f"Runtime config written to: {runtime_output}")



if __name__ == "__main__":
    main()
