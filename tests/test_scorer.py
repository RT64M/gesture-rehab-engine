from pathlib import Path
import json
import shutil

import numpy as np

from src.demo_profiles import build_demo_profiles
from src.descriptors import build_descriptor
from src.feedback import generate_feedback, summarize_deviations
from src.scorer import ColdStartScorer


def make_cluster(center: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.05, size=(n_samples, len(center)))
    return center + noise


def make_local_temp_dir(name: str) -> Path:
    root = Path.cwd() / ".tmp-tests"
    root.mkdir(exist_ok=True)
    path = root / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def build_artifacts(temp_dir: Path) -> tuple[Path, Path]:
    descriptor_dir = temp_dir / "descriptors"
    features_dir = temp_dir / "features"
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [f"f{i}" for i in range(4)]
    fist = make_cluster(np.array([0.0, 0.0, 0.0, 0.0]), n_samples=80, seed=1)
    palm = make_cluster(np.array([2.0, 2.0, 2.0, 2.0]), n_samples=80, seed=2)

    for gesture, cluster in {"fist": fist, "palm": palm}.items():
        train = cluster[:60]
        test = cluster[60:]
        descriptor = build_descriptor(gesture, train, feature_names)
        descriptor.save_json(descriptor_dir / f"{gesture}.json")
        np.savez(features_dir / f"{gesture}.npz", train=train, test=test, feature_names=np.array(feature_names))
    return descriptor_dir, features_dir


def test_score_decreases_as_sample_moves_away():
    temp_dir = make_local_temp_dir("scorer-monotonic")
    try:
        descriptor_dir, features_dir = build_artifacts(temp_dir)
        scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir)

        base = scorer.descriptors["fist"].mu
        near = base + np.array([0.05, 0.05, 0.05, 0.05])
        far = base + np.array([1.0, 1.0, 1.0, 1.0])

        near_result = scorer.score(near, "fist", update_user_model=False)
        far_result = scorer.score(far, "fist", update_user_model=False)

        assert near_result.mahalanobis_distance < far_result.mahalanobis_distance
        assert near_result.raw_score > far_result.raw_score
        assert near_result.display_score > far_result.display_score
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_predicted_gesture_matches_smallest_population_distance():
    temp_dir = make_local_temp_dir("scorer-predict")
    try:
        descriptor_dir, features_dir = build_artifacts(temp_dir)
        scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir)

        probe = scorer.descriptors["palm"].mu + np.array([0.05, -0.02, 0.03, 0.01])
        result = scorer.score(probe, "fist", update_user_model=False)

        assert result.predicted_gesture == "palm"
        assert result.population_distances["palm"] < result.population_distances["fist"]
        assert not result.matched
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_user_model_blend_weight_grows_and_caps():
    temp_dir = make_local_temp_dir("scorer-user-model")
    try:
        descriptor_dir, features_dir = build_artifacts(temp_dir)
        scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir, n_transition=5, lambda_max=0.6)
        model = scorer.get_user_model("fist")

        assert model.blend_weight == 0.0
        for _ in range(10):
            scorer.score(scorer.descriptors["fist"].mu, "fist", update_user_model=True)

        assert model.blend_weight == 0.6
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_feedback_returns_top_deviations_and_mismatch_hint():
    temp_dir = make_local_temp_dir("scorer-feedback")
    try:
        descriptor_dir, features_dir = build_artifacts(temp_dir)
        scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir)
        descriptor = scorer.descriptors["fist"]
        sample = descriptor.mu + np.array([0.8, 0.2, -0.6, 0.1])

        deviations = summarize_deviations(sample, descriptor, top_k=3)
        feedback = generate_feedback(sample, descriptor, target_gesture="fist", predicted_gesture="palm", top_k=3)

        assert len(deviations) == 3
        assert deviations[0]["magnitude"] >= deviations[1]["magnitude"] >= deviations[2]["magnitude"]
        assert feedback[0].startswith("当前更像")
        assert len(feedback) == 4
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_demo_profiles_produce_monotonic_scores():
    temp_dir = make_local_temp_dir("scorer-profiles")
    try:
        descriptor_dir, features_dir = build_artifacts(temp_dir)
        scorer = ColdStartScorer.from_artifacts(descriptor_dir, features_dir)
        profiles = build_demo_profiles("fist", scorer.descriptors)

        standard = scorer.score(profiles["standard"].vectors[0], "fist", update_user_model=False)
        mild = scorer.score(profiles["mild_deviation"].vectors[0], "fist", update_user_model=False)
        severe = scorer.score(profiles["severe_deviation"].vectors[0], "fist", update_user_model=False)

        assert standard.display_score > mild.display_score > severe.display_score
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

