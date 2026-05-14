from pathlib import Path
import shutil

import numpy as np

from src.descriptors import GestureDescriptor, build_descriptor, classify


def make_cluster(center: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.08, size=(n_samples, len(center)))
    return center + noise


def make_local_temp_dir(name: str) -> Path:
    root = Path.cwd() / ".tmp-tests"
    root.mkdir(exist_ok=True)
    path = root / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_build_descriptor_shapes_and_score_range():
    feature_names = ["f0", "f1", "f2"]
    cluster = make_cluster(np.array([1.0, 2.0, 3.0]), n_samples=40, seed=1)

    desc = build_descriptor("demo", cluster, feature_names)

    assert desc.gesture == "demo"
    assert desc.n_samples == 40
    assert desc.mu.shape == (3,)
    assert desc.sigma.shape == (3, 3)
    assert desc.sigma_inv.shape == (3, 3)
    assert desc.feature_names == feature_names

    perfect_score = desc.score(desc.mu)
    assert 0.0 < perfect_score <= 1.0
    assert np.isclose(perfect_score, 1.0)


def test_score_decreases_with_distance():
    feature_names = ["f0", "f1"]
    cluster = make_cluster(np.array([0.0, 0.0]), n_samples=50, seed=2)
    desc = build_descriptor("demo", cluster, feature_names)

    near = desc.mu + np.array([0.05, 0.05])
    far = desc.mu + np.array([2.0, 2.0])

    assert desc.mahalanobis(near) < desc.mahalanobis(far)
    assert desc.score(near) > desc.score(far)


def test_classify_prefers_closest_descriptor():
    feature_names = ["f0", "f1"]
    cluster_a = make_cluster(np.array([-1.0, -1.0]), n_samples=50, seed=3)
    cluster_b = make_cluster(np.array([2.0, 2.0]), n_samples=50, seed=4)

    desc_a = build_descriptor("a", cluster_a, feature_names)
    desc_b = build_descriptor("b", cluster_b, feature_names)

    probe = np.array([2.1, 1.9])
    best, distances = classify(probe, {"a": desc_a, "b": desc_b})

    assert best == "b"
    assert distances["b"] < distances["a"]


def test_descriptor_json_round_trip():
    feature_names = ["f0", "f1", "f2", "f3"]
    cluster = make_cluster(np.array([1.5, -2.0, 0.5, 3.0]), n_samples=60, seed=5)
    desc = build_descriptor("roundtrip", cluster, feature_names)

    temp_dir = make_local_temp_dir("descriptor-roundtrip")
    out_path = temp_dir / "roundtrip.json"
    try:
        desc.save_json(out_path)

        loaded = GestureDescriptor.load_json(out_path)

        assert loaded.gesture == desc.gesture
        assert loaded.n_samples == desc.n_samples
        assert loaded.feature_names == desc.feature_names
        assert np.allclose(loaded.mu, desc.mu)
        assert np.allclose(loaded.sigma, desc.sigma)
        assert np.allclose(loaded.sigma_inv, desc.sigma_inv)

        sample = cluster[0]
        assert np.isclose(loaded.mahalanobis(sample), desc.mahalanobis(sample))
        assert np.isclose(loaded.score(sample), desc.score(sample))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
