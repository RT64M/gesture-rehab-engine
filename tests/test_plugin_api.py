import json
import shutil
from pathlib import Path

import numpy as np

from src.plugin_api import score_plugin_request


DESCRIPTOR_DIR = Path("data/processed/descriptors")
FEATURES_DIR = Path("data/processed/features")


def make_local_temp_dir(name: str) -> Path:
    root = Path.cwd() / ".tmp-tests"
    root.mkdir(exist_ok=True)
    path = root / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def test_plugin_request_scores_feature_vector_and_persists_state():
    temp_dir = make_local_temp_dir("plugin-api")
    try:
        state_path = temp_dir / "state.json"
        descriptor = json.loads((DESCRIPTOR_DIR / "fist.json").read_text(encoding="utf-8"))
        payload = {
            "user_id": "test-user",
            "gesture": "fist",
            "feature_vector": descriptor["mu"],
            "update_user_model": True,
        }

        first = score_plugin_request(payload, DESCRIPTOR_DIR, FEATURES_DIR, state_path)
        second = score_plugin_request(payload, DESCRIPTOR_DIR, FEATURES_DIR, state_path)

        assert first["target_gesture"] == "fist"
        assert first["display_score"] >= 90
        assert second["state"]["gesture_samples"] == 2
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["users"]["test-user"]["gestures"]["fist"]["n_samples"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_plugin_request_accepts_landmarks_input_without_persistence():
    landmarks = np.asarray(
        [[0.5 + (index % 5) * 0.02, 0.9 - (index // 5) * 0.06] for index in range(21)],
        dtype=float,
    )
    payload = {
        "user_id": "landmark-user",
        "gesture": "palm",
        "landmarks": landmarks.tolist(),
        "update_user_model": False,
    }

    response = score_plugin_request(payload, DESCRIPTOR_DIR, FEATURES_DIR, state_path=None)

    assert response["plugin_api_version"] == "0.1"
    assert response["target_gesture"] == "palm"
    assert response["state"]["updated"] is False
    assert len(response["feature_names"]) == 26

