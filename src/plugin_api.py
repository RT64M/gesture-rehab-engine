from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .features import extract_features, feature_names
from .scorer import ColdStartScorer


PLUGIN_API_VERSION = "0.1"
DEFAULT_DESCRIPTOR_DIR = config.DATA_PROCESSED_DIR / "descriptors"
DEFAULT_FEATURES_DIR = config.DATA_PROCESSED_DIR / "features"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_state() -> dict[str, Any]:
    return {
        "plugin_api_version": PLUGIN_API_VERSION,
        "updated_at": _utc_now(),
        "users": {},
    }


def load_plugin_state(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return _new_state()
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    state.setdefault("plugin_api_version", PLUGIN_API_VERSION)
    state.setdefault("updated_at", _utc_now())
    state.setdefault("users", {})
    return state


def save_plugin_state(path: Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _utc_now()
    output_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _user_gesture_state(state: dict[str, Any], user_id: str, gesture: str) -> dict[str, Any]:
    users = state.setdefault("users", {})
    user_state = users.setdefault(user_id, {})
    gestures = user_state.setdefault("gestures", {})
    return gestures.setdefault(gesture, {"samples": [], "baseline_distances": []})


def _hydrate_user_model(
    scorer: ColdStartScorer,
    state: dict[str, Any],
    user_id: str,
    gesture: str,
) -> None:
    model = scorer.get_user_model(gesture)
    stored = _user_gesture_state(state, user_id, gesture)
    model._samples = [
        np.asarray(sample, dtype=np.float64)
        for sample in stored.get("samples", [])
    ]
    model._baseline_distances = [
        float(value)
        for value in stored.get("baseline_distances", [])
    ]


def _persist_user_model(
    scorer: ColdStartScorer,
    state: dict[str, Any],
    user_id: str,
    gesture: str,
) -> None:
    model = scorer.get_user_model(gesture)
    stored = _user_gesture_state(state, user_id, gesture)
    stored["samples"] = [sample.tolist() for sample in model._samples]
    stored["baseline_distances"] = [float(value) for value in model._baseline_distances]
    stored["n_samples"] = int(model.n_samples)
    stored["blend_weight"] = float(model.blend_weight)
    stored["tau"] = float(model.tau)
    stored["updated_at"] = _utc_now()


def _feature_vector_from_request(payload: dict[str, Any]) -> np.ndarray:
    if "feature_vector" in payload:
        vector = np.asarray(payload["feature_vector"], dtype=np.float64)
    elif "landmarks" in payload:
        landmarks = np.asarray(payload["landmarks"], dtype=np.float64)
        vector = extract_features(landmarks).geometric_vector
    else:
        raise ValueError("Plugin request must include either 'feature_vector' or 'landmarks'.")

    expected_dim = config.N_GEOMETRIC_FEATURES
    if vector.shape != (expected_dim,):
        raise ValueError(f"feature_vector must have shape ({expected_dim},), got {tuple(vector.shape)}.")
    return vector


def score_plugin_request(
    payload: dict[str, Any],
    descriptor_dir: Path = DEFAULT_DESCRIPTOR_DIR,
    features_dir: Path = DEFAULT_FEATURES_DIR,
    state_path: Path | None = None,
) -> dict[str, Any]:
    gesture = str(payload.get("gesture", ""))
    if gesture not in config.TARGET_GESTURES:
        raise ValueError(f"Unknown gesture: {gesture}")

    user_id = str(payload.get("user_id", "default"))
    update_user_model = bool(payload.get("update_user_model", True))
    top_k_feedback = int(payload.get("top_k_feedback", 3))
    tau_override = payload.get("tau_override")
    tau_override = None if tau_override is None else float(tau_override)

    vector = _feature_vector_from_request(payload)
    state = load_plugin_state(state_path)
    scorer = ColdStartScorer.from_artifacts(Path(descriptor_dir), Path(features_dir))
    _hydrate_user_model(scorer, state, user_id, gesture)

    result = scorer.score(
        vector,
        gesture,
        update_user_model=update_user_model,
        top_k_feedback=top_k_feedback,
        tau_override=tau_override,
    )
    if update_user_model:
        _persist_user_model(scorer, state, user_id, gesture)
        save_plugin_state(state_path, state)

    response = result.to_dict()
    response["plugin_api_version"] = PLUGIN_API_VERSION
    response["user_id"] = user_id
    response["feature_names"] = feature_names()
    response["state"] = {
        "updated": bool(update_user_model),
        "state_path": None if state_path is None else str(Path(state_path)),
        "gesture_samples": scorer.get_user_model(gesture).n_samples,
        "blend_weight": scorer.get_user_model(gesture).blend_weight,
    }
    return response

