from __future__ import annotations

from typing import Iterable

import numpy as np

from .descriptors import GestureDescriptor
from .labels import gesture_label


FINGER_LABELS = {
    "thumb": "拇指",
    "index": "食指",
    "middle": "中指",
    "ring": "无名指",
    "pinky": "小指",
}


def _gesture_label(gesture: str) -> str:
    return gesture_label(gesture)


def _humanize_feature_name(name: str) -> str:
    if name.startswith("dist_"):
        parts = name.removeprefix("dist_").removesuffix("_tip").split("_")
        if len(parts) == 2:
            left, right = parts
            return f"{FINGER_LABELS.get(left, left)}- {FINGER_LABELS.get(right, right)}指尖距离".replace("- ", "-")
    if name.startswith("spread_"):
        parts = name.removeprefix("spread_").split("_")
        if len(parts) == 2:
            left, right = parts
            return f"{FINGER_LABELS.get(left, left)}-{FINGER_LABELS.get(right, right)}张开角"
    if "_" not in name:
        return name
    finger, joint = name.split("_", 1)
    return f"{FINGER_LABELS.get(finger, finger)} {joint.upper()} 关节角度"


def _direction_text(feature_name: str, z_score: float) -> str:
    if feature_name.startswith("dist_"):
        return "偏大" if z_score > 0 else "偏小"
    if feature_name.startswith("spread_"):
        return "张开过大" if z_score > 0 else "张开不足"
    return "偏大" if z_score > 0 else "偏小"


def summarize_deviations(
    feature_vector: np.ndarray,
    descriptor: GestureDescriptor,
    top_k: int = 3,
) -> list[dict[str, float | str]]:
    feature_vector = np.asarray(feature_vector, dtype=np.float64)
    sigma_diag = np.clip(np.diag(descriptor.sigma), 1e-8, None)
    z_scores = (feature_vector - descriptor.mu) / np.sqrt(sigma_diag)
    order = np.argsort(np.abs(z_scores))[::-1]
    summary = []
    for index in order[:top_k]:
        summary.append(
            {
                "feature": descriptor.feature_names[index],
                "label": _humanize_feature_name(descriptor.feature_names[index]),
                "z_score": float(z_scores[index]),
                "magnitude": float(abs(z_scores[index])),
            }
        )
    return summary


def generate_feedback(
    feature_vector: np.ndarray,
    descriptor: GestureDescriptor,
    target_gesture: str,
    predicted_gesture: str | None = None,
    top_k: int = 3,
) -> list[str]:
    deviations = summarize_deviations(feature_vector, descriptor, top_k=top_k)
    lines: list[str] = []

    if predicted_gesture and predicted_gesture != target_gesture:
        lines.append(
            f"当前更像“{_gesture_label(predicted_gesture)}”，目标是“{_gesture_label(target_gesture)}”。"
        )

    if not deviations:
        lines.append("姿势已经接近标准，可以继续保持。")
        return lines

    strongest = deviations[0]["magnitude"]
    if strongest < 0.6 and not lines:
        return ["姿势已经接近标准，可以继续保持。"]

    for item in deviations:
        feature_name = str(item["feature"])
        label = str(item["label"])
        z_score = float(item["z_score"])
        lines.append(f"{label}{_direction_text(feature_name, z_score)}（{abs(z_score):.1f}σ）。")
    return lines


def feedback_payload(lines: Iterable[str]) -> list[str]:
    return [str(line) for line in lines]
