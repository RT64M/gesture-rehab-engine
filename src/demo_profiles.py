from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .descriptors import GestureDescriptor
from .labels import PROFILE_DESCRIPTIONS, PROFILE_LABELS


PROFILE_ORDER = [
    "standard",
    "mild_deviation",
    "severe_deviation",
    "improving_sequence",
]

@dataclass
class DemoProfile:
    name: str
    label: str
    description: str
    kind: str
    vectors: list[np.ndarray]
    competitor_gesture: str


def nearest_competitor(
    gesture: str,
    descriptors: dict[str, GestureDescriptor],
) -> str:
    target = descriptors[gesture]
    distances = {
        other_gesture: target.mahalanobis(other_descriptor.mu)
        for other_gesture, other_descriptor in descriptors.items()
        if other_gesture != gesture
    }
    return min(distances, key=distances.get)


def build_demo_profiles(
    gesture: str,
    descriptors: dict[str, GestureDescriptor],
) -> dict[str, DemoProfile]:
    target = descriptors[gesture]
    competitor = nearest_competitor(gesture, descriptors)
    delta = descriptors[competitor].mu - target.mu
    std = np.sqrt(np.clip(np.diag(target.sigma), 1e-8, None))
    direction = np.sign(delta)
    direction[direction == 0] = 1.0

    def offset(strength: float) -> np.ndarray:
        return target.mu + strength * std * direction

    return {
        "standard": DemoProfile(
            name="standard",
            label=PROFILE_LABELS["standard"],
            description=PROFILE_DESCRIPTIONS["standard"],
            kind="single",
            vectors=[target.mu.copy()],
            competitor_gesture=competitor,
        ),
        "mild_deviation": DemoProfile(
            name="mild_deviation",
            label=PROFILE_LABELS["mild_deviation"],
            description=PROFILE_DESCRIPTIONS["mild_deviation"],
            kind="single",
            vectors=[offset(0.75)],
            competitor_gesture=competitor,
        ),
        "severe_deviation": DemoProfile(
            name="severe_deviation",
            label=PROFILE_LABELS["severe_deviation"],
            description=PROFILE_DESCRIPTIONS["severe_deviation"],
            kind="single",
            vectors=[offset(1.5)],
            competitor_gesture=competitor,
        ),
        "improving_sequence": DemoProfile(
            name="improving_sequence",
            label=PROFILE_LABELS["improving_sequence"],
            description=PROFILE_DESCRIPTIONS["improving_sequence"],
            kind="sequence",
            vectors=[offset(1.5), offset(1.0), offset(0.5), target.mu.copy()],
            competitor_gesture=competitor,
        ),
    }
