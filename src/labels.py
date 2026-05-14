from __future__ import annotations


GESTURE_LABELS = {
    "call": "Call",
    "fist": "Fist",
    "four": "Four",
    "ok": "OK",
    "one": "One",
    "palm": "Palm",
    "peace": "Peace",
    "rock": "Rock",
    "stop": "Stop",
    "three": "Three",
}


PROFILE_LABELS = {
    "standard": "Standard Sample",
    "mild_deviation": "Mild Deviation",
    "severe_deviation": "Severe Deviation",
    "improving_sequence": "Improving Sequence",
}


PROFILE_DESCRIPTIONS = {
    "standard": "Uses the gesture descriptor mean vector directly.",
    "mild_deviation": "Applies a mild offset toward the most confusable direction.",
    "severe_deviation": "Applies a larger offset toward the most confusable direction.",
    "improving_sequence": "Moves gradually from severe deviation toward the standard pose.",
}


SCENARIO_LABELS = {
    "linear_recovery": "Linear Recovery",
    "s_curve_recovery": "S-Curve Recovery",
    "plateau": "Plateau",
    "fatigue_dip": "Fatigue Dip",
}


SCENARIO_DESCRIPTIONS = {
    "linear_recovery": "Scores rise steadily with limited fluctuation.",
    "s_curve_recovery": "Progress is slow early, faster in the middle, and stabilizes late.",
    "plateau": "Early improvement is followed by a plateau.",
    "fatigue_dip": "A temporary fatigue dip interrupts recovery before later improvement.",
}


def gesture_label(gesture: str) -> str:
    return GESTURE_LABELS.get(gesture, gesture)
