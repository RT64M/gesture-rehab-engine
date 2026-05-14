from __future__ import annotations

from dataclasses import dataclass


TOO_HARD = "too_hard"
IN_ZONE = "in_zone"
TOO_EASY = "too_easy"


ZONE_LABELS = {
    TOO_HARD: "太难",
    IN_ZONE: "适中",
    TOO_EASY: "太简单",
}


ZONE_DESCRIPTIONS = {
    TOO_HARD: "当前难度偏高，用户容易产生挫败感。",
    IN_ZONE: "当前难度处于有效挑战区，最适合训练。",
    TOO_EASY: "当前难度偏低，用户可能缺少挑战。",
}


@dataclass
class ChallengeZoneStatus:
    code: str
    label: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "label": self.label,
            "description": self.description,
        }


def get_challenge_zone(score: float, low: float = 60.0, high: float = 85.0) -> str:
    if score < low:
        return TOO_HARD
    if score > high:
        return TOO_EASY
    return IN_ZONE


def challenge_zone_status(score: float, low: float = 60.0, high: float = 85.0) -> ChallengeZoneStatus:
    code = get_challenge_zone(score, low=low, high=high)
    return ChallengeZoneStatus(
        code=code,
        label=ZONE_LABELS[code],
        description=ZONE_DESCRIPTIONS[code],
    )
