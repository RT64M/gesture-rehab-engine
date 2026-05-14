from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.progress_analyzer import PROGRESSION_COMPONENT_WEIGHTS


COMPONENT_LABELS = {
    "strictness_progression": "Tau tightening",
    "descriptor_recovery": "Descriptor recovery",
    "baseline_margin": "Score-baseline margin",
    "momentum_trend": "Momentum trend",
    "challenge_zone_alignment": "Challenge-zone alignment",
}


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _daily_records(data: dict) -> tuple[list[dict], list[dict]]:
    records = list(data["session_data"]["records"])
    daily = list(data["report"]["history_summary"]["daily_progression"])
    return records, daily


def _aggregate_records(records: list[dict]) -> dict[str, dict[str, float]]:
    by_timestamp: dict[str, list[dict]] = {}
    for record in records:
        by_timestamp.setdefault(record["timestamp"], []).append(record)

    rows: dict[str, dict[str, float]] = {}
    for timestamp, grouped in by_timestamp.items():
        rows[timestamp] = {
            "mean_score": float(np.mean([row["mean_score"] for row in grouped])),
            "baseline": float(np.mean([row["baseline"] for row in grouped])),
            "tau": float(np.mean([row["tau"] for row in grouped])),
            "momentum": float(np.mean([row["momentum"] for row in grouped])),
            "in_zone_ratio": float(np.mean([row["challenge_zone"] == "in_zone" for row in grouped])),
        }
    return rows


def _render_state_variables(data: dict, output_path: Path) -> None:
    records, daily = _daily_records(data)
    aggregates = _aggregate_records(records)
    timestamps = [row["timestamp"] for row in daily]
    x = np.arange(1, len(timestamps) + 1)
    score = [aggregates[timestamp]["mean_score"] for timestamp in timestamps]
    baseline = [aggregates[timestamp]["baseline"] for timestamp in timestamps]
    tau = [aggregates[timestamp]["tau"] for timestamp in timestamps]
    momentum = [aggregates[timestamp]["momentum"] for timestamp in timestamps]
    descriptor = [float(row.get("descriptor_recovery", 0.0)) for row in daily]

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.2), sharex=True, height_ratios=[1.1, 1.0])

    axes[0].axhspan(60.0, 85.0, color="#e8d8a8", alpha=0.28, label="Challenge zone")
    axes[0].plot(x, score, color="#2d7d74", linewidth=2.4, label="Mean score")
    axes[0].plot(x, baseline, color="#58595b", linewidth=2.0, linestyle="--", label="Baseline")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Observed Score State")
    axes[0].grid(alpha=0.22)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8, frameon=False)

    axes[1].plot(x, tau, color="#3567a7", linewidth=2.4, label="Mean tau")
    axes[1].plot(x, momentum, color="#b65f3a", linewidth=2.0, label="Momentum")
    axes[1].set_ylabel("Tau / momentum")
    axes[1].set_xlabel("Simulation day")
    axes[1].grid(alpha=0.22)

    right = axes[1].twinx()
    right.plot(x, descriptor, color="#6f8f3a", linewidth=2.0, linestyle="-.", label="Descriptor recovery")
    right.set_ylim(0.0, 1.0)
    right.set_ylabel("Descriptor recovery")

    lines, labels = axes[1].get_legend_handles_labels()
    right_lines, right_labels = right.get_legend_handles_labels()
    axes[1].legend(lines + right_lines, labels + right_labels, loc="upper center", ncol=3, fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_progress_components(data: dict, output_path: Path) -> None:
    _records, daily = _daily_records(data)
    weights = data["report"]["history_summary"].get(
        "progression_component_weights",
        PROGRESSION_COMPONENT_WEIGHTS,
    )
    x = np.arange(1, len(daily) + 1)
    component_keys = list(PROGRESSION_COMPONENT_WEIGHTS)
    weighted_components = [
        [float(row.get(key, 0.0)) * float(weights.get(key, 0.0)) for row in daily]
        for key in component_keys
    ]
    progression = [float(row["progression_index"]) for row in daily]

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    colors = ["#3567a7", "#6f8f3a", "#c79b31", "#b65f3a", "#8d6bb8"]
    ax.stackplot(
        x,
        weighted_components,
        labels=[COMPONENT_LABELS[key] for key in component_keys],
        colors=colors,
        alpha=0.82,
    )
    ax.plot(x, progression, color="#1f1f1f", linewidth=2.4, label="Composite progression")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Weighted contribution")
    ax.set_title("Progression Inferred from Scoring Variables")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures for inverse-progress interpretation.")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=config.ARTIFACTS_DIR / "phase4_linear_recovery" / "report.json",
        help="Phase-four report JSON used as the source data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.REPORT_FIGURES_DIR,
        help="Directory for generated paper figures.",
    )
    parser.add_argument(
        "--prefix",
        default="phase4_linear_recovery",
        help="Filename prefix for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_report(args.report_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_path = args.output_dir / f"{args.prefix}_state_variables.png"
    components_path = args.output_dir / f"{args.prefix}_progress_components.png"
    _render_state_variables(data, state_path)
    _render_progress_components(data, components_path)

    print(f"state_variables: {state_path}")
    print(f"progress_components: {components_path}")


if __name__ == "__main__":
    main()
