from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRATEGY_LABELS = {
    "fixed_standardized": "Fixed",
    "heuristic_rule": "Heuristic",
    "adaptive": "Adaptive",
}


METRIC_LABELS = {
    "ability_gain": "Ability\nGain",
    "challenge_ratio": "Challenge\nRatio",
    "feedback_accuracy": "Feedback\nAccuracy",
    "frustration_risk": "Frustration\nRisk",
    "score_ability_correlation": "Score-ability corr.",
    "scenario_variance": "Scenario variance",
}

CURVE_LABELS = {
    "linear_recovery": "Linear\nRecovery",
    "s_curve_recovery": "S-Curve\nRecovery",
    "plateau": "Plateau",
    "fatigue_dip": "Fatigue\nDip",
}


plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
]
plt.rcParams["axes.unicode_minus"] = False


def _strategy_order(summary: dict) -> list[str]:
    preferred = ["fixed_standardized", "heuristic_rule", "adaptive"]
    available = list(summary["strategy_metrics"])
    return [name for name in preferred if name in available] + [name for name in available if name not in preferred]


def _labels(names: list[str]) -> list[str]:
    return [STRATEGY_LABELS.get(name, name) for name in names]


def _ci_error(summary: dict, strategy: str, metric: str) -> tuple[float, float]:
    value = float(summary["strategy_metrics"][strategy][metric])
    interval = summary.get("confidence_intervals", {}).get(strategy, {}).get(metric)
    if not interval:
        return 0.0, 0.0
    return max(0.0, value - float(interval["low"])), max(0.0, float(interval["high"]) - value)


def plot_metric_bars(summary: dict, output_path: Path) -> None:
    strategies = _strategy_order(summary)
    metrics = ["ability_gain", "challenge_ratio", "feedback_accuracy", "frustration_risk"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 5.4))

    for index, strategy in enumerate(strategies):
        values = [float(summary["strategy_metrics"][strategy][metric]) for metric in metrics]
        yerr = np.asarray([_ci_error(summary, strategy, metric) for metric in metrics], dtype=np.float64).T
        offset = (index - (len(strategies) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, yerr=yerr, capsize=3, label=STRATEGY_LABELS.get(strategy, strategy))

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric value")
    ax.set_title("Phase 5 strategy metrics with 95% bootstrap intervals")
    ax.legend(frameon=False, ncols=len(strategies), loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_scenario_heatmap(summary: dict, output_path: Path) -> None:
    strategies = _strategy_order(summary)
    curves = sorted({curve for rows in summary["scenario_metrics"].values() for curve in rows})
    matrix = np.asarray(
        [
            [summary["scenario_metrics"][strategy][curve]["challenge_ratio"] for curve in curves]
            for strategy in strategies
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(curves)))
    ax.set_xticklabels([CURVE_LABELS.get(curve, curve.replace("_", "\n")) for curve in curves])
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels(_labels(strategies))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8.5)
    ax.set_title("Challenge-zone ratio by strategy and recovery curve")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Challenge-zone ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_effect_sizes(summary: dict, output_path: Path) -> None:
    effect_sizes = summary.get("pairwise_effect_sizes", {})
    tests = summary.get("pairwise_permutation_tests", {})
    names = list(effect_sizes)
    values = [float(effect_sizes[name]) for name in names]
    labels = [name.replace("__vs__", "\nvs\n") for name in names]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    colors = ["#4e79a7" if value >= 0 else "#e15759" for value in values]
    bars = ax.bar(np.arange(len(names)), values, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Cohen's d on ability gain")
    ax.set_title("Pairwise strategy effect sizes and permutation p-values")
    ax.grid(axis="y", alpha=0.25)

    for bar, name, value in zip(bars, names, values):
        p_value = tests.get(name, {}).get("p_value")
        label = f"d={value:.2f}"
        if p_value is not None:
            label += f"\np={float(p_value):.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + (0.03 if value >= 0 else -0.08),
            label,
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def generate_phase5_figures(summary_path: Path, output_dir: Path) -> dict[str, Path]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "phase5_strategy_metrics": output_dir / "C1_phase5_strategy_metrics.png",
        "phase5_scenario_heatmap": output_dir / "C2_phase5_scenario_heatmap.png",
        "phase5_effect_sizes": output_dir / "C3_phase5_effect_sizes.png",
    }
    plot_metric_bars(summary, outputs["phase5_strategy_metrics"])
    plot_scenario_heatmap(summary, outputs["phase5_scenario_heatmap"])
    plot_effect_sizes(summary, outputs["phase5_effect_sizes"])
    return outputs
