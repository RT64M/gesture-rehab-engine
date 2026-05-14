from __future__ import annotations

from html import escape
import json
from pathlib import Path

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

from . import config
from .progress_analyzer import ProgressReport, SessionData


GESTURE_DISPLAY = config.GESTURE_CN
COMPACT_GESTURE_LABELS = {}


def _render_progress_curve(report: ProgressReport, output_path: Path) -> None:
    rows = report.history_summary.get("daily_progression", [])
    x = list(range(1, len(rows) + 1))
    y = [float(row["progression_index"]) for row in rows]
    velocities = [float(row["recovery_velocity"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1.5])
    axes[0].plot(x, y, color="#2d7d74", linewidth=2.5, marker="o")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Progression Index")
    axes[0].set_title("Overall Recovery Curve")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, velocities, color="#d66d3d", linewidth=2.0, marker="o")
    axes[1].axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Velocity")
    axes[1].set_xlabel("Day")
    axes[1].set_title("Recovery Velocity")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_joint_radar(report: ProgressReport, output_path: Path) -> None:
    joint_map = report.history_summary.get("overall_joint_recovery", {})
    labels = [name for name in config.JOINT_NAMES if name in joint_map]
    values = [float(joint_map[name]) for name in labels]
    if not values:
        labels = list(config.JOINT_NAMES)
        values = [0.0 for _ in labels]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values_cycle = np.concatenate([values, values[:1]])
    angles_cycle = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(8.8, 8.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_cycle, values_cycle, color="#c79b31", linewidth=2.5)
    ax.fill(angles_cycle, values_cycle, color="#c79b31", alpha=0.18)
    ax.set_xticks(angles)
    ax.set_xticklabels([label.replace("_", "\n").title() for label in labels], fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Joint Recovery Radar", pad=24)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_tau_heatmap(session_data: SessionData, output_path: Path) -> None:
    records = session_data.sorted_records()
    timestamps = sorted({record.timestamp for record in records})
    gestures = [gesture for gesture in config.TARGET_GESTURES if any(record.gesture == gesture for record in records)]
    if not gestures:
        gestures = sorted({record.gesture for record in records})

    tau_matrix = np.full((len(gestures), len(timestamps)), np.nan, dtype=np.float64)
    gesture_index = {gesture: index for index, gesture in enumerate(gestures)}
    timestamp_index = {timestamp: index for index, timestamp in enumerate(timestamps)}

    for record in records:
        tau_matrix[gesture_index[record.gesture], timestamp_index[record.timestamp]] = float(record.tau)

    fig_height = max(5.0, len(gestures) * 0.36)
    fig, ax = plt.subplots(figsize=(max(9.0, len(timestamps) * 0.42), fig_height))
    image = ax.imshow(tau_matrix, aspect="auto", cmap="YlGnBu")
    ax.set_title("Tau Evolution Heatmap")
    ax.set_xlabel("Time")
    ax.set_ylabel("Gesture")
    ax.set_xticks(range(len(timestamps)))
    step = max(1, len(timestamps) // 12)
    shown_ticks = list(range(0, len(timestamps), step))
    if shown_ticks[-1] != len(timestamps) - 1:
        shown_ticks.append(len(timestamps) - 1)
    ax.set_xticks(shown_ticks)
    ax.set_xticklabels([timestamps[index] for index in shown_ticks], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(gestures)))
    ax.set_yticklabels(
        [COMPACT_GESTURE_LABELS.get(gesture, GESTURE_DISPLAY.get(gesture, gesture)) for gesture in gestures],
        fontsize=8 if len(gestures) > 8 else 9,
    )
    fig.colorbar(image, ax=ax, label="tau")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_html(report: ProgressReport, output_dir: Path) -> str:
    warnings_html = "".join(f"<li>{escape(item)}</li>" for item in report.warnings) or "<li>No warnings</li>"
    progression_items = "".join(
        f"<li><strong>{escape(GESTURE_DISPLAY.get(gesture, gesture))}</strong>: {value:.3f}</li>"
        for gesture, value in sorted(report.per_gesture_progression.items())
    )
    joint_items = "".join(
        f"<tr><td>{escape(joint)}</td><td>{value:.3f}</td></tr>"
        for joint, value in sorted(report.history_summary.get("overall_joint_recovery", {}).items())
    )

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Offline Progress Report</title>
    <style>
      body {{
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        margin: 0;
        background: #f6f3ed;
        color: #173329;
      }}
      .page {{
        max-width: 1080px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }}
      .hero {{
        background: linear-gradient(135deg, #fdf8ef, #eaf5ef);
        border: 1px solid #d8e6de;
        border-radius: 18px;
        padding: 24px;
      }}
      .cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin: 18px 0 28px;
      }}
      .card {{
        background: #ffffff;
        border-radius: 14px;
        padding: 16px;
        border: 1px solid #dde8e1;
      }}
      h1, h2 {{
        margin: 0 0 10px;
      }}
      section {{
        margin-top: 26px;
      }}
      img {{
        width: 100%;
        border-radius: 14px;
        border: 1px solid #dde8e1;
        background: #fff;
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 18px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        background: #fff;
        border-radius: 14px;
        overflow: hidden;
      }}
      td, th {{
        padding: 10px 12px;
        border-bottom: 1px solid #edf2ee;
        text-align: left;
      }}
      ul {{
        margin: 0;
        padding-left: 20px;
      }}
    </style>
  </head>
  <body>
    <div class="page">
      <header class="hero">
        <h1>Offline Progress Report</h1>
        <p>Progress is reverse-inferred from simulated Stage 2/3 round histories using formula variables such as tau, baseline, momentum, scores, and simulated user means. No real patient data are used.</p>
      </header>

      <section class="cards">
        <article class="card">
          <h2>Overall Progress</h2>
          <p>{report.progression_index:.3f}</p>
        </article>
        <article class="card">
          <h2>Composite Recovery Score</h2>
          <p>{report.recovery_score:.1f}</p>
        </article>
        <article class="card">
          <h2>Current Recovery Velocity</h2>
          <p>{report.recovery_velocity:.4f}</p>
        </article>
        <article class="card">
          <h2>Training Recommendation</h2>
          <p>{escape(report.recommendation)}</p>
        </article>
      </section>

      <section>
        <h2>Per-Gesture Progress</h2>
        <ul>{progression_items}</ul>
      </section>

      <section>
        <h2>Warnings</h2>
        <ul>{warnings_html}</ul>
      </section>

      <section class="grid">
        <div>
          <h2>Overall Progress Curve</h2>
          <img src="{escape((output_dir / 'progress_curve.png').name)}" alt="Overall progress curve">
        </div>
        <div>
          <h2>Joint Recovery Radar</h2>
          <img src="{escape((output_dir / 'joint_radar.png').name)}" alt="Joint recovery radar">
        </div>
        <div>
          <h2>Threshold Evolution Heatmap</h2>
          <img src="{escape((output_dir / 'tau_heatmap.png').name)}" alt="Threshold evolution heatmap">
        </div>
      </section>

      <section>
        <h2>Current Joint Recovery Ratios</h2>
        <table>
          <thead>
            <tr><th>Joint Feature</th><th>Recovery Ratio</th></tr>
          </thead>
          <tbody>{joint_items}</tbody>
        </table>
      </section>
    </div>
  </body>
</html>
"""


def generate_progress_report(
    report: ProgressReport,
    session_data: SessionData,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    progress_curve = output_path / "progress_curve.png"
    joint_radar = output_path / "joint_radar.png"
    tau_heatmap = output_path / "tau_heatmap.png"
    html_path = output_path / "progress_report.html"
    json_path = output_path / "report.json"

    _render_progress_curve(report, progress_curve)
    _render_joint_radar(report, joint_radar)
    _render_tau_heatmap(session_data, tau_heatmap)

    html_path.write_text(_render_html(report, output_path), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "report": report.to_dict(),
                "session_data": session_data.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "report_json": json_path,
        "report_html": html_path,
        "progress_curve": progress_curve,
        "joint_radar": joint_radar,
        "tau_heatmap": tau_heatmap,
    }
