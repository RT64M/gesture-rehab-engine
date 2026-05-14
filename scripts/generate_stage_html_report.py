from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.progression import SCENARIO_ORDER
from src import config


REPORT_DIR = config.ARTIFACTS_DIR / "stage_html_report"
FIGURES_DIRS = (config.REPORT_FIGURES_DIR, config.ARTIFACT_FIGURES_DIR)
EXTRA_OUTPUTS_DIR = config.ARTIFACTS_DIR / "extra_outputs"
PHASE5_SUMMARY = config.ARTIFACTS_DIR / "phase5_prep" / "summary.json"

PHASE_TITLES = {
    1: "Phase 1: Dataset Analysis and Descriptor Design",
    2: "Phase 2: Calibration-Free Cold-Start Scoring",
    3: "Phase 3: Momentum-Based Dynamic Thresholding",
    4: "Phase 4: Reverse Progress Assessment",
    5: "Phase 5: Scoring Strategy Comparison",
}

GESTURE_DISPLAY = config.GESTURE_CN

SCENARIO_LABELS = {
    "linear_recovery": "Linear Recovery",
    "s_curve_recovery": "S-Curve Recovery",
    "plateau": "Plateau",
    "fatigue_dip": "Fatigue Dip",
}

PROFILE_LABELS = {
    "standard": "Standard Sample",
    "mild_deviation": "Mild Deviation",
    "severe_deviation": "Severe Deviation",
    "improving_sequence": "Improving Sequence",
}


@dataclass
class FigureCard:
    filename: str
    title: str
    what_it_shows: str
    what_we_did: str
    takeaways: str


def rel(path: Path) -> str:
    return os.path.relpath(path, REPORT_DIR).replace(os.sep, "/")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_build_descriptor_summary() -> dict:
    text = (EXTRA_OUTPUTS_DIR / "build_descriptors_output.txt").read_text(encoding="utf-8")
    sample_pattern = re.compile(
        r"\[\s*(?P<label>[^\]]+)\]\s+"
        r"train paths:\s+\d+\s+valid samples:\s+(?P<train>\d+)\s+"
        r"val paths:\s+\d+\s+valid samples:\s+(?P<val>\d+)\s+"
        r"test paths:\s+\d+\s+valid samples:\s+(?P<test>\d+)\s+"
        r"merged train\+val samples:\s+(?P<merged>\d+)",
        re.MULTILINE,
    )
    samples = []
    for index, match in enumerate(sample_pattern.finditer(text)):
        raw_label = match.group("label").strip()
        gesture = config.TARGET_GESTURES[index] if index < len(config.TARGET_GESTURES) else raw_label
        samples.append(
            {
                "label": GESTURE_DISPLAY.get(gesture, GESTURE_DISPLAY.get(raw_label, raw_label)),
                "train": int(match.group("train")),
                "val": int(match.group("val")),
                "test": int(match.group("test")),
                "merged": int(match.group("merged")),
            }
        )

    overall_match = re.search(r"Overall accuracy:\s+(\d+)/(\d+)\s+=\s+([\d.]+)%", text)
    return {
        "samples": samples,
        "correct": int(overall_match.group(1)) if overall_match else None,
        "total": int(overall_match.group(2)) if overall_match else None,
        "accuracy": float(overall_match.group(3)) if overall_match else None,
    }


def load_cold_start_profiles() -> dict:
    profiles = {}
    for name in ("standard", "mild_deviation", "severe_deviation", "improving_sequence"):
        path = EXTRA_OUTPUTS_DIR / f"cold_start_{name}.txt"
        text = path.read_text(encoding="utf-8")
        steps = re.findall(
            r"\[step\s+(\d+)\]\s+score=\s*(\d+)\s+raw=([\d.]+)\s+distance=([\d.]+)\s+tau=([\d.]+)\s+predicted=([a-z_]+)\s+blend=([\d.]+)",
            text,
        )
        profiles[name] = {
            "name": PROFILE_LABELS[name],
            "raw_text": text,
            "steps": [
                {
                    "step": int(step),
                    "score": int(score),
                    "raw": float(raw),
                    "distance": float(distance),
                    "tau": float(tau),
                    "predicted": predicted,
                    "blend": float(blend),
                }
                for step, score, raw, distance, tau, predicted, blend in steps
            ],
        }
    return profiles


def load_phase4_reports() -> dict:
    payload = {}
    for scenario in SCENARIO_ORDER:
        path = config.ARTIFACTS_DIR / f"phase4_{scenario}" / "report.json"
        payload[scenario] = read_json(path)["report"]
    return payload


def render_nav(current_page: str) -> str:
    links = ['<a href="index.html">Overview</a>']
    for phase in range(1, 6):
        target = f"phase{phase}.html"
        cls = ' class="active"' if current_page == target else ""
        links.append(f'<a href="{target}"{cls}>Phase {phase}</a>')
    return '<nav class="nav">' + "".join(links) + "</nav>"


def render_metrics(metrics: list[tuple[str, str]]) -> str:
    cards = []
    for label, value in metrics:
        cards.append(
            f"""
            <article class="metric-card">
              <div class="metric-label">{escape(label)}</div>
              <div class="metric-value">{escape(value)}</div>
            </article>
            """
        )
    return '<section class="metric-grid">' + "".join(cards) + "</section>"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(col)}</th>" for col in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def render_figure_card(card: FigureCard) -> str:
    figure_path = next(
        (directory / card.filename for directory in FIGURES_DIRS if (directory / card.filename).exists()),
        config.ARTIFACT_FIGURES_DIR / card.filename,
    )
    return f"""
    <section class="figure-card">
      <h3>{escape(card.title)}</h3>
      <img src="{escape(rel(figure_path))}" alt="{escape(card.title)}">
      <div class="figure-notes">
        <p><strong>What this figure shows:</strong> {escape(card.what_it_shows)}</p>
        <p><strong>What this stage did:</strong> {escape(card.what_we_did)}</p>
        <p><strong>How to interpret it:</strong> {escape(card.takeaways)}</p>
      </div>
    </section>
    """


def wrap_page(title: str, intro: str, content: str, current_page: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(title)}</title>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <div class="page-shell">
      <header class="hero">
        <p class="eyebrow">Gesture Scoring Engine</p>
        <h1>{escape(title)}</h1>
        <p class="intro">{escape(intro)}</p>
        {render_nav(current_page)}
      </header>
      <main class="content">
        {content}
      </main>
    </div>
  </body>
</html>
"""


def build_index_page() -> str:
    sections = []
    for phase in range(1, 6):
        sections.append(
            f"""
            <article class="phase-link">
              <h2><a href="phase{phase}.html">{escape(PHASE_TITLES[phase])}</a></h2>
              <p>Open this phase page to review the stage objective, figure evidence, interpretation notes, and current conclusions.</p>
            </article>
            """
        )
    return wrap_page(
        title="Stage-by-Stage HTML Report Overview",
        intro="This report package organizes the project into five phase pages. All generated report text and figure explanations are written in English.",
        content="".join(sections),
        current_page="index.html",
    )


def extract_main_content(page_html: str) -> str:
    match = re.search(r'<main class="content">\s*(.*)\s*</main>', page_html, re.DOTALL)
    if not match:
        raise ValueError("Could not extract <main class=\"content\"> from generated page.")
    return match.group(1).strip()


def build_final_report_page(phase_pages: dict[int, str]) -> str:
    toc = []
    sections = []
    for phase in range(1, 6):
        toc.append(f'<a href="#phase-{phase}">{escape(PHASE_TITLES[phase])}</a>')
        sections.append(
            f"""
            <section class="final-phase-section" id="phase-{phase}">
              <div class="final-phase-header">
                <p class="phase-kicker">Phase {phase}</p>
                <h2>{escape(PHASE_TITLES[phase])}</h2>
              </div>
              {extract_main_content(phase_pages[phase])}
            </section>
            """
        )

    content = (
        '<section class="section">'
        '<h2>Report Contents</h2>'
        '<div class="final-toc">' + "".join(toc) + "</div>"
        '<p>This is the integrated single-page HTML report. It can be viewed directly, shared, or printed to PDF from the browser.</p>'
        '</section>'
        + "".join(sections)
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Gesture Scoring Engine Integrated HTML Report</title>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <div class="page-shell final-shell">
      <header class="hero">
        <p class="eyebrow">Gesture Scoring Engine</p>
        <h1>Integrated HTML Report</h1>
        <p class="intro">This page combines phases 1 through 5 into one English report, explaining what each figure shows, what each stage completed, and how to interpret the current results.</p>
        <nav class="nav"><a href="index.html">Stage Overview</a><a href="final_report.html" class="active">Final Report</a></nav>
      </header>
      <main class="content">
        {content}
      </main>
    </div>
  </body>
</html>
"""


def build_phase1_page(summary: dict) -> str:
    metrics = render_metrics(
        [
            ("Target gestures", f"{len(config.TARGET_GESTURES)} classes"),
            ("Fitting split", "train + val"),
            ("Evaluation split", "official test"),
            ("Overall accuracy", f"{summary['accuracy']:.2f}%"),
        ]
    )
    rows = [
        [
            escape(item["label"]),
            str(item["train"]),
            str(item["val"]),
            str(item["test"]),
            str(item["merged"]),
        ]
        for item in summary["samples"]
    ]
    sample_table = render_table(
        ["Gesture", "train", "val", "test", "train+val"],
        rows,
    )
    mapping_figures = [
        FigureCard(
            "gesture_mapping_grid.png",
            "Ten-Gesture Mapping Overview",
            "This compact grid combines the configured ten gesture maps as paper-style subfigures.",
            "Each small panel overlays normalized landmark samples and the mean skeleton for one target gesture.",
            "The grid keeps the report readable while preserving the single-gesture mapping files as supplemental reproducibility artifacts.",
        ),
    ]
    figures = mapping_figures + [
        FigureCard(
            "A1_inter_class_distance.png",
            "Inter-Class Mahalanobis Distance Matrix",
            "This figure compares class-to-class distances among target gesture descriptors.",
            "Each descriptor mean was evaluated against every class covariance model using Mahalanobis distance.",
            "Large off-diagonal values indicate that the learned descriptor space separates gesture classes well.",
        ),
        FigureCard(
            "A2_covariance_trace.png",
            "Covariance Trace by Gesture",
            "This figure measures the total within-class variation for each gesture descriptor.",
            "A covariance matrix was fitted for each gesture and summarized using its trace.",
            "Different trace magnitudes justify covariance-aware scoring: each gesture needs its own notion of strictness.",
        ),
        FigureCard(
            "B1_confusion_matrix.png",
            "Official-Test Confusion Matrix",
            "This figure shows the ten-class descriptor classifier on the official test split.",
            "Descriptors were fitted on train+val and evaluated with nearest-descriptor classification on the official test split.",
            f"Brighter diagonal cells are better. The current overall accuracy is {summary['accuracy']:.2f}%, supporting descriptor stability on public HaGRID data.",
        ),
        FigureCard(
            "B2_joint_angle_heatmap.png",
            "Mean Joint-Angle Heatmap",
            "This heatmap compares average values for the 15 joint-angle features across gestures.",
            "Joint-angle features were averaged per gesture and visualized across the target set.",
            "The heatmap exposes each gesture's joint signature, explaining why the geometry vector separates classes.",
        ),
        FigureCard(
            "B3_key_joint_boxplots.png",
            "Key Joint-Angle Boxplots",
            "This figure shows distributions, medians, and spreads for selected discriminative joint features.",
            "High-signal joint dimensions were selected and compared across gesture classes.",
            "Limited box overlap suggests that these joint dimensions contribute meaningful class separation.",
        ),
        FigureCard(
            "B4_fisher_ratio.png",
            "Fisher Ratio for 26 Features",
            "This chart ranks feature dimensions by discriminative power.",
            "Fisher ratios were computed for joint angles, fingertip distances, and spread angles.",
            "The ranking clarifies which feature groups carry the strongest class-separation signal.",
        ),
        FigureCard(
            "B5_ablation.png",
            "Feature-Group Ablation",
            "This figure compares classification accuracy across feature subsets.",
            "Descriptors were rebuilt with angle-only, distance-only, spread-only, and combined feature groups.",
            "The full 26D vector performs best, while several combinations remain strong, indicating a robust feature design.",
        ),
        FigureCard(
            "B6_distance_comparison.png",
            "Distance Metric Comparison",
            "This figure compares Mahalanobis, Euclidean, cosine, and Manhattan distance choices.",
            "Only the distance metric was changed; features and evaluation splits were kept fixed.",
            "Mahalanobis distance performs best, supporting the use of covariance-aware descriptor scoring.",
        ),
        FigureCard(
            "B7_covariance_matrices.png",
            "Descriptor Covariance Matrices",
            "This figure visualizes the absolute covariance structure for each gesture.",
            "Fitted covariance matrices were placed side by side for direct comparison.",
            "It extends the covariance-trace view by showing which feature dimensions are strict or flexible within each class.",
        ),
    ]
    figure_html = "".join(render_figure_card(card) for card in figures)
    content = (
        metrics
        + f'<section class="section"><h2>Stage Objective and Work Completed</h2><p>This phase builds stable, interpretable, and reproducible statistical descriptors for {len(config.TARGET_GESTURES)} target gestures from HaGRIDv2 landmarks. The pipeline downloads annotations, filters valid samples, extracts 26D geometric features, fits descriptors, and evaluates them on the official test split. The report presents compact grid figures in the main narrative; individual gesture maps remain available as supplemental artifacts.</p></section>'
        + '<section class="section"><h2>Sample Counts and Split Summary</h2>'
        + sample_table
        + "</section>"
        + f'<section class="section"><h2>Stage Conclusion</h2><p>Phase 1 produced the descriptor and feature artifacts used by later modules. The current public HaGRID official-test accuracy is {summary["accuracy"]:.2f}%, supporting the use of the geometric representation as the basis for cold-start scoring and dynamic thresholding.</p></section>'
        + figure_html
    )
    return wrap_page(
        title=PHASE_TITLES[1],
        intro="This page covers data preparation, feature modeling, and descriptor validation.",
        content=content,
        current_page="phase1.html",
    )


def build_phase2_page(cold_start_profiles: dict) -> str:
    profile_rows = []
    for key in ("standard", "mild_deviation", "severe_deviation", "improving_sequence"):
        item = cold_start_profiles[key]
        scores = " → ".join(str(step["score"]) for step in item["steps"])
        preds = " → ".join(step["predicted"] for step in item["steps"])
        profile_rows.append(
            [
                escape(item["name"]),
                str(len(item["steps"])),
                escape(scores),
                escape(preds),
            ]
        )
    metrics = render_metrics(
        [
            ("Example target gesture", "fist"),
            ("Displayed profiles", "4"),
            ("Best sample score", "100"),
            ("Improving sequence final score", "97"),
        ]
    )
    figures = [
        FigureCard(
            "A3_scoring_functions.png",
            "Scoring Function Comparison",
            "This figure compares several ways to map descriptor distance into display score, including exponential, linear, sigmoid, and Gaussian mappings.",
            "Phase 2 translates geometric deviation into user-facing scores, so the score mapping shape is a central design choice.",
            "The exponential mapping provides a smooth feedback gradient across mild and severe deviations without collapsing difficult samples immediately to zero.",
        ),
    ]
    profile_sections = []
    for key in ("standard", "mild_deviation", "severe_deviation", "improving_sequence"):
        item = cold_start_profiles[key]
        step_rows = [
            [
                str(step["step"]),
                str(step["score"]),
                f"{step['raw']:.4f}",
                f"{step['distance']:.3f}",
                f"{step['tau']:.2f}",
                escape(step["predicted"]),
                f"{step['blend']:.2f}",
            ]
            for step in item["steps"]
        ]
        if key == "standard":
            summary = "The standard sample uses the target descriptor mean vector to check whether the ideal pose receives a stable high score."
        elif key == "mild_deviation":
            summary = "The mild-deviation sample shifts slightly toward the most confusable direction to test whether the system still recognizes the target while providing corrective feedback."
        elif key == "severe_deviation":
            summary = "The severe-deviation sample shifts further toward a competing class to test whether cold-start scoring lowers the score and exposes the error pattern."
        else:
            summary = "The improving sequence moves from severe deviation toward the standard pose, showing how scores, predictions, and blend weight evolve over simulated practice."
        profile_sections.append(
            f"""
            <section class="section">
              <h3>{escape(item['name'])}</h3>
              <p>{escape(summary)}</p>
              {render_table(["Step", "Score", "Raw Score", "Distance", "Tau", "Predicted Gesture", "Blend"], step_rows)}
            </section>
            """
        )
    content = (
        metrics
        + '<section class="section"><h2>Stage Objective and Work Completed</h2><p>Phase 2 asks whether the system can produce a useful score on first use. It combines population descriptors, nearest-descriptor prediction, joint-level feedback, and gradual transition toward a simulated user model.</p></section>'
        + '<section class="section"><h2>Cold-Start Profile Summary</h2>'
        + render_table(["Profile", "Steps", "Score trace", "Prediction trace"], profile_rows)
        + "</section>"
        + '<section class="section"><h2>Stage Conclusion</h2><p>Across standard, deviated, and improving profiles, cold-start scoring produces a coherent score gradient and prediction trace. In the improving sequence, the system identifies early deviation, returns to the target gesture later, and increases the simulated user-model blend weight as more observations accumulate.</p></section>'
        + "".join(render_figure_card(card) for card in figures)
        + "".join(profile_sections)
    )
    return wrap_page(
        title=PHASE_TITLES[2],
        intro="This page focuses on calibration-free cold-start scoring and interpretable feedback.",
        content=content,
        current_page="phase2.html",
    )


def build_phase3_page() -> str:
    metrics = render_metrics(
        [
            ("Initial tau", "5.0"),
            ("Core objective", "Maintain the challenge zone"),
            ("Simulated curves", "4"),
            ("Main figures", "3"),
        ]
    )
    figures = [
        FigureCard(
            "A4_simulated_trajectories.png",
            "Dynamic Threshold Simulation Across Recovery Curves",
            "This figure combines score trajectories, tau evolution, and progression indicators to inspect automatic difficulty adjustment.",
            "Linear, S-curve, plateau, and regression-like trajectories were simulated, then the threshold manager adjusted tau round by round.",
            "A useful controller tightens the threshold during improvement and relaxes it during fatigue or regression, keeping training challenging without becoming punitive.",
        ),
        FigureCard(
            "A5_tau_sensitivity.png",
            "Tau Sensitivity Analysis",
            "This figure shows how mean display score changes with tau for different simulated ability levels.",
            "Several virtual ability levels were passed through the same scoring function while sweeping tau.",
            "Very small tau can collapse scores for lower-ability users, while very large tau can reduce discrimination among high-ability users.",
        ),
        FigureCard(
            "A3_scoring_functions.png",
            "Scoring Function Reference",
            "This figure also supports Phase 3 because dynamic thresholding acts through the same score-mapping function.",
            "Phase 3 keeps the scoring formula shape and adapts its temperature parameter over simulated practice.",
            "The mechanism changes scoring strictness, not the scoring rule family.",
        ),
    ]
    content = (
        metrics
        + '<section class="section"><h2>Stage Objective and Work Completed</h2><p>Phase 3 makes scoring strictness adaptive. Instead of using a fixed ruler, the system updates tau with momentum so that improving performance tightens scoring while plateau or fatigue can slow or reverse strictness increases.</p></section>'
        + '<section class="section"><h2>Stage Conclusion</h2><p>The simulations show differentiated responses: linear and S-curve recovery tighten tau over time, plateau behavior stabilizes, and fatigue dips trigger protective relaxation. This supports the goal of performance-aware difficulty adjustment.</p></section>'
        + "".join(render_figure_card(card) for card in figures)
    )
    return wrap_page(
        title=PHASE_TITLES[3],
        intro="This page explains how static scoring is upgraded into an adaptive threshold mechanism.",
        content=content,
        current_page="phase3.html",
    )


def build_phase4_page(phase4_reports: dict) -> str:
    rows = []
    for scenario in SCENARIO_ORDER:
        report = phase4_reports[scenario]
        rows.append(
            [
                escape(SCENARIO_LABELS[scenario]),
                f"{report['progression_index']:.3f}",
                f"{report['recovery_score']:.1f}",
                f"{report['recovery_velocity']:.4f}",
                escape("; ".join(report["warnings"]) if report["warnings"] else "No warnings"),
            ]
        )
    summary_table = render_table(
        ["Scenario", "progression_index", "recovery_score", "recovery_velocity", "Warnings"],
        rows,
    )
    figure_html = []
    representative_scenarios = ["linear_recovery", "fatigue_dip"]
    for scenario in representative_scenarios:
        title_prefix = SCENARIO_LABELS[scenario]
        report = phase4_reports[scenario]
        figure_html.append(
            render_figure_card(
                FigureCard(
                    f"phase4_{scenario}_progress_curve.png",
                    f"{title_prefix}: Overall Progress Curve",
                    "This figure places daily progression index and recovery velocity in the same view.",
                    "Phase 4 reorganizes simulated Stage 2/3 histories into longitudinal indicators rather than re-scoring a single gesture attempt.",
                    f"The current simulated scenario has progression_index={report['progression_index']:.3f} and recovery_score={report['recovery_score']:.1f}. Plateaued curves or near-zero velocity indicate that the training plan may need adjustment.",
                )
            )
        )
        figure_html.append(
            render_figure_card(
                FigureCard(
                    f"phase4_{scenario}_joint_radar.png",
                    f"{title_prefix}: Joint Recovery Radar",
                    "This figure shows recovery ratios at the joint-feature level.",
                    "The simulated user feature mean is compared with the population descriptor mean to estimate relative recovery by joint.",
                    f"The recommendation for this scenario is: {report['recommendation']}. It should align with the weakest regions in the radar chart.",
                )
            )
        )
        figure_html.append(
            render_figure_card(
                FigureCard(
                    f"phase4_{scenario}_tau_heatmap.png",
                    f"{title_prefix}: Tau Evolution Heatmap",
                    "This figure shows tau changes for target gestures over the simulated time span.",
                    "Tau values recorded at the end of simulated rounds are arranged by time and gesture.",
                    "Persistently high tau suggests a gesture still needs a more tolerant scoring standard; sustained tightening suggests more stable simulated recovery.",
                )
            )
        )
    content = (
        render_metrics(
            [
                ("Scenarios", "4"),
                ("Days per scenario", "30"),
                ("Figures per scenario", "3"),
                ("Displayed scenarios", "2 representative"),
            ]
        )
        + '<section class="section"><h2>Stage Objective and Work Completed</h2><p>Phase 4 asks how to evaluate simulated rehabilitation progress without using real patient data. The report does not revisit raw images or consume patient logs; it reverse-infers progress from simulated formula variables such as score, tau, baseline, momentum, challenge-zone state, and user mean. The table summarizes all four simulated scenarios; figures show representative recovery and fatigue cases in an academic-report style.</p></section>'
        + '<section class="section"><h2>Scenario Summary</h2>'
        + summary_table
        + "</section>"
        + '<section class="section"><h2>Stage Conclusion</h2><p>Phase 4 turns simulated process variables into an interpretable progress report. It explains how changes in scoring-formula variables correspond to simulated progress trends, lagging joints, and training-plan adjustments.</p></section>'
        + "".join(figure_html)
    )
    return wrap_page(
        title=PHASE_TITLES[4],
        intro="This page focuses on simulated longitudinal progress assessment from formula-variable histories.",
        content=content,
        current_page="phase4.html",
    )


def build_phase5_page(summary: dict) -> str:
    strategy_rows = []
    for name, metrics in summary["strategy_metrics"].items():
        intervals = summary.get("confidence_intervals", {}).get(name, {})
        gain_ci = intervals.get("ability_gain", {})
        challenge_ci = intervals.get("challenge_ratio", {})
        feedback_ci = intervals.get("feedback_accuracy", {})
        strategy_rows.append(
            [
                escape(name),
                f"{metrics['ability_gain']:.4f}",
                f"[{gain_ci.get('low', 0.0):.4f}, {gain_ci.get('high', 0.0):.4f}]",
                f"{metrics['challenge_ratio']:.4f}",
                f"[{challenge_ci.get('low', 0.0):.4f}, {challenge_ci.get('high', 0.0):.4f}]",
                f"{metrics['feedback_accuracy']:.4f}",
                f"[{feedback_ci.get('low', 0.0):.4f}, {feedback_ci.get('high', 0.0):.4f}]",
                f"{metrics['frustration_risk']:.4f}",
                f"{metrics['score_ability_correlation']:.4f}",
                f"{metrics['scenario_variance']:.6f}",
            ]
        )
    effect_rows = [
        [
            escape(name),
            f"{value:.4f}",
            f"{summary.get('pairwise_permutation_tests', {}).get(name, {}).get('p_value', 1.0):.4f}",
            str(summary.get("pairwise_permutation_tests", {}).get(name, {}).get("n_pairs", "")),
        ]
        for name, value in summary["pairwise_effect_sizes"].items()
    ]
    definition_rows = [
        [escape(metric), escape(description)]
        for metric, description in summary.get("metric_definitions", {}).items()
    ]
    narrative = summary.get("narrative", {})
    key_findings = "".join(
        f"<li>{escape(item)}</li>" for item in narrative.get("key_findings", [])
    )
    figure_html = "".join(
        render_figure_card(card)
        for card in [
            FigureCard(
                "C1_phase5_strategy_metrics.png",
                "Phase 5 Strategy Metrics with 95% Intervals",
                "This figure compares ability gain, challenge-zone ratio, feedback accuracy, and frustration risk across three strategies.",
                "Bootstrap confidence intervals were estimated from virtual-subject simulation results and displayed as grouped bars.",
                "The adaptive strategy is strongest on challenge-zone ratio and feedback accuracy while maintaining the highest ability_gain.",
            ),
            FigureCard(
                "C2_phase5_scenario_heatmap.png",
                "Challenge-Zone Ratio by Recovery Curve",
                "This heatmap shows how well each strategy keeps training inside the effective challenge zone.",
                "challenge_ratio was aggregated by strategy and recovery curve.",
                "The adaptive strategy maintains more stable challenge-zone coverage across all simulated recovery curves.",
            ),
            FigureCard(
                "C3_phase5_effect_sizes.png",
                "Pairwise Effect Sizes and Permutation Tests",
                "This figure summarizes Cohen's d and paired permutation-test p-values for ability_gain.",
                "Ability-gain differences were paired by virtual subject and evaluated with a sign-flip permutation test.",
                "The adaptive strategy has positive effect sizes in the current simulation, but the result remains paper-prototype evidence rather than clinical evidence.",
            ),
        ]
    )
    content = (
        render_metrics(
            [
                ("Strategies", "3"),
                ("Recovery curves", "4"),
                ("Statistics", "6 + CI + permutation"),
                ("Best aggregate strategy", "adaptive"),
            ]
        )
        + '<section class="section"><h2>Stage Objective and Work Completed</h2><p>Phase 5 evaluates whether the adaptive scoring logic is preferable to simpler alternatives. Virtual subjects, recovery curves, and three scoring strategies are compared in the same simulated environment. No real patient data are used.</p></section>'
        + '<section class="section"><h2>Strategy Metric Summary</h2>'
        + render_table(
            [
                "Strategy",
                "ability_gain",
                "ability_gain 95% CI",
                "challenge_ratio",
                "challenge 95% CI",
                "feedback_accuracy",
                "feedback 95% CI",
                "frustration_risk",
                "score_ability_correlation",
                "scenario_variance",
            ],
            strategy_rows,
        )
        + "</section>"
        + '<section class="section"><h2>Metric Definitions</h2>'
        + render_table(["Metric", "Definition"], definition_rows)
        + "</section>"
        + '<section class="section"><h2>Effect Size Summary</h2>'
        + render_table(["Strategy pair", "effect size", "permutation p", "paired n"], effect_rows)
        + "</section>"
        + '<section class="section"><h2>Relation to Phase 4</h2>'
        + f"<p>{escape(str(narrative.get('phase4_relation', '')))}</p>"
        + f"<ul>{key_findings}</ul>"
        + f"<p>{escape(str(narrative.get('interpretation_boundary', '')))}</p>"
        + "</section>"
        + figure_html
        + '<section class="section"><h2>Stage Conclusion</h2><p>Phase 5 provides a strategy-comparison framework with confidence intervals, paired permutation tests, and paper-ready figures. It turns the prototype from a runnable pipeline into an experimentally comparable system.</p></section>'
    )
    return wrap_page(
        title=PHASE_TITLES[5],
        intro="This page compares three scoring strategies in the same simulated rehabilitation setting. All subjects are virtual.",
        content=content,
        current_page="phase5.html",
    )


def write_styles() -> None:
    css = """
    :root {
      --bg: #f4efe6;
      --card: #fffdf9;
      --ink: #1f2b24;
      --muted: #5f6d63;
      --line: #ddd4c5;
      --accent: #2f7a67;
      --accent-soft: #e2f0eb;
      --shadow: 0 10px 28px rgba(45, 52, 47, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #f8f3ea 0%, #f1ebe2 100%);
      color: var(--ink);
      font-family: Inter, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      line-height: 1.7;
    }
    .page-shell {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 56px;
    }
    .hero {
      background: linear-gradient(135deg, #fff8ee, #edf6f1);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 28px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }
    .eyebrow {
      margin: 0 0 8px;
      color: var(--accent);
      letter-spacing: 0.08em;
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 700;
    }
    h1, h2, h3 { margin-top: 0; line-height: 1.25; }
    .intro { margin: 0; color: var(--muted); max-width: 900px; }
    .nav {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }
    .nav a {
      text-decoration: none;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.84);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 14px;
    }
    .nav a.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }
    .content {
      display: grid;
      gap: 18px;
    }
    .final-shell {
      max-width: 1240px;
    }
    .section, .figure-card, .phase-link {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: var(--shadow);
    }
    .phase-link h2 a {
      color: var(--ink);
      text-decoration: none;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }
    .metric-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }
    .metric-label { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .metric-value { font-size: 28px; font-weight: 700; }
    .table-wrap { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: white;
      border-radius: 14px;
      overflow: hidden;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid #ece4d8;
      text-align: left;
      vertical-align: top;
    }
    th { background: #f7f1e7; }
    img {
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      display: block;
      margin: 14px 0 16px;
      background: white;
    }
    .figure-notes p {
      margin: 0 0 10px;
    }
    .final-toc {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin: 12px 0 10px;
    }
    .final-toc a {
      display: block;
      padding: 12px 14px;
      border-radius: 14px;
      text-decoration: none;
      color: var(--ink);
      background: #f7f1e7;
      border: 1px solid var(--line);
    }
    .final-phase-section {
      display: grid;
      gap: 18px;
      padding-top: 10px;
    }
    .final-phase-header {
      background: linear-gradient(135deg, #f3fbf7, #fff8ee);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: var(--shadow);
    }
    .phase-kicker {
      margin: 0 0 8px;
      font-size: 13px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }
    pre {
      white-space: pre-wrap;
      background: #f8f4ec;
      border: 1px solid #eadfcd;
      border-radius: 14px;
      padding: 16px;
      overflow-x: auto;
      font-size: 13px;
      line-height: 1.55;
    }
    strong { color: #1d5143; }
    @media print {
      body {
        background: white;
      }
      .page-shell {
        max-width: none;
        padding: 0;
      }
      .hero, .section, .figure-card, .phase-link, .metric-card, .final-phase-header {
        box-shadow: none;
      }
      .nav {
        display: none;
      }
      .figure-card, .section, .metric-card, .final-phase-section {
        break-inside: avoid;
      }
      img {
        max-height: 88vh;
        object-fit: contain;
      }
    }
    """
    (REPORT_DIR / "styles.css").write_text(css, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    phase1_summary = load_build_descriptor_summary()
    cold_start_profiles = load_cold_start_profiles()
    phase4_reports = load_phase4_reports()
    phase5_summary = read_json(PHASE5_SUMMARY)

    write_styles()
    phase_pages = {
        1: build_phase1_page(phase1_summary),
        2: build_phase2_page(cold_start_profiles),
        3: build_phase3_page(),
        4: build_phase4_page(phase4_reports),
        5: build_phase5_page(phase5_summary),
    }
    (REPORT_DIR / "index.html").write_text(build_index_page(), encoding="utf-8")
    for phase, html in phase_pages.items():
        (REPORT_DIR / f"phase{phase}.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "final_report.html").write_text(build_final_report_page(phase_pages), encoding="utf-8")

    print(f"Stage HTML report generated at: {REPORT_DIR}")
    for path in sorted(REPORT_DIR.glob("*.html")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
