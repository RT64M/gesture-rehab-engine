"""Plot English gesture mapping figures for the paper/report package."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import collect_landmarks_for_gesture, find_annotation_file
from src.descriptors import GestureDescriptor
from src.geometry import normalize_landmarks, pca_align_2d


# MediaPipe 手部连接关系（用于画骨架）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
]
plt.rcParams["axes.unicode_minus"] = False

GESTURE_DISPLAY = config.GESTURE_CN
COMPACT_GESTURE_LABELS = {}


def _prepare_coords(landmarks: np.ndarray) -> np.ndarray:
    """Convert raw landmarks (N, 21, 2) to normalized/aligned coordinates."""
    coords = []
    for lm in landmarks:
        pts = normalize_landmarks(lm)

        # Canonicalize handedness (left/right) before averaging.
        # If not unified, mirrored samples can cancel x-structure and look like a line.
        wrist = pts[config.WRIST, :2]
        index_mcp = pts[config.INDEX_MCP, :2]
        pinky_mcp = pts[config.PINKY_MCP, :2]
        cross_z = (index_mcp[0] - wrist[0]) * (pinky_mcp[1] - wrist[1]) - (
            index_mcp[1] - wrist[1]
        ) * (pinky_mcp[0] - wrist[0])
        if cross_z < 0:
            pts[:, 0] *= -1.0

        pts = pca_align_2d(pts)
        coords.append(pts[:, :2])
    return np.asarray(coords, dtype=np.float64)


def _prepare_coords_no_mirror(landmarks: np.ndarray) -> np.ndarray:
    """Convert raw landmarks to normalized/aligned coordinates without mirroring."""
    coords = []
    for lm in landmarks:
        pts = normalize_landmarks(lm)
        pts = pca_align_2d(pts)
        coords.append(pts[:, :2])
    return np.asarray(coords, dtype=np.float64)


def _handedness_sign(landmark: np.ndarray) -> float:
    """Signed cross product at wrist using index_mcp and pinky_mcp."""
    pts = normalize_landmarks(landmark)
    wrist = pts[config.WRIST, :2]
    index_mcp = pts[config.INDEX_MCP, :2]
    pinky_mcp = pts[config.PINKY_MCP, :2]
    return (index_mcp[0] - wrist[0]) * (pinky_mcp[1] - wrist[1]) - (
        index_mcp[1] - wrist[1]
    ) * (pinky_mcp[0] - wrist[0])


def _descriptor_expression_text(
    gesture: str,
    descriptor: GestureDescriptor,
    global_mu: np.ndarray,
    top_k: int = 6,
) -> str:
    """Generate descriptor-expression text for the side panel."""
    mu = descriptor.mu
    sigma_diag = np.diag(descriptor.sigma)
    std = np.sqrt(np.clip(sigma_diag, 1e-12, None))

    dev = np.abs(mu - global_mu)
    idx = np.argsort(dev)[-top_k:][::-1]

    lines = [
        f"Gesture: {GESTURE_DISPLAY.get(gesture, gesture)} ({gesture})",
        f"Samples: N={descriptor.n_samples}",
        "",
        "Statistical descriptor:",
        "D_g = (mu_g, Sigma_g)",
        "",
        f"Top-{top_k} expressive dimensions (mu ± sigma):",
    ]

    for i in idx:
        name = descriptor.feature_names[i]
        lines.append(f"- {name:18s} = {mu[i]:.3f} ± {std[i]:.3f}")

    tight_idx = np.argsort(std)[:3]
    lines += ["", "Most constrained dimensions (smallest sigma):"]
    for i in tight_idx:
        lines.append(f"- {descriptor.feature_names[i]}  (sigma={std[i]:.3f})")

    return "\n".join(lines)


def _plot_single_gesture(
    gesture: str,
    coords: np.ndarray,
    descriptor: GestureDescriptor,
    global_mu: np.ndarray,
    out_path: Path,
) -> None:
    """Render and save one gesture figure."""
    mean_coords = coords.mean(axis=0)

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    ax_map = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])

    # --- 左侧：平面映射 ---
    # 所有样本关键点散点（低透明度体现分布密度）
    xy = coords.reshape(-1, 2)
    ax_map.scatter(xy[:, 0], xy[:, 1], s=6, alpha=0.05, c="#3b82f6", linewidths=0)

    # 均值骨架
    for u, v in HAND_CONNECTIONS:
        ax_map.plot(
            [mean_coords[u, 0], mean_coords[v, 0]],
            [mean_coords[u, 1], mean_coords[v, 1]],
            c="#111827",
            lw=2.0,
            alpha=0.9,
        )

    # 均值关键点
    ax_map.scatter(mean_coords[:, 0], mean_coords[:, 1], s=32, c="#ef4444", zorder=5)
    for i, (x, y) in enumerate(mean_coords):
        ax_map.text(x + 0.015, y + 0.015, str(i), fontsize=8, color="#374151")

    ax_map.set_title(
        f"{GESTURE_DISPLAY.get(gesture, gesture)} ({gesture})\n"
        f"Normalized 21-point cloud and mean skeleton (N={len(coords)})",
        fontsize=13,
    )
    ax_map.set_xlabel("x (normalized)")
    ax_map.set_ylabel("y (normalized)")
    ax_map.grid(alpha=0.25)
    ax_map.set_aspect("equal", adjustable="box")

    x_min, x_max = np.percentile(xy[:, 0], [1, 99])
    y_min, y_max = np.percentile(xy[:, 1], [1, 99])
    pad_x = (x_max - x_min) * 0.18 + 0.1
    pad_y = (y_max - y_min) * 0.18 + 0.1
    ax_map.set_xlim(x_min - pad_x, x_max + pad_x)
    ax_map.set_ylim(y_min - pad_y, y_max + pad_y)

    # --- 右侧：最终标注表达 ---
    expr = _descriptor_expression_text(gesture, descriptor, global_mu, top_k=6)
    ax_txt.axis("off")
    ax_txt.text(
        0.0,
        1.0,
        expr,
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.4,
    )

    fig.suptitle("Gesture dataset distribution and final descriptor", fontsize=15, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gesture_grid(
    entries: list[tuple[str, np.ndarray, GestureDescriptor]],
    out_path: Path,
    title: str,
    ncols: int,
) -> None:
    """Render a compact paper-style grid for a gesture set."""
    if not entries:
        raise ValueError("Cannot render an empty gesture grid")

    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.72 * ncols, 2.92 * nrows),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)

    for ax, (gesture, coords, descriptor) in zip(axes_flat, entries):
        mean_coords = coords.mean(axis=0)
        xy = coords.reshape(-1, 2)
        ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.035, c="#2563eb", linewidths=0)

        for u, v in HAND_CONNECTIONS:
            ax.plot(
                [mean_coords[u, 0], mean_coords[v, 0]],
                [mean_coords[u, 1], mean_coords[v, 1]],
                c="#111827",
                lw=1.2,
                alpha=0.95,
            )

        ax.scatter(mean_coords[:, 0], mean_coords[:, 1], s=14, c="#dc2626", zorder=5)
        x_min, x_max = np.percentile(xy[:, 0], [1, 99])
        y_min, y_max = np.percentile(xy[:, 1], [1, 99])
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        half_span = max(x_max - x_min, y_max - y_min) * 0.58 + 0.07
        ax.set_xlim(x_center - half_span, x_center + half_span)
        ax.set_ylim(y_center - half_span, y_center + half_span)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.14)
        ax.set_title(
            f"{COMPACT_GESTURE_LABELS.get(gesture, GESTURE_DISPLAY.get(gesture, gesture))}\nN={descriptor.n_samples}",
            fontsize=8.6,
        )

    for ax in axes_flat[len(entries):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot normalized gesture mapping and descriptor expression")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=800,
        help="Maximum samples per gesture (default: 800)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACT_FIGURES_DIR,
        help="Output directory for supplemental single-gesture figures (default: artifacts/figures)",
    )
    parser.add_argument(
        "--grid-output-dir",
        type=Path,
        default=config.REPORT_FIGURES_DIR,
        help="Output directory for the paper mapping grid (default: report/figures)",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=None,
        help="Gesture keys to render. Default renders all target gestures.",
    )
    parser.add_argument(
        "--split-handedness",
        action="store_true",
        help="Also emit separate left/right-handed plots in addition to the canonical plot.",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also emit one compact paper-style grid figure for the selected gesture set.",
    )
    parser.add_argument(
        "--grid-name",
        default=None,
        help="Optional grid output filename. Defaults to gesture_mapping_grid.png.",
    )
    parser.add_argument("--show", action="store_true", help="Display figures after saving")
    args = parser.parse_args()

    print("=" * 72)
    print("Plotting: normalized gesture mapping + descriptor expression")
    print("=" * 72)

    descriptors: dict[str, GestureDescriptor] = {}
    for g in config.TARGET_GESTURES:
        p = config.DATA_PROCESSED_DIR / "descriptors" / f"{g}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing descriptor file: {p}")
        descriptors[g] = GestureDescriptor.load_json(p)

    global_mu = np.mean(np.stack([d.mu for d in descriptors.values()], axis=0), axis=0)

    gestures = list(args.gestures) if args.gestures is not None else list(config.TARGET_GESTURES)
    grid_entries: list[tuple[str, np.ndarray, GestureDescriptor]] = []

    for gesture in gestures:
        if gesture not in config.TARGET_GESTURES:
            raise KeyError(f"Unsupported gesture: {gesture}")

        json_path = find_annotation_file(config.DATA_RAW_DIR, gesture)
        if json_path is None:
            raise FileNotFoundError(f"Cannot find source annotation JSON for gesture: {gesture}")

        landmarks = collect_landmarks_for_gesture(
            json_path,
            target_label=gesture,
            max_samples=args.max_samples,
        )
        if len(landmarks) == 0:
            raise RuntimeError(f"No valid samples found for gesture: {gesture}")

        canonical_coords = _prepare_coords(np.asarray(landmarks))
        output_path = args.output_dir / f"{gesture}_mapping.png"
        _plot_single_gesture(
            gesture=gesture,
            coords=canonical_coords,
            descriptor=descriptors[gesture],
            global_mu=global_mu,
            out_path=output_path,
        )
        print(f"[DONE] canonical: {output_path}")
        grid_entries.append((gesture, canonical_coords, descriptors[gesture]))

        if args.split_handedness:
            left_samples = []
            right_samples = []
            for lm in landmarks:
                if _handedness_sign(lm) >= 0:
                    right_samples.append(lm)
                else:
                    left_samples.append(lm)

            if left_samples and right_samples:
                left_coords = _prepare_coords_no_mirror(np.asarray(left_samples))
                right_coords = _prepare_coords_no_mirror(np.asarray(right_samples))

                left_path = args.output_dir / f"{gesture}_left_mapping.png"
                right_path = args.output_dir / f"{gesture}_right_mapping.png"

                _plot_single_gesture(
                    gesture=gesture,
                    coords=left_coords,
                    descriptor=descriptors[gesture],
                    global_mu=global_mu,
                    out_path=left_path,
                )
                _plot_single_gesture(
                    gesture=gesture,
                    coords=right_coords,
                    descriptor=descriptors[gesture],
                    global_mu=global_mu,
                    out_path=right_path,
                )
                print(f"[DONE] left     : {left_path}")
                print(f"[DONE] right    : {right_path}")
            else:
                print(f"[skip] split-handedness unavailable for {gesture}: one side has zero samples")

    if args.grid:
        grid_name = args.grid_name or "gesture_mapping_grid.png"
        grid_title = "Ten-Gesture Mapping Overview"
        ncols = 5
        grid_path = args.grid_output_dir / grid_name
        _plot_gesture_grid(grid_entries, grid_path, grid_title, ncols=ncols)
        print(f"[DONE] grid     : {grid_path}")

    if args.show:
        plt.show()

    print("\nAll done.")


if __name__ == "__main__":
    main()
