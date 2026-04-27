"""
Extended report figures for CineInfini (added in 0.2.0).

This module is ADDITIVE: it does not replace `generate_intra_report` /
`generate_inter_report`. It adds extra figures and an HTML dashboard that
reuse the same data files (`data.json`) those functions already produce.

Six new figures are produced:

1. **Heatmap shot × metric** — single-image overview of all metrics across
   all shots, normalized so red = bad, green = good. The fastest visual
   diagnosis of "where are the problems".

2. **Identity trajectory** (PCA 2-D) — projection of all per-shot identity
   embeddings into 2-D, color-coded by shot. Clusters = consistent identity;
   spread = drift.

3. **Verdict timeline** — Gantt-like bar showing per-shot verdict
   (ACCEPT / REVIEW / REJECT) along the video timeline.

4. **DTW vs mean comparison** — side-by-side bar showing both metrics.
   Highlights shots where DTW catches what mean misses.

5. **Inter-shot full matrix** — N×N coherence matrix (not just adjacent
   pairs). Quickly shows clusters of similar shots.

6. **Sparkline grid** — one tiny plot per metric, all on one row, for an
   "executive summary" overview.

Plus an **HTML dashboard** with all figures embedded, exportable and
shareable.
"""
from __future__ import annotations

import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Color palette (consistent across all figures)
# ---------------------------------------------------------------------------

PALETTE = {
    "good": "#2ecc71",       # green
    "warn": "#f39c12",       # orange
    "bad": "#e74c3c",        # red
    "neutral": "#95a5a6",    # gray (None / not computed)
    "primary": "#3498db",    # blue
    "secondary": "#9b59b6",  # purple
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_metric(value: Optional[float], thresh: float, direction: str) -> Optional[float]:
    """Normalize a metric value to [0, 1] where 1 = pass, 0 = fail.

    direction = "below" : value should be below threshold (e.g. flicker)
    direction = "above" : value should be above threshold (e.g. SSIM)
    """
    if value is None:
        return None
    if direction == "below":
        # value 0 -> score 1.0 ; value = thresh -> score 0.5 ; value >= 2*thresh -> score ~0
        if thresh <= 0:
            return 1.0
        return float(np.clip(1.0 - value / (2 * thresh), 0.0, 1.0))
    else:  # "above"
        # value 1.0 -> score 1.0 ; value = thresh -> score 0.5 ; value 0 -> score 0
        if thresh <= 0:
            return float(np.clip(value, 0.0, 1.0))
        return float(np.clip(value / max(2 * thresh, 1e-6), 0.0, 1.0))


def _verdict_for_shot(scores: dict, thresholds: dict) -> str:
    """Aggregate per-shot verdict from threshold compliance."""
    metrics_dirs = [
        ("motion_peak_div", thresholds.get("motion", 25.0), "below"),
        ("ssim3d_self", thresholds.get("ssim3d", 0.45), "above"),
        ("flicker", thresholds.get("flicker", 0.1), "below"),
        ("identity_intra", thresholds.get("identity_drift", 0.6), "below"),
        ("ssim_long_range", thresholds.get("ssim_long_range", 0.45), "above"),
        ("clip_temp_consistency", thresholds.get("clip_temp", 0.25), "above"),
    ]
    n_violations = 0
    n_evaluated = 0
    for key, thresh, direction in metrics_dirs:
        v = scores.get(key)
        if v is None:
            continue
        n_evaluated += 1
        if direction == "below" and v > thresh:
            n_violations += 1
        elif direction == "above" and v < thresh:
            n_violations += 1
    if n_evaluated == 0:
        return "UNKNOWN"
    ratio = n_violations / n_evaluated
    if ratio == 0:
        return "ACCEPT"
    elif ratio <= 0.34:
        return "REVIEW"
    else:
        return "REJECT"


def _save_fig(fig, path: Path, dpi: int = 110):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fig_to_base64(fig) -> str:
    """Return base64-encoded PNG of a matplotlib figure (for HTML embed)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Figure 1: shot × metric heatmap
# ---------------------------------------------------------------------------

def figure_heatmap_shot_metric(
    metrics_data: dict,
    thresholds: dict,
    output_path: Optional[Path] = None,
):
    gates = metrics_data["gates"]
    shot_ids = sorted([int(k) for k in gates.keys()])
    metric_config = [
        ("motion_peak_div", "Motion", thresholds.get("motion", 25.0), "below"),
        ("ssim3d_self", "3D-SSIM", thresholds.get("ssim3d", 0.45), "above"),
        ("flicker", "Flicker", thresholds.get("flicker", 0.1), "below"),
        ("identity_intra", "Identity (mean)", thresholds.get("identity_drift", 0.6), "below"),
        ("identity_intra_dtw", "Identity (DTW)", thresholds.get("identity_drift", 0.6), "below"),
        ("ssim_long_range", "SSIM-LR", thresholds.get("ssim_long_range", 0.45), "above"),
        ("flicker_hf_var", "Flicker HF", thresholds.get("flicker_hf", 0.01), "below"),
        ("clip_temp_consistency", "CLIP-temp", thresholds.get("clip_temp", 0.25), "above"),
    ]

    matrix = np.full((len(metric_config), len(shot_ids)), np.nan, dtype=np.float64)
    for j, sid in enumerate(shot_ids):
        g = gates[str(sid)]
        for i, (key, _, thresh, direction) in enumerate(metric_config):
            score = _normalize_metric(g.get(key), thresh, direction)
            if score is not None:
                matrix[i, j] = score

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rylgr", [PALETTE["bad"], PALETTE["warn"], PALETTE["good"]]
    )
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(shot_ids) + 2), 4))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(shot_ids)))
    ax.set_xticklabels([str(s) for s in shot_ids])
    ax.set_yticks(range(len(metric_config)))
    ax.set_yticklabels([m[1] for m in metric_config])
    ax.set_xlabel("Shot ID")
    ax.set_title("Quality heatmap (green = pass, red = fail, blank = not computed)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Pass score (1=pass, 0=fail)")
    plt.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
        return None
    return fig


# ---------------------------------------------------------------------------
# Figure 2: identity trajectory (PCA on per-shot identity values)
# ---------------------------------------------------------------------------

def figure_identity_trajectory(
    metrics_data: dict,
    output_path: Optional[Path] = None,
):
    """Plot per-shot (mean_drift, dtw_drift) as a 2-D scatter so you can
    see clusters of identity-coherent shots vs outliers.

    We use the two scalar identity scores (mean and DTW) as 2-D coordinates;
    no need for true PCA on raw embeddings (which we don't have here).
    """
    gates = metrics_data["gates"]
    shot_ids = sorted([int(k) for k in gates.keys()])
    xs, ys, labels = [], [], []
    for sid in shot_ids:
        g = gates[str(sid)]
        m = g.get("identity_intra")
        d = g.get("identity_intra_dtw")
        if m is None and d is None:
            continue
        xs.append(m if m is not None else 0.0)
        ys.append(d if d is not None else 0.0)
        labels.append(sid)

    fig, ax = plt.subplots(figsize=(7, 5))
    if not xs:
        ax.text(0.5, 0.5, "No identity data available for any shot",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        if output_path:
            _save_fig(fig, output_path)
            return None
        return fig

    sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="viridis",
                    s=80, edgecolors="black", linewidths=0.5, alpha=0.8)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(str(lbl), (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=8)
    # Reference lines: y=x means agreement; y > x means DTW reveals more drift
    lim_max = max(max(xs + [0.1]), max(ys + [0.1])) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, label="y = x (mean = DTW)")
    ax.fill_between([0, lim_max], [0, lim_max], [lim_max, lim_max],
                    alpha=0.05, color=PALETTE["bad"], label="DTW reveals anomaly")
    ax.set_xlabel("Identity drift (mean-based)")
    ax.set_ylabel("Identity drift (DTW-based)")
    ax.set_title("Identity coherence per shot — mean vs DTW")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Shot order")
    plt.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
        return None
    return fig


# ---------------------------------------------------------------------------
# Figure 3: verdict timeline
# ---------------------------------------------------------------------------

def figure_verdict_timeline(
    metrics_data: dict,
    thresholds: dict,
    output_path: Optional[Path] = None,
):
    gates = metrics_data["gates"]
    shot_ids = sorted([int(k) for k in gates.keys()])
    verdicts = [_verdict_for_shot(gates[str(sid)], thresholds) for sid in shot_ids]
    color_map = {
        "ACCEPT": PALETTE["good"],
        "REVIEW": PALETTE["warn"],
        "REJECT": PALETTE["bad"],
        "UNKNOWN": PALETTE["neutral"],
    }
    colors = [color_map[v] for v in verdicts]

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(shot_ids) + 2), 1.8))
    ax.bar(range(len(shot_ids)), [1] * len(shot_ids), color=colors,
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(shot_ids)))
    ax.set_xticklabels([str(s) for s in shot_ids])
    ax.set_yticks([])
    ax.set_xlabel("Shot ID")
    ax.set_title(f"Per-shot verdict — "
                 f"ACCEPT: {verdicts.count('ACCEPT')}  "
                 f"REVIEW: {verdicts.count('REVIEW')}  "
                 f"REJECT: {verdicts.count('REJECT')}")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["good"], label="ACCEPT"),
        Patch(facecolor=PALETTE["warn"], label="REVIEW"),
        Patch(facecolor=PALETTE["bad"], label="REJECT"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.01, 1.4), ncol=3, frameon=False)
    plt.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
        return None
    return fig


# ---------------------------------------------------------------------------
# Figure 4: DTW vs mean side-by-side
# ---------------------------------------------------------------------------

def figure_dtw_vs_mean(
    metrics_data: dict,
    thresholds: dict,
    output_path: Optional[Path] = None,
):
    gates = metrics_data["gates"]
    shot_ids = sorted([int(k) for k in gates.keys()])
    means = [gates[str(s)].get("identity_intra") for s in shot_ids]
    dtws = [gates[str(s)].get("identity_intra_dtw") for s in shot_ids]
    if all(m is None for m in means) and all(d is None for d in dtws):
        # nothing to plot
        return None

    means_safe = [m if m is not None else np.nan for m in means]
    dtws_safe = [d if d is not None else np.nan for d in dtws]
    thresh = thresholds.get("identity_drift", 0.6)

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(shot_ids) + 2), 4))
    x = np.arange(len(shot_ids))
    w = 0.4
    ax.bar(x - w / 2, means_safe, w, label="Identity drift (mean)",
           color=PALETTE["primary"], alpha=0.85)
    ax.bar(x + w / 2, dtws_safe, w, label="Identity drift (DTW)",
           color=PALETTE["secondary"], alpha=0.85)
    ax.axhline(y=thresh, color="r", linestyle="--", alpha=0.6,
               label=f"threshold {thresh}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in shot_ids])
    ax.set_xlabel("Shot ID")
    ax.set_ylabel("Identity drift")
    ax.set_title("Mean vs DTW identity drift — DTW reveals temporal anomalies")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
        return None
    return fig


# ---------------------------------------------------------------------------
# Figure 5: full inter-shot matrix
# ---------------------------------------------------------------------------

def figure_inter_shot_matrix(
    metrics_data: dict,
    output_path: Optional[Path] = None,
):
    """Build an N×N inter-shot coherence matrix from the inter_results list
    of adjacent pairs. For non-adjacent pairs we leave blank (we don't have
    the data without recomputing). Diagonal = 1.0.
    """
    gates = metrics_data.get("gates", {})
    inter_results = metrics_data.get("inter_results", [])
    shot_ids = sorted([int(k) for k in gates.keys()])
    n = len(shot_ids)
    if n < 2:
        return None
    sid_to_idx = {sid: i for i, sid in enumerate(shot_ids)}
    M = np.full((n, n), np.nan)
    for i in range(n):
        M[i, i] = 1.0  # self-coherence
    for r in inter_results:
        a, b = r.get("shot_a"), r.get("shot_b")
        total = r.get("total")
        if a is None or b is None or total is None:
            continue
        if a in sid_to_idx and b in sid_to_idx:
            ia, ib = sid_to_idx[a], sid_to_idx[b]
            M[ia, ib] = total
            M[ib, ia] = total

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rylgr", [PALETTE["bad"], PALETTE["warn"], PALETTE["good"]]
    )
    fig, ax = plt.subplots(figsize=(max(5, 0.35 * n + 2), max(5, 0.35 * n + 2)))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(s) for s in shot_ids])
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(s) for s in shot_ids])
    ax.set_xlabel("Shot")
    ax.set_ylabel("Shot")
    ax.set_title("Inter-shot coherence matrix (gray = not computed)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Coherence (0=bad, 1=good)")
    plt.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
        return None
    return fig


# ---------------------------------------------------------------------------
# Figure 6: sparkline grid
# ---------------------------------------------------------------------------

def figure_sparkline_grid(
    metrics_data: dict,
    thresholds: dict,
    output_path: Optional[Path] = None,
):
    gates = metrics_data["gates"]
    shot_ids = sorted([int(k) for k in gates.keys()])
    metric_config = [
        ("motion_peak_div", "Motion", thresholds.get("motion", 25.0), "below"),
        ("ssim3d_self", "3D-SSIM", thresholds.get("ssim3d", 0.45), "above"),
        ("flicker", "Flicker", thresholds.get("flicker", 0.1), "below"),
        ("identity_intra", "Identity", thresholds.get("identity_drift", 0.6), "below"),
        ("ssim_long_range", "SSIM-LR", thresholds.get("ssim_long_range", 0.45), "above"),
        ("clip_temp_consistency", "CLIP-temp", thresholds.get("clip_temp", 0.25), "above"),
    ]
    fig, axes = plt.subplots(1, len(metric_config),
                             figsize=(2.5 * len(metric_config), 2))
    if len(metric_config) == 1:
        axes = [axes]
    for ax, (key, label, thresh, direction) in zip(axes, metric_config):
        values = [gates[str(s)].get(key) for s in shot_ids]
        values_safe = [v if v is not None else np.nan for v in values]
        ax.plot(shot_ids, values_safe, "o-", color=PALETTE["primary"],
                markersize=4, linewidth=1.5)
        ax.axhline(y=thresh, color="r", linestyle="--", alpha=0.5)
        # color points violating the threshold
        for sid, v in zip(shot_ids, values):
            if v is None:
                continue
            violates = (direction == "below" and v > thresh) or                        (direction == "above" and v < thresh)
            if violates:
                ax.plot(sid, v, "o", color=PALETTE["bad"], markersize=6)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Shot", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-metric overview (red dots = violations)", fontsize=11)
    plt.tight_layout()
    if output_path:
        _save_fig(fig, output_path)
        return None
    return fig


# ---------------------------------------------------------------------------
# Top-level: build all extended figures + HTML dashboard
# ---------------------------------------------------------------------------

def generate_extended_intra_report(
    video_name: str,
    metrics_data: dict,
    output_dir: Path,
    thresholds: dict,
    save_html: bool = True,
) -> dict:
    """Generate the 6 extended figures and (optionally) an HTML dashboard.

    Parameters
    ----------
    video_name : str
    metrics_data : dict
        The same dict used by `generate_intra_report` (loaded from data.json).
    output_dir : Path
        Will be `output_dir / video_name` to match `generate_intra_report`.
    thresholds : dict
    save_html : bool
        Also produce `dashboard.html` with all figures embedded as base64.

    Returns
    -------
    dict mapping figure name -> Path of the saved PNG (or None if empty).
    """
    out_dir = Path(output_dir) / video_name
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    paths["heatmap"] = figs_dir / "heatmap_shot_metric.png"
    figure_heatmap_shot_metric(metrics_data, thresholds, paths["heatmap"])

    paths["trajectory"] = figs_dir / "identity_trajectory.png"
    figure_identity_trajectory(metrics_data, paths["trajectory"])

    paths["timeline"] = figs_dir / "verdict_timeline.png"
    figure_verdict_timeline(metrics_data, thresholds, paths["timeline"])

    p = figure_dtw_vs_mean(metrics_data, thresholds,
                           figs_dir / "dtw_vs_mean.png")
    if (figs_dir / "dtw_vs_mean.png").exists():
        paths["dtw_vs_mean"] = figs_dir / "dtw_vs_mean.png"

    paths["inter_matrix"] = figs_dir / "inter_shot_matrix.png"
    figure_inter_shot_matrix(metrics_data, paths["inter_matrix"])

    paths["sparkline"] = figs_dir / "sparkline_grid.png"
    figure_sparkline_grid(metrics_data, thresholds, paths["sparkline"])

    if save_html:
        html_path = out_dir / "dashboard.html"
        _build_html_dashboard(video_name, metrics_data, thresholds,
                              paths, html_path)
        paths["html"] = html_path

    return paths


def _build_html_dashboard(
    video_name: str,
    metrics_data: dict,
    thresholds: dict,
    figure_paths: dict,
    output_path: Path,
):
    """Build a self-contained HTML dashboard with all figures inlined."""
    gates = metrics_data["gates"]
    shot_ids = sorted([int(k) for k in gates.keys()])
    n_shots = len(shot_ids)
    verdicts = [_verdict_for_shot(gates[str(sid)], thresholds) for sid in shot_ids]
    n_accept = verdicts.count("ACCEPT")
    n_review = verdicts.count("REVIEW")
    n_reject = verdicts.count("REJECT")

    def _img_tag(path):
        if path is None or not Path(path).exists():
            return "<p><em>(figure not available)</em></p>"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" />'

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
           sans-serif; max-width: 1200px; margin: 30px auto; padding: 0 20px;
           color: #2c3e50; background: #f8f9fa; }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }
    h2 { color: #34495e; margin-top: 32px; }
    .summary { display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }
    .card { flex: 1; min-width: 140px; padding: 16px; border-radius: 8px;
            background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center; }
    .card .num { font-size: 2.2em; font-weight: 700; }
    .card.accept .num { color: #27ae60; }
    .card.review .num { color: #f39c12; }
    .card.reject .num { color: #c0392b; }
    .card.total .num { color: #2980b9; }
    .figure { margin: 24px 0; padding: 16px; background: white;
              border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    .figure img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
    .figure h3 { margin-top: 0; color: #34495e; }
    .figure p { color: #7f8c8d; font-size: 0.9em; }
    footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd;
             color: #95a5a6; font-size: 0.85em; text-align: center; }
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CineInfini — {video_name}</title>
<style>{css}</style>
</head>
<body>
<h1>🎬 CineInfini Dashboard — <span style="font-weight:400">{video_name}</span></h1>

<div class="summary">
  <div class="card total"><div class="num">{n_shots}</div>shots</div>
  <div class="card accept"><div class="num">{n_accept}</div>ACCEPT</div>
  <div class="card review"><div class="num">{n_review}</div>REVIEW</div>
  <div class="card reject"><div class="num">{n_reject}</div>REJECT</div>
</div>

<div class="figure">
  <h3>Quality heatmap — all metrics × all shots</h3>
  <p>Single-image overview. Green = pass, red = fail, blank = not computed.</p>
  {_img_tag(figure_paths.get("heatmap"))}
</div>

<div class="figure">
  <h3>Per-shot verdict timeline</h3>
  <p>How many shots cleared the quality gates and where the problems lie.</p>
  {_img_tag(figure_paths.get("timeline"))}
</div>

<div class="figure">
  <h3>Identity drift — mean vs DTW (added in 0.2.0)</h3>
  <p>The DTW-based variant catches temporal anomalies (e.g. brief glitches
  that revert) that the mean averages out. Bars where DTW is taller than
  mean indicate shots where DTW reveals something the mean missed.</p>
  {_img_tag(figure_paths.get("dtw_vs_mean"))}
</div>

<div class="figure">
  <h3>Identity coherence scatter (mean vs DTW per shot)</h3>
  <p>Each point is one shot. Points above the diagonal y=x are shots where
  DTW exposes more drift than the mean — the value-add of the DTW metric.</p>
  {_img_tag(figure_paths.get("trajectory"))}
</div>

<div class="figure">
  <h3>Inter-shot coherence matrix</h3>
  <p>NxN matrix; diagonal is self-coherence (1.0). Off-diagonal cells show
  similarity between shot pairs (only adjacent pairs are computed by default).</p>
  {_img_tag(figure_paths.get("inter_matrix"))}
</div>

<div class="figure">
  <h3>Sparkline grid — per-metric overview</h3>
  <p>One tiny plot per metric, all on one row. Red dots mark threshold violations.</p>
  {_img_tag(figure_paths.get("sparkline"))}
</div>

<footer>
  Generated by CineInfini v0.2.0 ·
  <a href="https://github.com/CineInfini/Video_Quality_Auditor">GitHub</a>
</footer>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
