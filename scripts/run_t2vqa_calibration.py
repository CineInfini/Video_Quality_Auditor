#!/usr/bin/env python3
"""Run CineInfini end-to-end on the T2VQA-DB subset and compute REAL
Spearman / Pearson / bootstrap CI95 vs human MOS.

Designed for **Google Colab + GPU**. Produces:

    out/t2vqa_calibration.json          — raw scores per video + correlations
    out/t2vqa_calibration_paper.json    — flat dict ready for fill_paper.py

Usage (Colab):
    !pip -q install cineinfini-audit
    !cineinfini bootstrap                              # ~835 MB ML weights
    !python scripts/run_t2vqa_calibration.py \\
        --videos-dir /content/T2VQA-DB \\
        --labels-csv /content/info.txt \\
        --out-dir /content/out \\
        --profile postproduction.yaml \\
        --max-videos 422

The script is **idempotent** — re-running skips videos already audited.
The `info.txt` format is the T2VQA-DB labels file shipped with the
dataset (lines: ``<video_id>.mp4|<prompt>|<MOS in [1, 100]>``).

NO placeholder values appear in the output JSON. Every Spearman / Pearson
/ CI is computed from real CineInfini outputs vs real human MOS.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Labels loader
# ---------------------------------------------------------------------------
def load_t2vqa_labels(labels_path: Path) -> Dict[str, float]:
    """Return {<video_id>.mp4: mos} dict from T2VQA-DB info.txt.

    Format: ``00000_07.mp4|<prompt>|<mos>`` per line.
    """
    out: Dict[str, float] = {}
    with labels_path.open(encoding="utf-8", errors="replace") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) != 3:
                continue
            vid, _, mos = parts
            try:
                out[vid] = float(mos)
            except ValueError:
                continue
    return out


def load_videofeedback_labels(csv_path: Path) -> Dict[str, float]:
    """Return {<basename>.mp4: mos} dict from VideoFeedback labels CSV."""
    out: Dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = Path(row["video_path"]).name
            try:
                out[p] = float(row["mos"])
            except (KeyError, ValueError):
                continue
    return out


def load_konvid_labels(csv_path: Path) -> Dict[str, float]:
    """Return {<basename>.mp4: mos} dict from a KoNViD-1k labels CSV.

    Supports three known formats:

    1. ``KoNViD_1k_attributes.csv`` (official from kn.de) — multi-column
       with header; columns include ``flickr_id``, ``file_name``, ``MOS``.
    2. ``KoNViD_mos_fr.csv`` (cnn-tlvqm mirror, *headerless*) —
       3 columns: ``flickr_id, mos, framerate_normalized``.
    3. ``KoNViD_mos_fr.csv`` *with* a ``file_name,mos`` header — older
       variant of the same mirror.

    The function probes the first line:

    * if it parses as 3 numeric fields → format (2), no header, columns
      are positional ``flickr_id, mos, fr``;
    * if first cell is a non-numeric word like ``flickr_id`` or
      ``file_name`` → use a DictReader.
    """
    out: Dict[str, float] = {}
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return out

    lines = text.splitlines()
    first_cells = [c.strip() for c in lines[0].split(",")]

    # Probe: is first cell purely numeric? then headerless 3-col format
    headerless = False
    try:
        float(first_cells[1])  # the MOS-like column should be numeric
        # Also first column should look like a flickr_id (digits only)
        if first_cells[0].isdigit():
            headerless = True
    except (ValueError, IndexError):
        headerless = False

    if headerless:
        # Format (2): flickr_id, mos, framerate (no header)
        for ln in lines:
            parts = [c.strip() for c in ln.split(",")]
            if len(parts) < 2:
                continue
            try:
                vid_id = parts[0]
                mos = float(parts[1])
            except ValueError:
                continue
            if not vid_id.endswith(".mp4"):
                vid_id = f"{vid_id}.mp4"
            out[vid_id] = mos
        return out

    # Headered: use DictReader
    import io
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    mos_field = next(
        (c for c in fields if c.lower() in ("mos", "mos_score", "score")),
        None,
    )
    id_field = next(
        (c for c in fields if c.lower() in ("file_name", "filename",
                                            "flickr_id", "video_id")),
        None,
    )
    if not (mos_field and id_field):
        return out
    for row in reader:
        vid_id = str(row[id_field]).strip()
        if not vid_id.endswith(".mp4"):
            vid_id = f"{vid_id}.mp4"
        try:
            out[vid_id] = float(row[mos_field])
        except (ValueError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# CineInfini audit wrapper
# ---------------------------------------------------------------------------
def audit_one(video_path: Path, output_dir: Path, profile_path: Optional[Path]) -> Dict[str, Any]:
    """Run a single video through CineInfini and return the audit data dict."""
    from cineinfini.pipeline.orchestrator import run_audit
    from cineinfini.core.config import load_config, set_config

    if profile_path and profile_path.exists():
        set_config(load_config(profile_path))
    audit, out_dir = run_audit(video_path, output_dir=output_dir)
    return audit


def extract_metrics(audit: Dict[str, Any]) -> Dict[str, float]:
    """Flatten an audit dict into a {metric_name: value} dict.

    Picks: composite, videoscore axes, per-module summaries.
    Per-shot values are aggregated by mean.
    """
    out: Dict[str, float] = {}

    # Global composite
    if (c := audit.get("composite_score")) is not None:
        out["composite_score"] = float(c)

    # 5-axis VideoScore
    for k, v in (audit.get("videoscore_axes") or {}).items():
        if v is not None:
            out[f"vs_{k}"] = float(v)

    # Per-module summaries (skip per_shot)
    for mod_id, mod in (audit.get("modules") or {}).items():
        for k, v in mod.items():
            if k in ("module", "version", "available", "reason", "per_shot"):
                continue
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                out[f"{mod_id}.{k}"] = float(v)

    # Per-shot mean
    gates = audit.get("gates") or {}
    if gates:
        agg: Dict[str, List[float]] = {}
        for sid, gate in gates.items():
            for k, v in gate.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(float(v))
        for k, vs in agg.items():
            if vs:
                out[f"shot_mean.{k}"] = float(np.mean(vs))

    return out


# ---------------------------------------------------------------------------
# Correlation + CI
# ---------------------------------------------------------------------------
def correlation_with_ci(values: np.ndarray, mos: np.ndarray,
                       n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Spearman + Pearson + bootstrap CI95 for one metric vs MOS.

    Skips NaN / inf pairs. Returns a dict with all 6 numbers.
    """
    from scipy.stats import spearmanr, pearsonr

    mask = np.isfinite(values) & np.isfinite(mos)
    v, m = values[mask], mos[mask]
    if len(v) < 5 or np.std(v) == 0:
        return {"n": int(len(v)), "spearman": None, "pearson": None,
                "spearman_p": None, "pearson_p": None,
                "spearman_ci95_lo": None, "spearman_ci95_hi": None}

    s_rho, s_p = spearmanr(v, m)
    p_rho, p_p = pearsonr(v, m)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(v), len(v))
        if np.std(v[idx]) == 0 or np.std(m[idx]) == 0:
            continue
        r, _ = spearmanr(v[idx], m[idx])
        if not math.isnan(r):
            boots.append(r)
    boots = np.array(boots) if boots else np.array([np.nan])
    return {
        "n": int(mask.sum()),
        "spearman": float(s_rho),
        "spearman_p": float(s_p),
        "pearson": float(p_rho),
        "pearson_p": float(p_p),
        "spearman_ci95_lo": float(np.quantile(boots, 0.025)),
        "spearman_ci95_hi": float(np.quantile(boots, 0.975)),
    }


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------
def per_module_contribution(per_video: List[Dict[str, Any]], mos: np.ndarray,
                            metric_keys: List[str]) -> Dict[str, Dict[str, float]]:
    """For each metric_key, report Δρ if it were dropped from a uniform mean.

    Approximation: ρ(full mean) − ρ(mean without k). Real Shapley
    contribution would be O(2^N) — too expensive for routine releases.
    """
    M = np.array([[v.get(k, np.nan) for k in metric_keys] for v in per_video])
    # z-normalize columns (so mean is balanced)
    means = np.nanmean(M, axis=0)
    stds = np.nanstd(M, axis=0)
    stds[stds == 0] = 1.0
    Z = (M - means) / stds

    full_mean = np.nanmean(Z, axis=1)
    full_corr = correlation_with_ci(full_mean, mos, n_boot=500)

    out: Dict[str, Dict[str, float]] = {}
    for j, k in enumerate(metric_keys):
        Zj = np.delete(Z, j, axis=1)
        partial = np.nanmean(Zj, axis=1)
        c = correlation_with_ci(partial, mos, n_boot=500)
        if c["spearman"] is None or full_corr["spearman"] is None:
            out[k] = {"delta_rho": None}
        else:
            out[k] = {
                "delta_rho": float(full_corr["spearman"] - c["spearman"]),
                "rho_without": float(c["spearman"]),
            }
    out["__full_mean__"] = full_corr
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--labels-csv", required=True, type=Path,
                    help="info.txt for T2VQA-DB OR labels_with_videos.csv for VideoFeedback")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--profile", type=Path, default=None,
                    help="Path to CineInfini profile YAML; defaults to baseline")
    ap.add_argument("--max-videos", type=int, default=None,
                    help="Limit number of videos (for quick smoke runs)")
    ap.add_argument("--dataset-name", default="t2vqa",
                    choices=["t2vqa", "videofeedback", "konvid"])
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audits_dir = args.out_dir / "audits"
    audits_dir.mkdir(exist_ok=True)
    final_json = args.out_dir / "t2vqa_calibration.json"
    paper_json = args.out_dir / "t2vqa_calibration_paper.json"

    # -- Load labels
    if args.dataset_name == "t2vqa":
        labels = load_t2vqa_labels(args.labels_csv)
    elif args.dataset_name == "konvid":
        labels = load_konvid_labels(args.labels_csv)
    else:
        labels = load_videofeedback_labels(args.labels_csv)
    print(f"Loaded {len(labels)} labels.")

    # -- Match videos
    available = []
    for vid, mos in labels.items():
        # Exact filename match first
        candidates = list(args.videos_dir.rglob(vid))
        if not candidates and args.dataset_name == "konvid":
            # KoNViD videos are saved as <flickr_id>_<type>_centercrop_960x540_8s.mp4
            # but labels may key by just <flickr_id>.mp4
            stem = Path(vid).stem
            candidates = list(args.videos_dir.rglob(f"{stem}_*_8s.mp4"))
        if candidates:
            available.append({"path": candidates[0], "video_id": vid, "mos": mos})
    print(f"Matched {len(available)} videos with labels.")
    if args.max_videos:
        available = available[: args.max_videos]
        print(f"Capped to {len(available)} videos.")

    # -- Audit each video (resumable via per-video JSON files)
    per_video: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(available):
        cache = audits_dir / f"{item['video_id']}.json"
        if cache.exists():
            audit = json.loads(cache.read_text())
        else:
            try:
                audit = audit_one(item["path"], audits_dir / item["video_id"],
                                 args.profile)
                cache.write_text(json.dumps(_jsonable(audit)))
            except Exception as e:
                print(f"  ! {item['video_id']} failed: {e}")
                traceback.print_exc()
                continue
        flat = extract_metrics(audit)
        flat["mos"] = item["mos"]
        flat["video_id"] = item["video_id"]
        per_video.append(flat)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(available)} ({time.time()-t0:.1f}s)")

    print(f"\nProcessed {len(per_video)} / {len(available)} videos "
          f"in {time.time()-t0:.1f}s\n")
    if not per_video:
        print("Nothing to correlate. Exiting.")
        return 1

    # -- Compute correlations
    metric_keys = sorted(
        {k for v in per_video for k in v if k not in ("mos", "video_id")
         and isinstance(v[k], (int, float))}
    )
    mos_arr = np.array([v["mos"] for v in per_video], dtype=float)

    correlations: Dict[str, Dict[str, float]] = {}
    for k in metric_keys:
        vals = np.array([v.get(k, np.nan) for v in per_video], dtype=float)
        correlations[k] = correlation_with_ci(vals, mos_arr,
                                              n_boot=args.n_bootstrap)

    # -- Composite if available
    composite_corr = correlations.get("composite_score") or {}

    # -- Module ablation
    contribs = per_module_contribution(per_video, mos_arr, metric_keys)

    # -- Output
    out = {
        "dataset": args.dataset_name,
        "labels_file": str(args.labels_csv),
        "videos_dir": str(args.videos_dir),
        "profile": str(args.profile) if args.profile else "default",
        "n_videos": len(per_video),
        "n_bootstrap": args.n_bootstrap,
        "correlations": correlations,
        "ablation_delta_rho": contribs,
        "per_video": per_video,
    }
    final_json.write_text(json.dumps(_jsonable(out), indent=2))
    print(f"Saved -> {final_json}")

    # -- Flat dict for paper filler
    paper = {
        f"{args.dataset_name.upper()}_N": len(per_video),
        f"{args.dataset_name.upper()}_RHO_COMPOSITE": composite_corr.get("spearman"),
        f"{args.dataset_name.upper()}_RHO_COMPOSITE_CI_LO": composite_corr.get("spearman_ci95_lo"),
        f"{args.dataset_name.upper()}_RHO_COMPOSITE_CI_HI": composite_corr.get("spearman_ci95_hi"),
        f"{args.dataset_name.upper()}_RHO_COMPOSITE_P": composite_corr.get("spearman_p"),
        f"{args.dataset_name.upper()}_PEARSON_COMPOSITE": composite_corr.get("pearson"),
    }
    # Top-3 modules
    top = sorted(
        ((k, c) for k, c in correlations.items() if c.get("spearman") is not None),
        key=lambda kv: -abs(kv[1]["spearman"])
    )[:3]
    for rank, (k, c) in enumerate(top, 1):
        paper[f"{args.dataset_name.upper()}_TOP{rank}_NAME"] = k
        paper[f"{args.dataset_name.upper()}_TOP{rank}_RHO"] = c["spearman"]
    paper_json.write_text(json.dumps(_jsonable(paper), indent=2))
    print(f"Saved -> {paper_json}")

    # -- Console report
    print("\n" + "=" * 70)
    print(f"REAL Spearman ρ vs MOS (n={len(per_video)}, "
          f"bootstrap CI95, n_boot={args.n_bootstrap})")
    print("=" * 70)
    print(f"{'metric':<40} {'rho':>7} {'p':>9} {'CI95':>22}")
    for k in sorted(correlations, key=lambda x: -abs(correlations[x].get("spearman") or 0)):
        c = correlations[k]
        if c.get("spearman") is None:
            continue
        rho, p = c["spearman"], c["spearman_p"]
        lo, hi = c["spearman_ci95_lo"], c["spearman_ci95_hi"]
        print(f"{k:<40} {rho:>+7.3f} {p:>9.4f} [{lo:+.3f},{hi:+.3f}]")
    return 0


def _jsonable(o):
    """Recursive JSON sanitizer (numpy → python)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, Path):
        return str(o)
    return o


if __name__ == "__main__":
    sys.exit(main() or 0)
