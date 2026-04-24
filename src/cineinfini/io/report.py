import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from math import pi

def safe_format(v, fmt=".3f"):
    """Retourne une valeur formatée ou '—' si None."""
    return f"{v:{fmt}}" if v is not None else "—"

def generate_intra_report(video_name, metrics_data, output_dir, thresholds):
    out_dir = Path(output_dir) / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(metrics_data, indent=2))

    gates = metrics_data["gates"]
    shot_ids = sorted([int(k) for k in gates.keys()])
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    # Bar charts per metric
    metric_config = [
        ("motion_peak_div", "Motion peak", thresholds.get("motion", 25.0), "below"),
        ("ssim3d_self", "3D-SSIM", thresholds.get("ssim3d", 0.45), "above"),
        ("flicker", "Flicker", thresholds.get("flicker", 0.1), "below"),
        ("identity_intra", "Identity drift", thresholds.get("identity_drift", 0.6), "below"),
        ("ssim_long_range", "SSIM long range", thresholds.get("ssim_long_range", 0.45), "above"),
        ("flicker_hf_var", "Flicker HF var", thresholds.get("flicker_hf", 0.01), "below"),
        ("clip_temp_consistency", "CLIP temporal", thresholds.get("clip_temp", 0.25), "above"),
    ]

    for metric, label, thresh, direction in metric_config:
        if thresh is None:
            continue
        values = []
        for s in shot_ids:
            v = gates[str(s)].get(metric, 0.0)
            values.append(v if v is not None else 0.0)
        if all(v == 0.0 for v in values):
            continue

        plt.figure(figsize=(10,4))
        bars = plt.bar(shot_ids, values, color='steelblue')
        plt.axhline(y=thresh, color='r', linestyle='--', label=f"threshold {thresh:.3f}")
        for i, v in enumerate(values):
            if (direction == "below" and v > thresh) or (direction == "above" and v < thresh):
                bars[i].set_color('red')
        plt.xlabel('Shot ID')
        plt.ylabel(label)
        plt.title(f'{label} per shot')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(figs_dir / f"{metric}.png", dpi=100)
        plt.close()

    # Radar chart (averages)
    stats = {}
    for metric in ["motion_peak_div","ssim3d_self","flicker","identity_intra",
                   "ssim_long_range","clip_temp_consistency"]:
        vals = [gates[str(s)].get(metric) for s in shot_ids if gates[str(s)].get(metric) is not None]
        stats[metric] = np.mean(vals) if vals else 0.0

    norm_vals = [
        1 - min(stats["motion_peak_div"] / thresholds.get("motion",25.0), 1.0) if thresholds.get("motion",25.0) > 0 else 1.0,
        min(stats["ssim3d_self"] / thresholds.get("ssim3d",0.45), 1.0) if thresholds.get("ssim3d",0.45) > 0 else 0.0,
        1 - min(stats["flicker"] / thresholds.get("flicker",0.1), 1.0) if thresholds.get("flicker",0.1) > 0 else 1.0,
        1 - min(stats["identity_intra"] / thresholds.get("identity_drift",0.6), 1.0) if thresholds.get("identity_drift",0.6) > 0 else 1.0,
        min(stats["ssim_long_range"] / thresholds.get("ssim_long_range",0.45), 1.0) if thresholds.get("ssim_long_range",0.45) > 0 else 0.0,
        min(stats["clip_temp_consistency"] / thresholds.get("clip_temp",0.25), 1.0) if thresholds.get("clip_temp",0.25) > 0 else 0.0,
    ]

    categories = ["Motion", "3D-SSIM", "Flicker", "Identity drift", "SSIM long range", "CLIP temporal"]
    angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    norm_vals += norm_vals[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, norm_vals, linewidth=2, label=video_name)
    ax.fill(angles, norm_vals, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0,1)
    ax.set_title("Performance radar (closer to outer edge = better)", size=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1,1.0))
    plt.tight_layout()
    plt.savefig(figs_dir / "radar_chart.png", dpi=100)
    plt.close()

    # Markdown dashboard (safe formatting, no broken f-string)
    md_lines = []
    md_lines.append("# Exhaustive Dashboard – {}".format(video_name))
    md_lines.append("")
    md_lines.append("## Overview")
    md_lines.append("- Duration analysed: {} seconds".format(metrics_data['video_info'].get('duration')))
    md_lines.append("- Number of shots: {}".format(len(gates)))
    md_lines.append("")
    md_lines.append("## Per‑Shot Metrics")
    md_lines.append("| Shot | Motion | 3D-SSIM | Flicker | Identity drift | SSIM LR | CLIP temp |")
    md_lines.append("|------|--------|---------|---------|----------------|---------|-----------|")
    for sid in shot_ids:
        g = gates[str(sid)]
        md_lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                sid,
                safe_format(g.get('motion_peak_div')),
                safe_format(g.get('ssim3d_self')),
                safe_format(g.get('flicker')),
                safe_format(g.get('identity_intra')),
                safe_format(g.get('ssim_long_range')),
                safe_format(g.get('clip_temp_consistency'))
            )
        )
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("*Generated by CineInfini*")
    md = "\n".join(md_lines)
    (out_dir / "dashboard.md").write_text(md, encoding="utf-8")

    print(f"Dashboard generated in {out_dir / 'dashboard.md'}")
    return out_dir


def generate_inter_report(intra_report_dirs, output_dir, thresholds, comparison_name="comparison"):
    out_dir = Path(output_dir) / comparison_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for d in intra_report_dirs:
        json_path = Path(d) / "data.json"
        if not json_path.exists():
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        gates = data["gates"]
        motion = [g.get("motion_peak_div") for g in gates.values() if g.get("motion_peak_div") is not None]
        ssim = [g.get("ssim3d_self") for g in gates.values() if g.get("ssim3d_self") is not None]
        flicker = [g.get("flicker") for g in gates.values() if g.get("flicker") is not None]
        identity = [g.get("identity_intra") for g in gates.values() if g.get("identity_intra") is not None]
        ssim_lr = [g.get("ssim_long_range") for g in gates.values() if g.get("ssim_long_range") is not None]
        clip_temp = [g.get("clip_temp_consistency") for g in gates.values() if g.get("clip_temp_consistency") is not None]
        comp_vals = [g.get("composite") for g in gates.values() if g.get("composite") is not None]

        rows.append({
            "video": Path(d).name,
            "motion_mean": np.mean(motion) if motion else 0,
            "ssim_mean": np.mean(ssim) if ssim else 0,
            "flicker_mean": np.mean(flicker) if flicker else 0,
            "identity_mean": np.mean(identity) if identity else 0,
            "ssim_lr_mean": np.mean(ssim_lr) if ssim_lr else 0,
            "clip_temp_mean": np.mean(clip_temp) if clip_temp else 0,
            "composite_mean": np.mean(comp_vals) if comp_vals else 0,
            "n_shots": len(gates)
        })

    if not rows:
        print("No valid intra-reports found")
        return None

    df = pd.DataFrame(rows)
    (out_dir / "data.json").write_text(df.to_json(orient="records", indent=2))

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    metric_config = [
        ("motion_mean", "Motion (peak)", thresholds.get("motion", 25.0), "below"),
        ("ssim_mean", "3D-SSIM", thresholds.get("ssim3d", 0.45), "above"),
        ("flicker_mean", "Flicker", thresholds.get("flicker", 0.1), "below"),
        ("identity_mean", "Identity drift", thresholds.get("identity_drift", 0.6), "below"),
        ("ssim_lr_mean", "SSIM long range", thresholds.get("ssim_long_range", 0.45), "above"),
        ("clip_temp_mean", "CLIP temporal", thresholds.get("clip_temp", 0.25), "above"),
        ("composite_mean", "Composite Score", 0.5, "above"),
    ]

    for metric, label, thresh, direction in metric_config:
        plt.figure(figsize=(10,6))
        bars = plt.bar(df["video"], df[metric], color='steelblue')
        if metric != "composite_mean":
            plt.axhline(y=thresh, color='red', linestyle='--', label=f"threshold {thresh:.3f}")
        for i, v in enumerate(df[metric]):
            if metric != "composite_mean":
                if (direction == "below" and v > thresh) or (direction == "above" and v < thresh):
                    bars[i].set_color('red')
            else:
                if v < df[metric].median():
                    bars[i].set_color('red')
        plt.ylabel(label)
        plt.title(f'Average {label} per video')
        plt.xticks(rotation=45, ha='right')
        if metric != "composite_mean":
            plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / f"{metric}.png", dpi=100)
        plt.close()

    # Radar inter-video
    categories = ["Motion (low is good)", "SSIM 3D (high is good)", "Flicker (low is good)",
                  "Identity drift (low is good)", "SSIM long range (high is good)", "CLIP temporal (high is good)"]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    for _, row in df.iterrows():
        norm_vals = [
            1 - min(row["motion_mean"] / thresholds.get("motion",25.0), 1.0) if thresholds.get("motion",25.0)>0 else 1.0,
            min(row["ssim_mean"] / thresholds.get("ssim3d",0.45), 1.0) if thresholds.get("ssim3d",0.45)>0 else 0.0,
            1 - min(row["flicker_mean"] / thresholds.get("flicker",0.1), 1.0) if thresholds.get("flicker",0.1)>0 else 1.0,
            1 - min(row["identity_mean"] / thresholds.get("identity_drift",0.6), 1.0) if thresholds.get("identity_drift",0.6)>0 else 1.0,
            min(row["ssim_lr_mean"] / thresholds.get("ssim_long_range",0.45), 1.0) if thresholds.get("ssim_long_range",0.45)>0 else 0.0,
            min(row["clip_temp_mean"] / thresholds.get("clip_temp",0.25), 1.0) if thresholds.get("clip_temp",0.25)>0 else 0.0,
        ]
        angles = [n / len(categories) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals, linewidth=2, label=row["video"])
        ax.fill(angles, norm_vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=7)
    ax.set_ylim(0,1)
    ax.set_title("Normalised performance radar (outer = better)", size=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.0))
    plt.tight_layout()
    plt.savefig(figs_dir / "radar_chart.png", dpi=100)
    plt.close()

    # Markdown inter-report (safe)
    md_lines = []
    md_lines.append("# Exhaustive Inter‑Video Dashboard – {}".format(comparison_name))
    md_lines.append("*Generated on {}*".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
    md_lines.append("")
    md_lines.append("## 1. Overview")
    md_lines.append("Number of videos compared: {}".format(len(df)))
    md_lines.append("Total shots analysed: {}".format(df['n_shots'].sum()))
    md_lines.append("")
    md_lines.append("## 2. Average Metrics per Video")
    md_lines.append(df[['video','n_shots','motion_mean','ssim_mean','flicker_mean',
                        'identity_mean','ssim_lr_mean','clip_temp_mean','composite_mean']].round(3).to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## 3. Ranking (Composite Quality Score)")
    md_lines.append(df.sort_values('composite_mean', ascending=False)[['video','composite_mean']].round(3).to_markdown(index=False))
    md_lines.append("*Higher composite score indicates better overall quality.*")
    md_lines.append("")
    md_lines.append("## 4. Best and Worst per Metric")
    for metric, label in [("motion_mean","Motion (lower is better)"),("ssim_mean","SSIM 3D (higher is better)"),
                          ("flicker_mean","Flicker (lower is better)"),("identity_mean","Identity drift (lower is better)"),
                          ("ssim_lr_mean","SSIM long range (higher is better)"),("clip_temp_mean","CLIP temporal (higher is better)")]:
        if "lower" in label:
            best = df.loc[df[metric].idxmin()]
            worst = df.loc[df[metric].idxmax()]
        else:
            best = df.loc[df[metric].idxmax()]
            worst = df.loc[df[metric].idxmin()]
        md_lines.append("- **{}** → Best: {} ({:.3f}), Worst: {} ({:.3f})".format(label, best['video'], best[metric], worst['video'], worst[metric]))
    md_lines.append("")
    md_lines.append("## 5. Statistical Summary (across videos)")
    md_lines.append(df[['motion_mean','ssim_mean','flicker_mean','identity_mean',
                        'ssim_lr_mean','clip_temp_mean','composite_mean']].describe().round(3).to_markdown())
    md_lines.append("")
    md_lines.append("## 6. Improvement Suggestions (by video)")
    for _, row in df.iterrows():
        issues = []
        if row['motion_mean'] > thresholds.get('motion',25.0): issues.append("high motion")
        if row['ssim_mean'] < thresholds.get('ssim3d',0.45): issues.append("low SSIM")
        if row['flicker_mean'] > thresholds.get('flicker',0.1): issues.append("flicker")
        if row['identity_mean'] > thresholds.get('identity_drift',0.6): issues.append("identity drift")
        if row['ssim_lr_mean'] < thresholds.get('ssim_long_range',0.45): issues.append("long‑range instability")
        if row['clip_temp_mean'] < thresholds.get('clip_temp',0.25): issues.append("semantic inconsistency")
        if issues:
            md_lines.append("- **{}** : {}. See its intra‑report for detailed improvement steps.".format(row['video'], ', '.join(issues)))
    md_lines.append("")
    md_lines.append("## 7. Figures")
    for metric, label, _, _ in metric_config:
        md_lines.append("![{}](figures/{}.png)".format(label, metric))
    md_lines.append("![Radar chart](figures/radar_chart.png)")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("*This dashboard was automatically generated by the CineInfini pipeline.*")

    (out_dir / "dashboard.md").write_text("\n".join(md_lines), encoding="utf-8")
    return out_dir
