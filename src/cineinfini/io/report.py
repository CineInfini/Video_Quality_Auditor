"""Generate Markdown dashboards and figures"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from math import pi

def fmt(val):
    return "—" if val is None else f"{val:.3f}"

def generate_intra_report(video_name, metrics_data, output_dir, thresholds, create_html=True):
    out_dir = Path(output_dir) / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(metrics_data, indent=2))
    gates = metrics_data["gates"]
    shot_ids = sorted([int(k) for k in gates.keys()])
    inter_results = metrics_data.get("inter_results", [])
    narrative_scores = metrics_data.get("narrative_coherence", [])
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

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
        values = [gates[str(s)].get(metric, 0) or 0 for s in shot_ids]
        plt.figure(figsize=(10,4))
        bars = plt.bar(shot_ids, values, color='steelblue')
        plt.axhline(y=thresh, color='r', linestyle='--', label=f'threshold {thresh}')
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

    # Count faces
    face_counts = [1 if gates[str(s)].get("identity_intra") is not None else 0 for s in shot_ids]
    total_faces = sum(face_counts)
    face_stats = f"Total faces detected: {total_faces} (in {sum(1 for c in face_counts if c>0)} shots)"

    # Inter-shot coherence lines
    inter_lines = []
    if inter_results:
        inter_lines.append("## Inter‑Shot Coherence Loss")
        inter_lines.append("| A→B | Structure | Style | Semantic | **Total** |")
        inter_lines.append("|---:|---:|---:|---:|---:|")
        for r in inter_results:
            inter_lines.append(f"| {r['shot_a']}→{r['shot_b']} | {r['structure']:.3f} | {r['style']:.3f} | {r['semantic']:.3f} | **{r['total']:.3f}** |")
        avg_total = np.mean([r['total'] for r in inter_results])
        if avg_total > 0.7:
            inter_lines.append("> **Interpretation:** Very high visual coherence between consecutive shots – smooth transitions.")
        elif avg_total > 0.5:
            inter_lines.append("> **Interpretation:** Moderate coherence – typical for normal scene changes.")
        else:
            inter_lines.append("> **Interpretation:** Low coherence – likely abrupt scene changes or content mismatches.")
    else:
        inter_lines.append("No inter‑shot coherence data available (only one shot).")

    # Narrative coherence lines
    nar_lines = []
    if narrative_scores:
        nar_lines.append("## Narrative Coherence (DINOv2)")
        nar_lines.append("| Transition | Similarity |")
        nar_lines.append("|---:|---:|")
        for i, sim in enumerate(narrative_scores):
            nar_lines.append(f"| {i+1}→{i+2} | {sim:.3f} |")
        avg_nar = np.mean(narrative_scores) if narrative_scores else 1.0
        if avg_nar < thresholds.get("narrative_coherence", 0.7):
            nar_lines.append(f"> **Warning:** Low narrative coherence (mean {avg_nar:.3f} < {thresholds.get('narrative_coherence',0.7)}) – possible logical inconsistencies between shots.")
        else:
            nar_lines.append(f"> **Interpretation:** Good narrative consistency (mean {avg_nar:.3f}).")
    else:
        nar_lines.append("Narrative coherence not computed.")

    # Statistics
    stats = {}
    for metric in ["motion_peak_div","ssim3d_self","flicker","identity_intra","ssim_long_range","flicker_hf_var","clip_temp_consistency"]:
        vals = [gates[str(s)].get(metric) for s in shot_ids if gates[str(s)].get(metric) is not None]
        if vals:
            stats[metric] = {"mean": np.mean(vals), "std": np.std(vals), "min": np.min(vals), "max": np.max(vals)}
        else:
            stats[metric] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    comp_vals = [gates[str(s)].get("composite", 0) for s in shot_ids]
    stats["composite"] = {"mean": np.mean(comp_vals), "std": np.std(comp_vals), "min": np.min(comp_vals), "max": np.max(comp_vals)}

    # Radar chart
    categories = ["Motion", "SSIM 3D", "Flicker", "Identity drift", "SSIM long range", "CLIP temporal"]
    norm_vals = [
        1 - min(stats["motion_peak_div"]["mean"] / thresholds.get("motion",25.0), 1.0) if thresholds.get("motion",25.0)>0 else 1.0,
        min(stats["ssim3d_self"]["mean"] / thresholds.get("ssim3d",0.45), 1.0) if thresholds.get("ssim3d",0.45)>0 else 0.0,
        1 - min(stats["flicker"]["mean"] / thresholds.get("flicker",0.1), 1.0) if thresholds.get("flicker",0.1)>0 else 1.0,
        1 - min(stats["identity_intra"]["mean"] / thresholds.get("identity_drift",0.6), 1.0) if thresholds.get("identity_drift",0.6)>0 else 1.0,
        min(stats["ssim_long_range"]["mean"] / thresholds.get("ssim_long_range",0.45), 1.0) if thresholds.get("ssim_long_range",0.45)>0 else 0.0,
        min(stats["clip_temp_consistency"]["mean"] / thresholds.get("clip_temp",0.25), 1.0) if thresholds.get("clip_temp",0.25)>0 else 0.0,
    ]
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

    # Write Markdown
    md_lines = [f"# Exhaustive Dashboard – {video_name}", f"*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*", ""]
    video_info = metrics_data.get("video_info", {})
    md_lines.append("## 1. Video Overview")
    md_lines.append(f"- **Duration analysed**: {video_info.get('duration', 'N/A')} seconds")
    md_lines.append(f"- **Resolution (frames)**: {video_info.get('resolution', 'N/A')}")
    md_lines.append(f"- **Frames per second**: {video_info.get('fps', 'N/A')}")
    md_lines.append(f"- **Number of shots**: {len(gates)}")
    md_lines.append(f"- **Embedder**: {metrics_data.get('params_used', {}).get('embedder', 'N/A')}")
    md_lines.append(f"- **Semantic scorer**: {metrics_data.get('params_used', {}).get('semantic_scorer', 'N/A')}")
    md_lines.append(f"- **Face detection**: {face_stats}\n")

    md_lines.append("## 2. Per‑Shot Metrics")
    md_lines.append("| Shot | Motion | SSIM3D | Flicker | Identity | SSIM LR | Flicker HF | CLIP temp | Composite |")
    md_lines.append("|------|--------|--------|---------|----------|---------|------------|-----------|-----------|")
    for sid in shot_ids:
        g = gates[str(sid)]
        md_lines.append(f"| {sid} | {fmt(g.get('motion_peak_div'))} | {fmt(g.get('ssim3d_self'))} | {fmt(g.get('flicker'))} | {fmt(g.get('identity_intra'))} | {fmt(g.get('ssim_long_range'))} | {fmt(g.get('flicker_hf_var'))} | {fmt(g.get('clip_temp_consistency'))} | {fmt(g.get('composite'))} |")
    md_lines.append("")

    md_lines.append("## 3. Statistical Summary")
    md_lines.append("| Metric | Mean | Std | Min | Max |")
    md_lines.append("|--------|------|-----|-----|-----|")
    for metric, label in [("motion_peak_div","Motion peak"),("ssim3d_self","3D-SSIM"),("flicker","Flicker"),
                          ("identity_intra","Identity drift"),("ssim_long_range","SSIM long range"),
                          ("flicker_hf_var","Flicker HF var"),("clip_temp_consistency","CLIP temporal"),
                          ("composite","Composite Score")]:
        s = stats[metric]
        md_lines.append(f"| {label} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} | {s['max']:.3f} |")
    md_lines.append("")

    md_lines.append("## 4. Alerts (Shots exceeding thresholds)")
    alerts = []
    for sid in shot_ids:
        g = gates[str(sid)]
        if g.get('motion_peak_div') and g['motion_peak_div'] > thresholds.get('motion',25.0):
            alerts.append(f"- **Shot {sid}**: excessive motion ({g['motion_peak_div']:.1f} > {thresholds.get('motion',25.0)}) – *camera shake or fast action*")
        if g.get('ssim3d_self') and g['ssim3d_self'] < thresholds.get('ssim3d',0.45):
            alerts.append(f"- **Shot {sid}**: low temporal stability (SSIM={g['ssim3d_self']:.3f}) – *compression artefacts or noise*")
        if g.get('flicker') and g['flicker'] > thresholds.get('flicker',0.1):
            alerts.append(f"- **Shot {sid}**: flicker ({g['flicker']:.3f}) – *luminance variation*")
        if g.get('identity_intra') and g['identity_intra'] > thresholds.get('identity_drift',0.6):
            alerts.append(f"- **Shot {sid}**: identity drift ({g['identity_intra']:.3f}) – *face appearance change*")
        if g.get('ssim_long_range') and g['ssim_long_range'] < thresholds.get('ssim_long_range',0.45):
            alerts.append(f"- **Shot {sid}**: long‑range morphing ({g['ssim_long_range']:.3f}) – *unwanted deformation*")
        if g.get('clip_temp_consistency') and g['clip_temp_consistency'] < thresholds.get('clip_temp',0.25):
            alerts.append(f"- **Shot {sid}**: semantic temporal inconsistency ({g['clip_temp_consistency']:.3f}) – *content changed unexpectedly*")
    md_lines.extend(alerts if alerts else ["No alerts."])
    md_lines.append("")

    md_lines.append("## 5. Inter‑Shot Coherence")
    md_lines.extend(inter_lines)
    md_lines.append("")
    md_lines.append("## 6. Narrative Coherence")
    md_lines.extend(nar_lines)
    md_lines.append("")

    md_lines.append("## 7. Improvement Suggestions")
    avg_motion = stats["motion_peak_div"]["mean"]
    avg_ssim = stats["ssim3d_self"]["mean"]
    avg_flicker = stats["flicker"]["mean"]
    avg_identity = stats["identity_intra"]["mean"]
    avg_ssim_lr = stats["ssim_long_range"]["mean"]
    avg_clip = stats["clip_temp_consistency"]["mean"]
    recs = []
    if avg_motion > thresholds.get('motion',25.0):
        recs.append(f"- **High motion (mean {avg_motion:.1f})** → Use stabilisation (`ffmpeg -vf vidstabdetect,vidstabtransform`) or reduce camera movement.")
    if avg_ssim < thresholds.get('ssim3d',0.45):
        recs.append(f"- **Low temporal stability (mean SSIM {avg_ssim:.2f})** → Increase encoding bitrate (`-crf 18`) or apply temporal denoising (`-vf hqdn3d`).")
    if avg_flicker > thresholds.get('flicker',0.1):
        recs.append(f"- **Flicker (mean {avg_flicker:.3f})** → Apply deflicker filter (`ffmpeg -vf deflicker`).")
    if avg_identity > thresholds.get('identity_drift',0.6):
        recs.append(f"- **Identity drift (mean {avg_identity:.2f})** → Normalise illumination (`-vf eq=brightness=0:contrast=1.2`) or use better face alignment.")
    if avg_ssim_lr < thresholds.get('ssim_long_range',0.45):
        recs.append(f"- **Long‑range instability (mean {avg_ssim_lr:.2f})** → Reduce shot duration or avoid extreme viewpoint changes.")
    if avg_clip < thresholds.get('clip_temp',0.25):
        recs.append(f"- **Semantic inconsistency (mean {avg_clip:.2f})** → Improve prompt or use a higher‑level model (TULIP).")
    if not recs:
        recs.append("✅ All metrics are within acceptable ranges. No immediate improvement needed.")
    md_lines.extend(recs)
    md_lines.append("")
    md_lines.append("## 8. Figures")
    md_lines.append("Below are the bar charts for each metric (red bars indicate values exceeding the threshold).")
    md_lines.append("*(See the displayed images below this text.)*")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("*This dashboard was automatically generated by the CineInfini pipeline.*")
    (out_dir / "dashboard.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅ Exhaustive Markdown dashboard generated in {out_dir / 'dashboard.md'}")
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
        composite_vals = [g.get("composite", 0) for g in gates.values() if g.get("composite") is not None]
        rows.append({
            "video": Path(d).name,
            "motion_mean": np.mean(motion) if motion else 0,
            "ssim_mean": np.mean(ssim) if ssim else 0,
            "flicker_mean": np.mean(flicker) if flicker else 0,
            "identity_mean": np.mean(identity) if identity else 0,
            "ssim_lr_mean": np.mean(ssim_lr) if ssim_lr else 0,
            "clip_temp_mean": np.mean(clip_temp) if clip_temp else 0,
            "composite_mean": np.mean(composite_vals) if composite_vals else 0,
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
            plt.axhline(y=thresh, color='red', linestyle='--', label=f'threshold {thresh}')
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

    md_lines = [f"# Exhaustive Inter‑Video Dashboard – {comparison_name}",
                f"*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                f"\n## 1. Overview\nNumber of videos compared: {len(df)}",
                f"Total shots analysed: {df['n_shots'].sum()}",
                "\n## 2. Average Metrics per Video",
                df[['video','n_shots','motion_mean','ssim_mean','flicker_mean','identity_mean',
                    'ssim_lr_mean','clip_temp_mean','composite_mean']].round(3).to_markdown(index=False),
                "\n## 3. Ranking (Composite Quality Score)",
                df.sort_values('composite_mean', ascending=False)[['video','composite_mean']].round(3).to_markdown(index=False),
                "\n*Higher composite score indicates better overall quality.*",
                "\n## 4. Best and Worst per Metric"]
    for metric, label in [("motion_mean","Motion (lower is better)"),("ssim_mean","SSIM 3D (higher is better)"),
                          ("flicker_mean","Flicker (lower is better)"),("identity_mean","Identity drift (lower is better)"),
                          ("ssim_lr_mean","SSIM long range (higher is better)"),("clip_temp_mean","CLIP temporal (higher is better)")]:
        if "lower" in label:
            best = df.loc[df[metric].idxmin()]
            worst = df.loc[df[metric].idxmax()]
        else:
            best = df.loc[df[metric].idxmax()]
            worst = df.loc[df[metric].idxmin()]
        md_lines.append(f"- **{label}** → Best: {best['video']} ({best[metric]:.3f}), Worst: {worst['video']} ({worst[metric]:.3f})")
    md_lines.append("\n## 5. Statistical Summary (across videos)")
    stats = df[['motion_mean','ssim_mean','flicker_mean','identity_mean','ssim_lr_mean','clip_temp_mean','composite_mean']].describe().round(3)
    md_lines.append(stats.to_markdown())
    md_lines.append("\n## 6. Improvement Suggestions (by video)")
    for _, row in df.iterrows():
        issues = []
        if row['motion_mean'] > thresholds.get('motion',25.0): issues.append("high motion")
        if row['ssim_mean'] < thresholds.get('ssim3d',0.45): issues.append("low SSIM")
        if row['flicker_mean'] > thresholds.get('flicker',0.1): issues.append("flicker")
        if row['identity_mean'] > thresholds.get('identity_drift',0.6): issues.append("identity drift")
        if row['ssim_lr_mean'] < thresholds.get('ssim_long_range',0.45): issues.append("long‑range instability")
        if row['clip_temp_mean'] < thresholds.get('clip_temp',0.25): issues.append("semantic inconsistency")
        if issues:
            md_lines.append(f"- **{row['video']}** : {', '.join(issues)}. See its intra‑report for detailed improvement steps.")
    md_lines.append("\n## 7. Figures")
    for metric, label, _, _ in metric_config:
        md_lines.append(f"![{label}](figures/{metric}.png)")
    md_lines.append("![Radar chart](figures/radar_chart.png)")
    md_lines.append("\n---\n*This dashboard was automatically generated by the CineInfini pipeline.*")
    (out_dir / "dashboard.md").write_text("\n".join(md_lines), encoding="utf-8")
    return out_dir
