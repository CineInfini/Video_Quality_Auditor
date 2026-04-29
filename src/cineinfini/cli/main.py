"""Command-line interface for CineInfini."""
from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
@click.version_option()
def cli():
    """CineInfini – Video Quality Audit Pipeline."""


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--models-only", is_flag=True, help="Only fetch ML model weights")
@click.option("--videos-only", is_flag=True, help="Only fetch test videos")
@click.option("--include-optional", is_flag=True,
              help="Also fetch cfg.optional_models (DOVER, FAST-VQA)")
@click.option("--force", is_flag=True, help="Re-download even if files exist")
@click.option("--no-ffmpeg-check", is_flag=True, help="Skip ffmpeg / ffprobe check")
@click.option("--skip-download", is_flag=True, help="Don't download, only verify")
@click.option("--json-out", type=click.Path(), default=None,
              help="Write the bootstrap report as JSON")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None,
              help="Use this YAML config instead of the default")
def bootstrap(models_only, videos_only, include_optional, force, no_ffmpeg_check,
              skip_download, json_out, config_path):
    """Ensure ffmpeg + model weights + test videos are present.

    Reads paths from `cfg/config.yaml` (or the default `Config()`). Models go
    into `paths.models_dir`, videos into `paths.test_videos_dir`. Already
    present files are left alone unless `--force` is given.

    `--include-optional` also fetches the cross-benchmark weights
    (DOVER, FAST-VQA) declared in `cfg.optional_models`.
    """
    from ..core.bootstrap import (
        bootstrap_all, ensure_models, ensure_test_videos, ensure_system_deps,
        BootstrapReport, print_report, _ensure_one, AssetStatus,
        resolve_optional_model_url,
    )
    from ..core.config import get_config, load_config, set_config

    if config_path:
        set_config(load_config(config_path))
    cfg = get_config()

    if models_only and videos_only:
        raise click.UsageError("--models-only and --videos-only are mutually exclusive")

    if models_only:
        report = BootstrapReport()
        report.system = ensure_system_deps(() if no_ffmpeg_check else ("ffmpeg", "ffprobe"))
        report.models = ensure_models(cfg, force=force, skip_download=skip_download)
    elif videos_only:
        report = BootstrapReport()
        report.system = ensure_system_deps(() if no_ffmpeg_check else ("ffmpeg", "ffprobe"))
        report.videos = ensure_test_videos(cfg, force=force, skip_download=skip_download)
    else:
        report = bootstrap_all(
            cfg, require_ffmpeg=not no_ffmpeg_check,
            force=force, skip_download=skip_download,
        )

    if include_optional and not videos_only:
        click.echo("\nFetching optional models (DOVER, FAST-VQA, ...)")
        models_dir = cfg.models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        for key, entry in (cfg.optional_models or {}).items():
            try:
                url = resolve_optional_model_url(entry)
            except Exception as e:
                click.echo(f"  ✗ {key}: cannot resolve URL ({e})")
                continue
            target = models_dir / entry.get("filename", f"{key}.bin")
            status = _ensure_one(
                key, url, target, entry.get("sha256"),
                force=force, skip_download=skip_download,
            )
            report.models.append(status)
            mark = "✓" if status.ok else "✗"
            click.echo(f"  {mark} {key:14s} {target.name} ({status.message})")

    print_report(report)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        click.echo(f"\nReport written to {json_out}")


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None,
              help="YAML config file (or profile from cfg/profiles/) to use")
@click.option("--output", "-o", default=None,
              help="Output directory for the report (default: cfg.reports_dir)")
@click.option("--models", "-m", default=None,
              help="Override the models directory")
@click.option("--duration", "-d", default=60, type=int,
              help="Cap audit at N seconds of input video (default: 60)")
@click.option("--full", is_flag=True,
              help="Force full-video processing even on long inputs")
@click.option("--auto-bootstrap", is_flag=True,
              help="Run `bootstrap` first (download missing models if needed)")
def audit(video_path, config_path, output, models, duration, full, auto_bootstrap):
    """Audit a single video and write reports."""
    from ..pipeline.audit import audit_video, set_global_paths, CONFIG
    from ..core.config import get_config, load_config, set_config

    if config_path:
        set_config(load_config(config_path))
    cfg = get_config()
    if not models:
        models = cfg.models_dir()
    if not output:
        output = cfg.reports_dir()
    models = Path(models)
    output = Path(output)
    if auto_bootstrap:
        from ..core.bootstrap import bootstrap_all, print_report
        print_report(bootstrap_all(cfg, require_ffmpeg=True))
    set_global_paths(output, output / "benchmark")
    CONFIG["max_duration_s"] = 999999 if full else duration
    metrics, report_dir = audit_video(video_path)
    click.echo(f"✅ Report saved to {report_dir}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--vids", "-v", multiple=True, required=True)
@click.option("--output", "-o", default="comparison")
@click.option("--duration", "-d", default=10, type=int)
@click.option("--full", is_flag=True)
@click.option("--download-dir", default=None)
@click.option("--output-root", default=None)
@click.option("--models", "-m", default=None)
def compare(vids, output, duration, full, download_dir, output_root, models):
    """Compare exactly two videos."""
    from ..pipeline.audit import set_global_paths
    from ..compare import compare_videos
    from ..core.config import get_config

    if len(vids) != 2:
        raise click.BadParameter("Exactly two videos required.")
    cfg = get_config()
    if not models:
        models = cfg.models_dir()
    reports_dir = Path(output_root) if output_root else cfg.reports_dir()
    set_global_paths(models, reports_dir, reports_dir / "benchmark")
    inter_dir = compare_videos(vids[0], vids[1], output, duration, full, download_dir, output_root)
    click.echo(f"✅ Inter-video report: {inter_dir / 'dashboard.md'}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="multi_audit")
@click.option("--duration", "-d", default=10, type=int)
@click.option("--full", is_flag=True)
def benchmark(videos, output, duration, full):
    """Run audit on multiple videos and aggregate results."""
    from ..benchmark import audit_multiple_videos
    out_dir = audit_multiple_videos(list(videos), output, duration, full)
    click.echo(f"✅ Benchmark report: {out_dir / 'dashboard.md'}")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@cli.command(name="config")
@click.option("--show", is_flag=True, help="Print effective configuration as YAML")
@click.option("--validate", type=click.Path(exists=True),
              help="Validate a YAML configuration file")
@click.option("--paths", is_flag=True, help="Print resolved paths only")
def config_cmd(show, validate, paths):
    """Inspect / validate the configuration."""
    from ..core.config import get_config, load_config
    import yaml

    if validate:
        try:
            cfg = load_config(validate)
            click.echo(f"✅ {validate} is valid (modules: {len(cfg.modules)}, "
                       f"thresholds: {len(cfg.thresholds)})")
        except Exception as e:
            click.echo(f"❌ {validate} is invalid: {e}", err=True)
            raise SystemExit(1)
        return

    cfg = get_config()
    if paths:
        for k in cfg.paths:
            click.echo(f"{k:18s} -> {cfg.resolve_path(k)}")
        return
    if show:
        click.echo(yaml.dump(cfg.to_dict(), default_flow_style=False, sort_keys=False))
        return
    click.echo("Use --show, --validate FILE, or --paths.")


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@cli.command(name="datasets")
@click.option("--list", "list_all", is_flag=True, help="List registered datasets")
@click.option("--info", default=None, help="Print details about one dataset (key)")
@click.option("--check", is_flag=True, help="Check which datasets are present locally")
@click.option("--fetch", default=None, metavar="KEY",
              help="Download a dataset (uses partial-ZIP when supported)")
@click.option("--only", default=None, multiple=True, metavar="PATTERN",
              help="Glob pattern(s) for partial-ZIP fetch (e.g. '*.json'). Can repeat.")
@click.option("--list-files", default=None, metavar="KEY",
              help="List files inside a remote ZIP without downloading them")
@click.option("--force", is_flag=True, help="Re-download even if files exist")
def datasets_cmd(list_all, info, check, fetch, only, list_files, force):
    """Inspect / locate / fetch validation & calibration datasets.

    Some datasets have direct download URLs (PUBLIC); others are gated
    behind a registration form. ``--fetch`` only works on the former
    (those flagged ``auto_downloadable: true`` in the config).

    Partial-ZIP extraction:
        cineinfini datasets --fetch bvi_hfr --only "*.mp4"
        cineinfini datasets --fetch vbench_eval --only "VBench-master/prompts/*"

    Pre-flight inspection:
        cineinfini datasets --list-files bvi_hfr
    """
    from ..core.config import get_config
    from ..core.partial_zip import (
        HTTPRangeReader, list_remote_zip, extract_from_remote_zip,
    )
    cfg = get_config()

    # -------------------- list-files (no download) --------------------
    if list_files:
        entry = cfg.datasets.get(list_files)
        if not entry:
            click.echo(f"Unknown dataset key: {list_files}", err=True)
            raise SystemExit(1)
        url = entry.get("url")
        if not url or entry.get("format") != "zip":
            click.echo(f"{list_files} has no direct ZIP URL — see --info", err=True)
            raise SystemExit(1)
        click.echo(f"Listing files in {url} (no download)...")
        try:
            files = list_remote_zip(url)
        except Exception as e:
            click.echo(f"Failed: {e}", err=True)
            raise SystemExit(1)
        for f in files[:200]:
            size_str = f"{f['size'] / 1e6:7.1f}MB" if f["size"] else "       -"
            click.echo(f"  {size_str}  {f['name']}")
        if len(files) > 200:
            click.echo(f"  ... ({len(files) - 200} more)")
        click.echo(f"Total: {len(files)} entries")
        return

    # -------------------- fetch --------------------
    if fetch:
        entry = cfg.datasets.get(fetch)
        if not entry:
            click.echo(f"Unknown dataset key: {fetch}", err=True)
            raise SystemExit(1)
        if not entry.get("auto_downloadable"):
            click.echo(f"{fetch} is not auto-downloadable (registration "
                       f"required). See `cineinfini datasets --info {fetch}`.",
                       err=True)
            raise SystemExit(1)
        url = entry.get("url")
        target = cfg.dataset_dir(fetch)
        click.echo(f"Fetching {fetch} -> {target}")

        # ---- HuggingFace datasets path ----
        if entry.get("fetch_method") == "huggingface_datasets":
            click.echo(f"  Method: huggingface_datasets ({entry.get('size_mb', '?')} MB)")
            click.echo(f"  Homepage: {entry.get('homepage', '')}")
            try:
                from datasets import load_dataset
            except ImportError:
                click.echo("❌ The `datasets` library is required for this dataset.", err=True)
                click.echo("   Install with: pip install datasets", err=True)
                raise SystemExit(1)
            cmd = entry.get("fetch_command", "")
            # Parse simple form: load_dataset('repo', name='config')
            try:
                # Extract repo + name
                import re
                m = re.match(r"load_dataset\(['\"]([^'\"]+)['\"](?:,\s*name=['\"]([^'\"]+)['\"])?", cmd)
                if not m:
                    raise ValueError(f"Cannot parse fetch_command: {cmd!r}")
                repo, name = m.group(1), m.group(2)
                click.echo(f"  Loading: {repo}" + (f" (config: {name})" if name else ""))
                ds = load_dataset(repo, name=name) if name else load_dataset(repo)
                target.mkdir(parents=True, exist_ok=True)
                ds.save_to_disk(str(target))
                click.echo(f"✅ Saved to {target}")
                click.echo(f"   Splits: {list(ds.keys())}")
                for split_name, split in ds.items():
                    click.echo(f"     {split_name}: {len(split)} rows")
                videos_mirror = entry.get("videos_mirror")
                if videos_mirror:
                    click.echo(f"\n   Note: actual MP4 videos live at {videos_mirror}")
                    click.echo(f"   They are NOT downloaded by --fetch. Each sample has a")
                    click.echo(f"   'video link' field — pull videos individually as needed.")
            except Exception as e:
                click.echo(f"❌ Failed: {e}", err=True)
                raise SystemExit(1)
            return

        if not url:
            click.echo(f"{fetch} has no URL configured", err=True)
            raise SystemExit(1)
        click.echo(f"  URL: {url}")
        size_disp = entry.get('size_gb') or entry.get('size_mb')
        size_unit = 'GB' if entry.get('size_gb') else 'MB'
        click.echo(f"  Size: {size_disp or '?'} {size_unit}")

        patterns = list(only) if only else entry.get("partial_patterns")
        fmt = entry.get("format", "zip")

        if fmt == "zip" and patterns:
            click.echo(f"  Partial-ZIP mode, patterns: {patterns}")
            try:
                files = extract_from_remote_zip(
                    url, target_dir=target,
                    only=patterns, overwrite=force,
                )
                click.echo(f"✅ Extracted {len(files)} files to {target}")
            except Exception as e:
                click.echo(f"❌ Failed: {e}", err=True)
                raise SystemExit(1)
        else:
            # Full download fallback (would use core.bootstrap._download_url)
            click.echo("  Full download mode (no patterns / non-zip format)")
            from ..core.bootstrap import _download_url
            target.mkdir(parents=True, exist_ok=True)
            archive_path = target / Path(url).name
            try:
                _download_url(url, archive_path)
                click.echo(f"✅ Saved to {archive_path}")
                click.echo(f"   You may need to extract manually (format: {fmt})")
            except Exception as e:
                click.echo(f"❌ Failed: {e}", err=True)
                raise SystemExit(1)
        return

    # -------------------- info --------------------
    if info:
        entry = cfg.datasets.get(info)
        if not entry:
            click.echo(f"Unknown dataset key: {info}", err=True)
            click.echo(f"Available: {', '.join(sorted(cfg.datasets.keys()))}")
            raise SystemExit(1)
        click.echo(f"=== {entry.get('name', info)} ===")
        click.echo(f"Key:               {info}")
        for field in ("description", "purpose", "license", "size_gb",
                      "format", "auto_downloadable"):
            v = entry.get(field)
            if v is not None:
                click.echo(f"{field.capitalize():18s} {v}")
        for field in ("url", "registration_url", "homepage", "paper", "arxiv"):
            v = entry.get(field)
            if v:
                click.echo(f"{field.capitalize():18s} {v}")
        if entry.get("partial_patterns"):
            click.echo(f"{'Partial patterns':18s} {entry['partial_patterns']}")
        target = cfg.dataset_dir(info)
        click.echo(f"Expected at:       {target}")
        present = cfg.dataset_present(info)
        click.echo(f"Present locally:   {'✓ yes' if present else '✗ no'}")
        layout = entry.get("expected_layout") or {}
        if layout:
            click.echo("Expected layout:")
            for k, v in layout.items():
                click.echo(f"  {k}: {v}")
        if entry.get("citation"):
            click.echo("Citation:")
            click.echo(f"  {entry['citation']}")
        return

    if check:
        click.echo(f"Datasets directory: {cfg.datasets_dir()}")
        for key in sorted(cfg.datasets.keys()):
            entry = cfg.datasets[key]
            mark = "✓" if cfg.dataset_present(key) else "✗"
            target = cfg.dataset_dir(key)
            click.echo(f"  [{mark}] {key:18s} {entry.get('name','?'):20s} -> {target}")
        return

    # default: list
    click.echo(f"Registered datasets ({len(cfg.datasets)}):")
    for key in sorted(cfg.datasets.keys()):
        entry = cfg.datasets[key]
        present = "✓ present" if cfg.dataset_present(key) else "✗ missing"
        auto = "🔽 auto" if entry.get("auto_downloadable") else "🔒 gated"
        click.echo(f"  {key:18s} {entry.get('name','?'):20s} "
                   f"({entry.get('purpose','?'):14s}) {auto} {present}")
    click.echo()
    click.echo("Run `cineinfini datasets --info <key>` for details + URL.")
    click.echo("Run `cineinfini datasets --fetch <key>` to download (auto-only).")
    click.echo("Run `cineinfini datasets --fetch <key> --only PATTERN` for partial extract.")

# ---------------------------------------------------------------------------
# export-vbench / score
# ---------------------------------------------------------------------------
@cli.command(name="export-vbench")
@click.argument("audit_data", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output JSON path (default: <audit_data>.vbench.json)")
@click.option("--model-name", default="CineInfini",
              help="'model' field in the export (default: CineInfini)")
def export_vbench_cmd(audit_data, output, model_name):
    """Export an audit's data.json to VBench-compatible JSON.

    Maps CineInfini's native metrics onto VBench's 7 quality dimensions
    (subject_consistency, background_consistency, temporal_flickering,
    motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality).
    The 9 condition-consistency dimensions stay null because they need a
    prompt suite that's out of scope for a no-reference auditor.
    """
    from ..io.exporters import export_vbench_json
    src = Path(audit_data)
    if src.is_dir():
        src = src / "data.json"
    if not src.exists():
        raise click.UsageError(f"Audit data not found: {src}")
    data = json.loads(src.read_text())
    out = Path(output) if output else src.with_suffix(".vbench.json")
    export_vbench_json(data, out, model_name=model_name)
    click.echo(f"VBench export written to {out}")


@cli.command()
@click.argument("audit_data", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output JSON path for the scoring summary")
def score(audit_data, output):
    """Compute VideoScore-style 5-axis fusion + composite global score.

    Produces the same output shape as VideoScore (HuggingFace TIGER, 2024):
    Visual Quality, Temporal Consistency, Dynamic Degree, Text-to-Video
    Alignment, Factual Consistency — plus a single composite quality
    number for one-number ranking.
    """
    from ..aggregators import (
        compute_videoscore_axes, compute_global_score,
    )
    src = Path(audit_data)
    if src.is_dir():
        src = src / "data.json"
    if not src.exists():
        raise click.UsageError(f"Audit data not found: {src}")
    data = json.loads(src.read_text())
    axes = compute_videoscore_axes(data)
    composite = compute_global_score(axes)
    summary = {
        "video": data.get("video", {}).get("name") or src.parent.name,
        "videoscore_axes": axes,
        "composite_score": composite,
    }
    click.echo("=== VideoScore-style fusion ===")
    for k, v in axes.items():
        click.echo(f"  {k:30s} {v:.3f}" if v is not None else f"  {k:30s}   n/a")
    click.echo(f"  {'composite_score':30s} "
               f"{composite:.3f}" if composite is not None else f"  composite: n/a")
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str))
        click.echo(f"Written to {out}")


# ---------------------------------------------------------------------------
# calibrate / watch / list-modules / list-renderers
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--dataset", "-D", default="videofeedback",
              help="Calibration dataset key (default: videofeedback). "
                   "Run `cineinfini datasets` to see all available.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output path for calibration_report.json")
@click.option("--max-videos", type=int, default=50,
              help="Cap number of videos to audit (default: 50)")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None)
def calibrate(dataset, output, max_videos, config_path):
    """Calibrate CineInfini metrics against a labelled MOS dataset.

    Audits each video in the dataset and computes Spearman + Pearson
    correlation between every native metric and the human MOS labels
    in the dataset. Writes a calibration_report.json that ranks
    metrics by predictive power.

    Default dataset is 'videofeedback' (TIGER-Lab/VideoFeedback, auto-
    downloadable via `cineinfini datasets --fetch videofeedback`). For
    BVI-VFI access, fill the registration form linked in
    `cineinfini datasets --info t2vqa_db`.
    """
    from ..core.config import get_config, load_config, set_config
    from ..pipeline.orchestrator import run_audit
    from pathlib import Path as _P
    import math

    if config_path:
        set_config(load_config(config_path))
    cfg = get_config()
    ds_meta = (cfg.datasets or {}).get(dataset)
    if not ds_meta:
        raise click.UsageError(f"Unknown dataset '{dataset}'. "
                               f"Available: {list((cfg.datasets or {}).keys())}")
    ds_dir = _P(cfg.paths.get("datasets_dir", "~/.cineinfini/datasets")).expanduser() / dataset
    if not ds_dir.exists():
        click.echo(f"Dataset directory not found at {ds_dir}.")
        click.echo(f"Run `cineinfini datasets --fetch {dataset}` first.")
        raise SystemExit(1)
    # Find videos + labels in the dataset directory
    video_paths = sorted([p for p in ds_dir.rglob("*.mp4")] +
                          [p for p in ds_dir.rglob("*.mkv")])[:max_videos]
    labels_file = (ds_dir / "labels.json")
    if not video_paths:
        raise click.UsageError(f"No videos found under {ds_dir}")
    if not labels_file.exists():
        click.echo(f"Warning: {labels_file} not found — will compute metric "
                   f"distributions only, no correlation analysis.")
        labels = None
    else:
        labels = json.loads(labels_file.read_text())

    click.echo(f"Auditing {len(video_paths)} videos from '{dataset}'...")
    audits = []
    for vp in video_paths:
        try:
            data, _ = run_audit(vp)
            audits.append((vp.name, data))
            click.echo(f"  ✓ {vp.name}: composite = "
                       f"{data.get('composite_score', 'n/a')}")
        except Exception as e:
            click.echo(f"  ✗ {vp.name}: {e}")

    # Pearson correlation between each metric and MOS (when labels present)
    def pearson(xs, ys):
        n = len(xs)
        if n < 3:
            return None
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        return num / (dx*dy) if dx and dy else None

    correlations = {}
    if labels:
        # Collect metric values + MOS pairs
        all_metrics = set()
        for _, data in audits:
            for gate in (data.get("gates") or {}).values():
                for k, v in gate.items():
                    if isinstance(v, (int, float)) and k != "composite":
                        all_metrics.add(k)
        for metric in all_metrics:
            xs, ys = [], []
            for vname, data in audits:
                if vname not in labels:
                    continue
                # average metric across shots for this video
                vals = []
                for gate in (data.get("gates") or {}).values():
                    if isinstance(gate.get(metric), (int, float)):
                        vals.append(gate[metric])
                if vals:
                    xs.append(sum(vals) / len(vals))
                    ys.append(float(labels[vname]))
            r = pearson(xs, ys)
            if r is not None:
                correlations[metric] = {"pearson": r, "n": len(xs)}

    out_path = _P(output) if output else (cfg.reports_dir() / f"calibration_{dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": dataset,
        "n_videos_audited": len(audits),
        "n_videos_with_labels": len([1 for n, _ in audits if labels and n in labels]) if labels else 0,
        "correlations": dict(sorted(correlations.items(),
                                    key=lambda kv: -abs(kv[1]["pearson"]))),
    }
    out_path.write_text(json.dumps(report, indent=2, default=str))
    click.echo(f"\n✅ Calibration report: {out_path}")
    if correlations:
        click.echo(f"\nTop 5 metrics by |Pearson| with MOS:")
        for metric, stats in list(report["correlations"].items())[:5]:
            click.echo(f"  {metric:30s} r = {stats['pearson']:+.3f} (n={stats['n']})")


@cli.command()
@click.argument("watch_dir", type=click.Path(exists=True))
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None)
@click.option("--interval", default=2.0, type=float,
              help="Polling interval in seconds (default: 2.0)")
@click.option("--extensions", default=".mp4,.mkv,.mov,.webm",
              help="Comma-separated extensions to watch (default: .mp4,.mkv,.mov,.webm)")
def watch(watch_dir, config_path, interval, extensions):
    """Watch a directory and audit any new video as it appears.

    Polls the directory every `--interval` seconds. When a new video
    file (matching the configured extensions) is detected, runs an
    audit on it and writes the report next to it. Exit with Ctrl-C.

    Useful for live monitoring of T2V generation pipelines or
    streaming QC.
    """
    import time as _time
    from ..core.config import load_config, set_config
    from ..pipeline.orchestrator import run_audit

    if config_path:
        set_config(load_config(config_path))
    watch_dir = Path(watch_dir).resolve()
    exts = tuple(e.strip() for e in extensions.split(","))
    seen = set(p for p in watch_dir.iterdir() if p.suffix.lower() in exts)
    click.echo(f"Watching {watch_dir} (interval={interval}s, ext={exts})")
    click.echo(f"Initial files (will not be re-audited): {len(seen)}")
    click.echo("Press Ctrl-C to stop.")
    try:
        while True:
            _time.sleep(interval)
            for p in watch_dir.iterdir():
                if p.suffix.lower() not in exts or p in seen:
                    continue
                seen.add(p)
                click.echo(f"\n[{p.name}] new video detected — auditing")
                try:
                    data, out_dir = run_audit(p)
                    click.echo(f"  ✓ composite = {data.get('composite_score', 'n/a')}, "
                               f"report: {out_dir}/dashboard.html")
                except Exception as e:
                    click.echo(f"  ✗ failed: {e}")
    except KeyboardInterrupt:
        click.echo("\nStopped.")


@cli.command(name="list-modules")
def list_modules_cmd():
    """List every registered audit module with description and version."""
    import cineinfini.modules  # noqa: F401  ensure registry populated
    from ..core.registry import all_modules
    from ..core.config import get_config
    cfg = get_config()
    mods = all_modules()
    click.echo(f"Registered modules: {len(mods)}")
    click.echo("")
    for mod_id, entry in sorted(mods.items()):
        enabled = cfg.modules.get(mod_id, {}).get("enabled", False)
        mark = "✓ ON " if enabled else "  off"
        desc = (entry.description or "")[:60]
        click.echo(f"  [{mark}] {mod_id:28s} v{entry.version:8s} {desc}")


@cli.command(name="list-renderers")
def list_renderers_cmd():
    """List every registered output renderer."""
    import cineinfini.io.renderers  # noqa: F401
    from ..core.ui_registry import all_renderers
    from ..core.config import get_config
    cfg = get_config()
    rends = all_renderers()
    active = set(cfg.active_renderers())
    click.echo(f"Registered renderers: {len(rends)}")
    click.echo("")
    for rid, entry in sorted(rends.items()):
        mark = "✓ ON " if rid in active else "  off"
        desc = (entry.description or "")[:60] if hasattr(entry, "description") else ""
        click.echo(f"  [{mark}] {rid:18s} {desc}")


# ---------------------------------------------------------------------------
# watch — re-audit a directory whenever videos change
# ---------------------------------------------------------------------------
@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None,
              help="YAML config file (or profile)")
@click.option("--output", "-o", default=None,
              help="Output directory (default: cfg.reports_dir)")
@click.option("--interval", default=2.0, type=float,
              help="Polling interval in seconds (default: 2.0)")
@click.option("--patterns", default="*.mp4,*.mov,*.mkv,*.webm",
              help="Comma-separated glob patterns (default: video formats)")
@click.option("--audit-existing", is_flag=True,
              help="Audit files already present at startup (default: only new ones)")
def watch(directory, config_path, output, interval, patterns, audit_existing):
    """Watch DIRECTORY for new videos and audit each one as it appears.

    Uses stdlib polling (no inotify/watchdog dependency). Audits each
    matching file once, then waits for new files. Exit with Ctrl-C.
    """
    import time
    from fnmatch import fnmatch
    from ..pipeline.audit import audit_video, set_global_paths, CONFIG
    from ..core.config import get_config, load_config, set_config

    if config_path:
        set_config(load_config(config_path))
    cfg = get_config()
    out = Path(output) if output else cfg.reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    set_global_paths(out, out / "benchmark")

    pat_list = [p.strip() for p in patterns.split(",") if p.strip()]
    watch_dir = Path(directory)
    seen: set = set()
    if not audit_existing:
        for f in watch_dir.iterdir():
            if any(fnmatch(f.name, p) for p in pat_list):
                seen.add(f.name)
        click.echo(f"Watching {watch_dir} (skipping {len(seen)} existing files)")

    click.echo(f"Patterns: {pat_list}")
    click.echo(f"Output:   {out}")
    click.echo("Press Ctrl-C to stop.")
    try:
        while True:
            for f in sorted(watch_dir.iterdir()):
                if f.name in seen:
                    continue
                if not any(fnmatch(f.name, p) for p in pat_list):
                    continue
                click.echo(f"\n→ Auditing {f.name}")
                try:
                    _, report_dir = audit_video(f)
                    click.echo(f"  ✓ Report: {report_dir}")
                except Exception as e:
                    click.echo(f"  ✗ Failed: {e}", err=True)
                seen.add(f.name)
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")


# ---------------------------------------------------------------------------
# serve — minimal HTTP service for audit (stdlib only)
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--port", default=8765, type=int, help="Port")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None,
              help="YAML config to load at startup")
@click.option("--output", "-o", default=None,
              help="Output directory for reports (default: cfg.reports_dir)")
def serve(host, port, config_path, output):
    """Run a minimal HTTP audit service (no Flask/FastAPI dependency).

    Endpoints:
      GET  /health                    -> 200 OK + version
      GET  /info                      -> active modules + profile info
      POST /audit?path=/abs/video.mp4 -> JSON audit data

    Returns the same JSON as `cineinfini audit`. Single-threaded; for
    production use put a real WSGI server in front.
    """
    import http.server
    import urllib.parse
    from ..pipeline.audit import audit_video, set_global_paths
    from ..core.config import get_config, load_config, set_config

    if config_path:
        set_config(load_config(config_path))
    cfg = get_config()
    out = Path(output) if output else cfg.reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    set_global_paths(out, out / "benchmark")

    from .. import __version__ as version

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict):
            body = json.dumps(payload, indent=2, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            click.echo(f"[serve] {self.address_string()} {fmt % args}")

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                return self._json(200, {"status": "ok", "version": version})
            if parsed.path == "/info":
                return self._json(200, {
                    "version": version,
                    "active_modules": cfg.enabled_modules(),
                    "active_renderers": cfg.active_renderers(),
                })
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/audit":
                return self._json(404, {"error": "not found"})
            qs = urllib.parse.parse_qs(parsed.query)
            path = (qs.get("path") or [None])[0]
            if not path or not Path(path).exists():
                return self._json(400, {"error": "missing or invalid 'path' query parameter"})
            try:
                data, report_dir = audit_video(Path(path))
                return self._json(200, {
                    "report_dir": str(report_dir),
                    "video": data.get("video_name"),
                    "n_shots": data.get("n_shots"),
                    "composite_score": data.get("composite_score"),
                    "videoscore_axes": data.get("videoscore_axes"),
                    "modules": list((data.get("modules") or {}).keys()),
                })
            except Exception as e:
                return self._json(500, {"error": str(e)})

    click.echo(f"CineInfini {version} serving at http://{host}:{port}")
    click.echo("  GET  /health")
    click.echo("  GET  /info")
    click.echo("  POST /audit?path=/abs/path/video.mp4")
    click.echo("Press Ctrl-C to stop.")
    server = http.server.HTTPServer((host, port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
        server.server_close()


# ---------------------------------------------------------------------------
# calibrate — Spearman / Pearson against MOS labels
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--labels-csv", required=True, type=click.Path(exists=True),
              help="CSV with columns: video_path, mos (or video, score)")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None,
              help="YAML config to use for the audits")
@click.option("--output", "-o", default="calibration_report.json",
              type=click.Path(),
              help="Output report path (default: calibration_report.json)")
@click.option("--metric", default="composite_score",
              help="Which audit field to correlate against MOS "
                   "(default: composite_score)")
@click.option("--limit", default=0, type=int,
              help="Audit at most N videos (0 = all)")
def calibrate(labels_csv, config_path, output, metric, limit):
    """Compute Spearman + Pearson correlation between a CineInfini metric
    and human MOS labels from a CSV file.

    Expected CSV format (header required):

        video_path,mos
        /abs/path/v1.mp4,3.8
        /abs/path/v2.mp4,2.1

    Alternative column names are accepted: 'video' for video_path,
    'score' or 'dmos' for mos.

    Output: a JSON report with per-video predictions and
    correlation coefficients.
    """
    import csv
    import math
    from ..pipeline.audit import audit_video, set_global_paths
    from ..core.config import get_config, load_config, set_config

    if config_path:
        set_config(load_config(config_path))
    cfg = get_config()
    set_global_paths(cfg.reports_dir(), cfg.reports_dir() / "benchmark")

    # Read labels CSV
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise click.UsageError("Labels CSV is empty")

    # Resolve column aliases
    cols = {c.lower(): c for c in rows[0].keys()}
    path_col = cols.get("video_path") or cols.get("video")
    mos_col = cols.get("mos") or cols.get("score") or cols.get("dmos")
    if not (path_col and mos_col):
        raise click.UsageError(
            f"CSV must have columns video_path/video and mos/score/dmos. "
            f"Got: {list(rows[0].keys())}"
        )

    if limit > 0:
        rows = rows[:limit]
    click.echo(f"Calibrating against {len(rows)} videos...")

    results = []
    for i, r in enumerate(rows):
        path = r[path_col]
        try:
            mos = float(r[mos_col])
        except ValueError:
            click.echo(f"  [skip] non-numeric MOS: {r[mos_col]!r}", err=True)
            continue
        if not Path(path).exists():
            click.echo(f"  [skip] missing video: {path}", err=True)
            continue
        try:
            data, _ = audit_video(Path(path))
            pred = data.get(metric)
            if pred is None:
                # try modules.<X>.mean_*
                for mod in (data.get("modules") or {}).values():
                    if isinstance(mod, dict) and metric in mod:
                        pred = mod[metric]
                        break
            results.append({
                "video": Path(path).name,
                "mos": mos,
                "prediction": pred,
            })
            click.echo(f"  [{i+1}/{len(rows)}] {Path(path).name} mos={mos} pred={pred}")
        except Exception as e:
            click.echo(f"  [fail] {path}: {e}", err=True)

    # Correlations
    valid = [(r["mos"], r["prediction"]) for r in results
             if r["prediction"] is not None]
    if len(valid) < 3:
        click.echo("Not enough valid pairs for correlation (need ≥ 3)", err=True)
        spearman = pearson = None
    else:
        xs = [v[0] for v in valid]
        ys = [v[1] for v in valid]
        # Pearson
        n = len(xs)
        mx, my = sum(xs)/n, sum(ys)/n
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        pearson = num / (dx*dy) if dx and dy else None
        # Spearman = Pearson over ranks
        def rank(arr):
            sorted_pairs = sorted(enumerate(arr), key=lambda p: p[1])
            ranks = [0.0] * len(arr)
            for r, (orig_i, _) in enumerate(sorted_pairs, start=1):
                ranks[orig_i] = r
            return ranks
        rx, ry = rank(xs), rank(ys)
        n = len(rx)
        mx, my = sum(rx)/n, sum(ry)/n
        num = sum((x-mx)*(y-my) for x, y in zip(rx, ry))
        dx = math.sqrt(sum((x-mx)**2 for x in rx))
        dy = math.sqrt(sum((y-my)**2 for y in ry))
        spearman = num / (dx*dy) if dx and dy else None

    report = {
        "metric": metric,
        "n_videos": len(rows),
        "n_valid": len(valid),
        "pearson": pearson,
        "spearman": spearman,
        "results": results,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    click.echo(f"\n=== Calibration report ===")
    click.echo(f"  metric:   {metric}")
    click.echo(f"  videos:   {len(valid)}/{len(rows)} valid pairs")
    if pearson is not None:
        click.echo(f"  Pearson:  {pearson:.3f}")
        click.echo(f"  Spearman: {spearman:.3f}")
    click.echo(f"  Saved to: {out_path}")


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
@click.option("--port", default=8000, type=int, help="Bind port (default: 8000)")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), default=None)
def serve(host, port, config_path):
    """Run a small REST API exposing audit, score, and export-vbench.

    Endpoints (when FastAPI is installed):
      POST /audit          form-data: video=<file>           -> data.json
      POST /score          json: <audit_data>                -> 5 axes + composite
      POST /export-vbench  json: <audit_data>                -> VBench JSON
      GET  /health                                            -> {"ok": true}
      GET  /modules                                           -> registered modules
      GET  /renderers                                         -> registered renderers

    Graceful degradation: when FastAPI is not installed, prints the
    install one-liner and exits with non-zero status. The `--help`
    output is always available regardless.
    """
    try:
        import fastapi  # noqa: F401
        import uvicorn   # noqa: F401
    except ImportError:
        click.echo("serve requires FastAPI + uvicorn. Install with:")
        click.echo("    pip install 'cineinfini-audit[serve]'")
        click.echo("or directly:")
        click.echo("    pip install fastapi uvicorn[standard] python-multipart")
        raise SystemExit(2)

    from ..core.config import load_config, set_config, get_config
    if config_path:
        set_config(load_config(config_path))

    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse
    import tempfile, shutil

    app = FastAPI(title="CineInfini", version=get_config().__class__.__module__.rsplit(".", 1)[0])

    @app.get("/health")
    def health():
        return {"ok": True, "service": "cineinfini"}

    @app.get("/modules")
    def modules_endpoint():
        import cineinfini.modules  # noqa: F401
        from ..core.registry import all_modules
        cfg = get_config()
        return {
            mod_id: {
                "version": e.version, "description": e.description,
                "enabled": cfg.modules.get(mod_id, {}).get("enabled", False),
            }
            for mod_id, e in all_modules().items()
        }

    @app.get("/renderers")
    def renderers_endpoint():
        import cineinfini.io.renderers  # noqa: F401
        from ..core.ui_registry import all_renderers
        return {rid: {"description": getattr(e, "description", "")}
                for rid, e in all_renderers().items()}

    @app.post("/audit")
    async def audit_endpoint(video: UploadFile = File(...)):
        from ..pipeline.orchestrator import run_audit
        with tempfile.NamedTemporaryFile(suffix=Path(video.filename).suffix, delete=False) as f:
            shutil.copyfileobj(video.file, f)
            tmp_path = f.name
        try:
            data, _ = run_audit(tmp_path)
            return JSONResponse(content=json.loads(json.dumps(data, default=str)))
        finally:
            try: Path(tmp_path).unlink()
            except Exception: pass

    @app.post("/score")
    def score_endpoint(audit_data: dict):
        from ..aggregators import compute_videoscore_axes, compute_global_score
        axes = compute_videoscore_axes(audit_data)
        composite = compute_global_score(axes)
        return {"videoscore_axes": axes, "composite_score": composite}

    @app.post("/export-vbench")
    def export_vbench_endpoint(audit_data: dict):
        from ..io.exporters import map_to_vbench
        return {"scores": map_to_vbench(audit_data)}

    click.echo(f"CineInfini server starting on http://{host}:{port}")
    click.echo("Endpoints: /health /modules /renderers /audit /score /export-vbench")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
