"""Command-line interface for CineInfini."""
import click
from pathlib import Path
from ..pipeline.audit import audit_video, set_global_paths, CONFIG

@click.group()
@click.version_option()
def cli():
    """CineInfini – Video Quality Audit Pipeline."""
    pass

@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None)
@click.option("--models", "-m", default=None)
@click.option("--duration", "-d", default=60, type=int)
@click.option("--full", is_flag=True)
def audit(video_path, output, models, duration, full):
    if not models: models = Path.home() / ".cineinfini/models"
    if not output: output = Path.cwd() / "reports"
    set_global_paths(models, output, output / "benchmark")
    CONFIG["max_duration_s"] = 999999 if full else duration
    metrics, report_dir = audit_video(video_path)
    click.echo(f"✅ Report saved to {report_dir}")

@cli.command()
@click.option("--vids", "-v", multiple=True, required=True)
@click.option("--output", "-o", default="comparison")
@click.option("--duration", "-d", default=10, type=int)
@click.option("--full", is_flag=True)
@click.option("--download-dir", default=None)
@click.option("--output-root", default=None)
@click.option("--models", "-m", default=None)
def compare(vids, output, duration, full, download_dir, output_root, models):
    if len(vids) != 2:
        raise click.BadParameter("Exactly two videos required.")
    if not models: models = Path.home() / ".cineinfini/models"
    reports_dir = Path(output_root) if output_root else Path.cwd() / "reports"
    set_global_paths(models, reports_dir, reports_dir / "benchmark")
    from ..compare import compare_videos
    inter_dir = compare_videos(vids[0], vids[1], output, duration, full, download_dir, output_root)
    click.echo(f"✅ Inter-video report: {inter_dir / 'dashboard.md'}")

@cli.command()
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="multi_audit")
@click.option("--duration", "-d", default=10, type=int)
@click.option("--full", is_flag=True)
def benchmark(videos, output, duration, full):
    from ..benchmark import audit_multiple_videos
    out_dir = audit_multiple_videos(list(videos), output, duration, full)
    click.echo(f"✅ Benchmark report: {out_dir / 'dashboard.md'}")

if __name__ == "__main__": cli()
