"""Command-line interface for CineInfini"""
import click
from pathlib import Path
from ..pipeline.audit import audit_video, set_global_paths, CONFIG

@click.group()
def cli():
    pass

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None, help='Output directory for reports')
@click.option('--models', '-m', default=None, help='Models directory')
@click.option('--duration', '-d', default=60, type=int, help='Max duration to analyze (seconds)')
@click.option('--full', is_flag=True, help='Analyse entire video (ignore duration limit)')
def audit(video_path, output, models, duration, full):
    """Run video quality audit."""
    # Set paths (to be implemented)
    if not models:
        models = Path.home() / ".cineinfini/models"
    if not output:
        output = Path.cwd() / "reports"
    set_global_paths(models, output, output / "benchmark")
    if full:
        CONFIG["max_duration_s"] = 999999
    else:
        CONFIG["max_duration_s"] = duration
    # Run audit
    metrics, report_dir = audit_video(video_path)
    click.echo(f"Report saved to {report_dir}")

if __name__ == "__main__":
    cli()
