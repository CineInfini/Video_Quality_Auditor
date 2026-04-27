"""
CineInfini – Video comparison tool (local files or URLs)
"""

import urllib.request
from pathlib import Path
from typing import Union

from cineinfini import audit_video
from cineinfini.io.report import generate_inter_report
from cineinfini.pipeline.audit import CONFIG

def compare_videos(
    video1: Union[str, Path],
    video2: Union[str, Path],
    output_subdir: str = "comparison",
    max_duration_s: int = 10,
    force_full_video: bool = False,
    download_dir: Union[str, Path] = None,
    output_root: Union[str, Path] = None,
) -> Path:
    if download_dir is None:
        download_dir = Path.cwd()
    else:
        download_dir = Path(download_dir).expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    if output_root is None:
        inter_root = Path.cwd() / "reports" / "inter"
    else:
        inter_root = Path(output_root).expanduser().resolve()
    inter_root.mkdir(parents=True, exist_ok=True)

    def resolve_path(source: Union[str, Path]) -> Path:
        source = str(source)
        if source.startswith(("http://", "https://")):
            local_name = source.split("/")[-1].split("?")[0]
            if not local_name or "." not in local_name:
                local_name = "downloaded_video.mp4"
            local_path = download_dir / local_name
            if not local_path.exists():
                print(f"  Downloading {source} ...")
                urllib.request.urlretrieve(source, local_path)
                print(f"  -> {local_path}")
            return local_path
        return Path(source).expanduser().resolve()

    v1_path = resolve_path(video1)
    v2_path = resolve_path(video2)

    video_params = {} if force_full_video else {"max_duration_s": max_duration_s}

    print(f"\n🔍 Auditing {v1_path.name}...")
    _, report1 = audit_video(str(v1_path), video_params=video_params, force_full_video=force_full_video)
    print(f"   Intra report: {report1}")

    print(f"\n🔍 Auditing {v2_path.name}...")
    _, report2 = audit_video(str(v2_path), video_params=video_params, force_full_video=force_full_video)
    print(f"   Intra report: {report2}")

    print("\n📊 Generating comparison report...")
    generate_inter_report(
        intra_report_dirs=[report1, report2],
        output_dir=inter_root,
        thresholds=CONFIG["thresholds"],
        comparison_name=output_subdir
    )
    inter_dir = inter_root / output_subdir
    print(f"✅ Inter report created: {inter_dir / 'dashboard.md'}")
    return inter_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video1")
    parser.add_argument("video2")
    parser.add_argument("--output", "-o", default="comparison")
    parser.add_argument("--duration", "-d", type=int, default=10)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    compare_videos(args.video1, args.video2, args.output, args.duration, args.full)
