#!/usr/bin/env python
"""Utility script to run an audit on a video file."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cineinfini import adaptive_multi_stage_audit

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_audit.py <video_path> [--full]")
        sys.exit(1)
    video = sys.argv[1]
    full = "--full" in sys.argv
    report_dir = adaptive_multi_stage_audit(video, force_full_video=full)
    print(f"Report saved to {report_dir}")
