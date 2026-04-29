#!/usr/bin/env python3
"""Sandbox-side ingestion of the ML feature tarball produced by
CineInfini_ML_Extract_Colab.ipynb.

This script:
  1. Locates uploaded tarball parts at /mnt/user-data/uploads/
  2. Reassembles them with cat (no-op if single file)
  3. Verifies the SHA256 against SHA256SUMS.txt (if uploaded)
  4. Untars to /tmp/cineinfini_ml_features/
  5. Validates content shape and reports per-dataset coverage
  6. Loads scalar features and prints quick Spearman ρ vs MOS

Usage:
    python scripts/ingest_ml_features.py \\
        --uploads-dir /mnt/user-data/uploads \\
        --extract-dir /tmp/cineinfini_ml_features
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def reassemble(uploads_dir: Path, target: Path) -> Path:
    """If parts present, cat them into target. Otherwise copy single tarball."""
    parts = sorted(uploads_dir.glob('cineinfini_ml_features.tar.gz.part*'))
    single = uploads_dir / 'cineinfini_ml_features.tar.gz'
    if parts:
        print(f'  reassembling {len(parts)} parts: '
              f'{[p.name for p in parts]}')
        with target.open('wb') as out:
            for p in parts:
                with p.open('rb') as f:
                    while True:
                        buf = f.read(1 << 20)
                        if not buf:
                            break
                        out.write(buf)
        print(f'  → {target} ({target.stat().st_size / 1e6:.1f} MB)')
    elif single.exists():
        if target.exists():
            target.unlink()
        target.symlink_to(single.resolve())
        print(f'  → {target} (single file, {single.stat().st_size / 1e6:.1f} MB)')
    else:
        print(f'  ERROR: neither parts nor single tarball found in {uploads_dir}',
              file=sys.stderr)
        sys.exit(1)
    return target


def verify_sha256(uploads_dir: Path, tarball: Path) -> bool:
    """If SHA256SUMS.txt is present, verify against it."""
    sums_file = uploads_dir / 'SHA256SUMS.txt'
    if not sums_file.exists():
        print('  SHA256SUMS.txt not uploaded — skipping verification')
        return True
    expected = {}
    for ln in sums_file.read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            expected[parts[1]] = parts[0]
    name = 'cineinfini_ml_features.tar.gz'
    if name in expected:
        actual = sha256_file(tarball)[: len(expected[name])]
        match = (actual == expected[name])
        status = '✓' if match else '✗'
        print(f'  SHA256 {status}: expected {expected[name]}, got {actual}')
        return match
    print('  SHA256SUMS.txt does not list the assembled tarball')
    return True


def untar(tarball: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        tar.extractall(extract_dir)
    inner = extract_dir / 'cineinfini_ml_features'
    if not inner.exists():
        # Find the actual extracted folder
        candidates = list(extract_dir.iterdir())
        if candidates:
            inner = candidates[0]
    print(f'  extracted → {inner}')
    return inner


def validate_content(features_dir: Path) -> dict:
    """Verify expected files, report shapes."""
    expected = {
        'clip_features.npz': 'CLIP features',
        'dinov2_features.npz': 'DINOv2 features',
        'arcface_features.npz': 'ArcFace features',
        'ml_scalar_features.jsonl': 'scalar features',
        'manifest.jsonl': 'manifest',
    }
    report = {}
    for fname, desc in expected.items():
        p = features_dir / fname
        if not p.exists():
            print(f'  ✗ MISSING: {fname}')
            report[fname] = None
            continue
        size_mb = p.stat().st_size / 1e6
        if fname.endswith('.npz'):
            data = np.load(p, allow_pickle=True)
            n_videos = len(data.files)
            sample_shape = data[data.files[0]].shape if data.files else None
            print(f'  ✓ {desc}: {n_videos} videos, shape {sample_shape}, {size_mb:.2f} MB')
            report[fname] = {'n_videos': n_videos, 'shape': sample_shape, 'size_mb': size_mb}
        else:
            n_lines = sum(1 for _ in p.open())
            print(f'  ✓ {desc}: {n_lines} lines, {size_mb:.2f} MB')
            report[fname] = {'n_lines': n_lines, 'size_mb': size_mb}
    return report


def quick_correlation_preview(features_dir: Path):
    """Compute Spearman ρ vs MOS per metric per dataset for a sanity check."""
    from scipy.stats import spearmanr

    p = features_dir / 'ml_scalar_features.jsonl'
    if not p.exists():
        return
    rows = [json.loads(ln) for ln in p.read_text().splitlines()]
    print(f'\n=== Spearman ρ vs MOS (sanity-check on uploaded features) ===')
    for ds in ['t2vqa', 'konvid', 'videofeedback']:
        sub = [r for r in rows if r.get('dataset') == ds]
        if not sub:
            print(f'\n{ds}: 0 videos')
            continue
        print(f'\n{ds} (n={len(sub)}):')
        for k in ['clipscore', 'clip_consistency', 'clip_consistency_min',
                  'dinov2_consistency', 'dinov2_consistency_min',
                  'identity_dtw', 'identity_drift_max']:
            vals = np.array([r.get(k, np.nan) for r in sub], dtype=float)
            mos = np.array([r['mos'] for r in sub], dtype=float)
            mask = np.isfinite(vals) & np.isfinite(mos)
            if mask.sum() < 5:
                print(f'  {k:<28} (insufficient data)')
                continue
            rho, p_val = spearmanr(vals[mask], mos[mask])
            sig = '*** ' if p_val < 0.001 else ('**  ' if p_val < 0.01
                  else ('*   ' if p_val < 0.05 else '    '))
            print(f'  {k:<28} ρ={rho:+.3f}  p={p_val:.4f}  '
                  f'(n={mask.sum():3d}) {sig}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--uploads-dir', type=Path,
                    default=Path('/mnt/user-data/uploads'))
    ap.add_argument('--extract-dir', type=Path,
                    default=Path('/tmp/cineinfini_ml_features_extracted'))
    ap.add_argument('--tarball-tmp', type=Path,
                    default=Path('/tmp/_features_assembled.tar.gz'))
    args = ap.parse_args()

    print(f'== CineInfini ML feature ingestion ==')
    print(f'\n[1/5] Locate and reassemble tarball')
    tarball = reassemble(args.uploads_dir, args.tarball_tmp)

    print(f'\n[2/5] Verify SHA256')
    verify_sha256(args.uploads_dir, tarball)

    print(f'\n[3/5] Extract')
    features_dir = untar(tarball, args.extract_dir)

    print(f'\n[4/5] Validate content')
    report = validate_content(features_dir)

    print(f'\n[5/5] Sanity-check correlations')
    quick_correlation_preview(features_dir)

    # Write a status file for downstream scripts
    status = {
        'features_dir': str(features_dir),
        'report': report,
    }
    status_path = features_dir / '_ingestion_status.json'
    status_path.write_text(json.dumps(status, indent=2))
    print(f'\nStatus written to {status_path}')
    print(f'\n✓ Features ready for hybrid analysis at {features_dir}')


if __name__ == '__main__':
    main()
