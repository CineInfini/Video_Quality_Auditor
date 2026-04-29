# CineInfini — Complete User Manual (v0.4.8.4)

This is the authoritative reference for using CineInfini. It covers
everything from first install to advanced usage. For shorter overviews
see [`QUICKSTART.md`](QUICKSTART.md) and [`INSTALLATION.md`](INSTALLATION.md).

## Table of contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [The CLI — eight commands](#3-the-cli)
4. [The five performance profiles](#4-performance-profiles)
5. [The 21 audit modules](#5-modules)
6. [Output formats](#6-outputs)
7. [Configuration system](#7-configuration)
8. [Cross-benchmark integrations](#8-cross-benchmark)
9. [Datasets and validation](#9-datasets)
10. [Programmatic API](#10-api)
11. [Troubleshooting](#11-troubleshooting)
12. [Going deeper](#12-going-deeper)

---

## 1. Overview

CineInfini is a **no-reference, modular, multi-dimensional auditor** for
videos — natural or AI-generated. A single `cineinfini audit` run produces:

- **Native CineInfini metrics** (19 of them, organised across 21 registered modules)
- **VBench-compatible JSON** (16 dimensions, 7 measured directly)
- **VideoScore-style 5-axis fusion + 1 composite global score**
- **DOVER aesthetic + technical** (when the optional wrapper is wired)
- **FAST-VQA quality score** (idem)

Per-shot verdicts (ACCEPT / REVIEW / REJECT / BLOCKED) tell you *which*
shots have problems and *why*, instead of one black-box number.

Architecture in one picture:

```
   video.mp4
      │
      ▼
   [shot detection] ──► [frame extraction]
      │                       │
      ▼                       ▼
   ┌─────────────────────────────────┐
   │  Audit registry                 │
   │  (cfg.modules.<X>.enabled = ?)  │
   ├─────────────────────────────────┤
   │  motion_coherence               │
   │  identity_consistency (ArcFace+DTW)  │
   │  semantic_consistency (CLIP)    │
   │  background_consistency (SSIM)  │
   │  aesthetic_cinematic            │
   │  causal_reasoning               │
   │  ... 15 more native modules     │
   │  dover_score (wrapper)          │
   │  fastvqa_score (wrapper)        │
   └────────────┬────────────────────┘
                ▼
   ┌─────────────────────────────────┐
   │  Aggregators                    │
   │  - phase4 (ACCEPT/REVIEW/REJECT) │
   │  - VideoScore 5-axis fusion     │
   │  - VBench 16-dim mapping        │
   └────────────┬────────────────────┘
                ▼
   ┌─────────────────────────────────┐
   │  Renderers                      │
   │  data.json / dashboard.html /   │
   │  report.pdf / dashboard.md /    │
   │  benchmark_report (multi-video) │
   └─────────────────────────────────┘
```

Every box is configurable via `cfg/config.yaml`. Every module is
registered with `@register_module` and can be enabled/disabled
independently. None are required at run-time — disabling all of them
gives you an empty (but valid) audit.

---

## 2. Installation

### 2.1 Minimal install (pure-CV only)

```bash
pip install cineinfini-audit
```

Gives you the framework + 16 pure-CV modules. No external weights are
required for the 3 default-on modules to produce meaningful output on
small videos.

### 2.2 With ML modules

```bash
cineinfini bootstrap
```

Downloads the four Tier-1 weights (~835 MB total):

| Asset | Size | Used by |
|---|---|---|
| `arcface.onnx` | 166 MB | identity_consistency, subject_consistency_long |
| `yunet.onnx` | 232 KB | face detection |
| `clip_vit_b32` | 338 MB | semantic_consistency, multi_modal_safety |
| `dinov2_vitb14` | 330 MB | long_term_narrative, origin_detection |

Re-running `bootstrap` is idempotent — already-present files are skipped
unless `--force` is passed.

### 2.3 With cross-benchmark wrappers (DOVER, FAST-VQA)

```bash
pip install torch torchvision
pip install dover-vqa                           # or git clone + pip install -e .
pip install fast-vqa                            # or git clone + pip install -e .
cineinfini bootstrap --include-optional         # +395 MB
```

The wrappers will then report `available: true` when the audit runs
their modules and surface DOVER's aesthetic+technical and FAST-VQA's
quality score in the same `gates` structure as the native metrics.

### 2.4 System requirements

- Python ≥ 3.9
- ffmpeg ≥ 4.0 (auto-detected; CineInfini won't install it but warns if missing)
- ~1 GB free disk for Tier-1 weights, +395 MB for optional models
- GPU optional but recommended for `academic.yaml` profile

---

## 3. The CLI

CineInfini exposes 8 top-level commands. Run `cineinfini --help` for the
list and `cineinfini <cmd> --help` for per-command flags.

### 3.1 `cineinfini audit <video>`

The main entry point. Audits a single video.

```bash
cineinfini audit input.mp4 \
    --config cfg/profiles/postproduction.yaml \
    --output ./reports/ \
    --duration 60                # optional: cap audit at 60s of video
```

Important flags:

| Flag | What |
|---|---|
| `--config / -c` | YAML config (or profile) to use |
| `--output / -o` | Output directory (default: `~/.cineinfini/reports/<video_stem>/`) |
| `--duration` | Cap audit at N seconds of input |
| `--full` | Force full-video processing even on long inputs |
| `--auto-bootstrap` | Run `bootstrap` first if any required model is missing |

### 3.2 `cineinfini compare <v1> <v2>`

Side-by-side audit producing a comparison report.

```bash
cineinfini compare original.mp4 generated.mp4 --output ./diff/
```

### 3.3 `cineinfini benchmark <directory>`

Audit multiple videos and aggregate. Produces a single `benchmark_report.html`
plus per-video JSON.

```bash
cineinfini benchmark ./ai_videos/ --config cfg/profiles/postproduction.yaml
```

### 3.4 `cineinfini bootstrap`

Fetches ffmpeg/models/test-videos. Already-present files are skipped.

```bash
cineinfini bootstrap                          # everything
cineinfini bootstrap --models-only            # just the 4 Tier-1 weights
cineinfini bootstrap --videos-only            # just the test videos
cineinfini bootstrap --include-optional       # also DOVER + FAST-VQA
cineinfini bootstrap --force                  # re-download all
cineinfini bootstrap --skip-download          # verify only, no fetch
cineinfini bootstrap --json-out report.json   # machine-readable status
```

### 3.5 `cineinfini config`

Inspects/validates the active configuration.

```bash
cineinfini config                             # print active config
cineinfini config --check                     # validate paths + URLs
cineinfini config --modules                   # list registered modules
```

### 3.6 `cineinfini datasets`

Manages validation/calibration datasets.

```bash
cineinfini datasets                           # list (default)
cineinfini datasets --info bvi_hfr            # full details + URL
cineinfini datasets --check                   # which are present locally
cineinfini datasets --list-files bvi_hfr      # ZIP contents WITHOUT download
cineinfini datasets --fetch bvi_hfr                          # full download
cineinfini datasets --fetch bvi_hfr --only "*.mp4"           # partial-ZIP
cineinfini datasets --fetch vbench_eval --only "VBench-master/prompts/*"
```

Partial-ZIP extraction uses HTTP Range requests so we can grab a few
MB of needed files from a multi-GB remote ZIP without downloading the
whole thing.

### 3.7 `cineinfini export-vbench <audit_dir>`

Emit VBench-compatible JSON from an existing audit.

```bash
cineinfini export-vbench ./reports/my_video/ --output vbench.json
```

The schema matches what the official VBench evaluation kit consumes:
`{model, version, video, scores, measured_dimensions, unmeasured_dimensions}`.

### 3.8 `cineinfini score <audit_dir>`

Compute VideoScore-style 5-axis fusion + composite.

```bash
cineinfini score ./reports/my_video/
# === VideoScore-style fusion ===
#   visual_quality                 0.687
#   temporal_consistency           0.838
#   dynamic_degree                 0.440
#   text_to_video_alignment        n/a (no prompt)
#   factual_consistency            0.720
#   composite_score                0.671
```

Also writes `videoscore.json` if `--output` is supplied.

---

## 4. Performance profiles

CineInfini ships with five pre-baked configurations in `cfg/profiles/`.
Each is a thin YAML override of `cfg/config.yaml`; pick the one that
matches your hardware and time budget.

| Profile | Target | Modules ON | Frames/shot | Use case |
|---|---|---|---|---|
| **`realtime.yaml`** | < 2s/min CPU | 3 | 8 | Live monitoring, streaming QC |
| **`ultralight.yaml`** | < 5s total CPU | 3 | 4 | Prompt-eng. loops, CI gates |
| **`postproduction.yaml`** | ~5-10s/min GPU | 9 | 16 | Studio QC, daily review |
| **`academic.yaml`** | No time budget | 21 | 32 | Paper benchmarks, leaderboard subs |
| **`low_memory.yaml`** | < 4 GB VRAM | 8 | 12 | Free-tier Colab, M1, edge |

Use a profile by passing `--config cfg/profiles/<n>.yaml`. You can
also override individual fields inline:

```bash
cineinfini audit video.mp4 --config cfg/profiles/postproduction.yaml \
    -- modules.dover_score.enabled=true
```

For exhaustive performance numbers per profile see
[`benchmarking/PROFILES.md`](benchmarking/PROFILES.md).

---

## 5. Modules

CineInfini's 21 registered modules fall in three groups:

### 5.1 Pure-CV modules (16)

Work without any extra weights. Pure numpy + cv2 + (sometimes) CLIP.

| Module | Default | What it scores |
|---|---|---|
| `motion_coherence` | **ON** | Optical-flow peak divergence + flicker + SSIM3D |
| `identity_consistency` | **ON** | ArcFace embeddings + DTW for face identity drift |
| `semantic_consistency` | **ON** | CLIP-based per-frame similarity within shot |
| `background_consistency` | off | SSIM long-range first-vs-last frame |
| `aesthetic_cinematic` | off | Rule-of-thirds + HSV harmony + contrast |
| `causal_reasoning` | off | Anti-gravity / unphysical motion (flow analysis) |
| `temporal_signature` | off | Cyclic coherence of optical flow |
| `physics_plausibility` | off | Centroid trajectory smoothness + permanence |
| `trustworthiness` | off | SSIM under additive Gaussian noise |
| `world_model_surprise` | off | Per-frame surprise via flow-warp prediction error |
| `creative_composition` | off | Shot-rhythm variety + cut density |
| `long_term_narrative` | off | DINOv2 cosine over multi-shot segments |
| `subject_consistency_long` | off | Identity DTW across non-adjacent shots |
| `benchmark_forensic` | off | SSIM under successive re-encoding |
| `explainability` | off | Approximate Shapley attribution |
| `benchmark_fusion` | off | Aggregate native metrics into VBench-dim naming |

### 5.2 ML-required modules (3, graceful degradation)

These register but report `available: false` if their prerequisites
aren't met. They never fake science.

| Module | Required | Status |
|---|---|---|
| `origin_detection` | `origin_classifier.npz` (must train — recipe in `MISSING_ASSETS.md`) | 🟡 Stub |
| `multi_modal_safety` | CLIP weights + `pip install clip` | 🟡 Stub |
| `prompt_alignment_fine` | CLIP + per-shot prompts in config | 🟡 Stub |

### 5.3 Cross-benchmark wrappers (2, opt-in)

Surface competitor scores in our output.

| Module | Required | Status |
|---|---|---|
| `dover_score` | `DOVER.pth` + `pip install dover-vqa` | 🟡 Stub |
| `fastvqa_score` | `fast-vqa_v0_3.pth` + `pip install fast-vqa` | 🟡 Stub |

### 5.4 Enabling a module

In your YAML config:

```yaml
modules:
  causal_reasoning:
    enabled: true              # turn it on
    gravity_penalty: 8.0       # tweak module-specific params
    vertical_flow_threshold: 0.1
```

Or inline at the CLI:

```bash
cineinfini audit video.mp4 -- modules.causal_reasoning.enabled=true
```

### 5.5 Adding a new module (extending CineInfini)

```python
# my_module.py
from cineinfini.core.registry import register_module
from cineinfini.core.context import VideoContext

@register_module(
    "my_score",
    description="One-line summary of what it does",
    version="1.0",
)
def my_score(ctx: VideoContext) -> dict:
    per_shot = {}
    for sid, frames in ctx.shot_frames.items():
        # your scoring logic
        per_shot[sid] = {"my_score": 0.42}
    return {
        "module": "my_score",
        "version": "1.0",
        "available": True,
        "per_shot": per_shot,
    }
```

Then add it to `cfg.modules` in YAML and import the file once. That's it.

---

## 6. Outputs

Every audit writes the following to `<output_dir>/`:

| File | Contents | Size typical |
|---|---|---|
| `data.json` | All gates, per-module summaries, verdicts | 50-500 KB |
| `dashboard.md` | Human-readable Markdown summary | 5-50 KB |
| `dashboard.html` | Interactive HTML (light/dark theme) | 100-500 KB |
| `report.pdf` | Print-ready PDF (multi-backend) | 200 KB - 2 MB |

Optional exports (run separately):

| Command | Output |
|---|---|
| `cineinfini export-vbench audit_dir/` | `<audit_dir>/audit.vbench.json` |
| `cineinfini score audit_dir/` | stdout + optionally `<audit_dir>/videoscore.json` |
| `cineinfini benchmark dir/` | `benchmark_report.{md,html,csv,json}` |

### 6.1 Reading `data.json`

```json
{
  "video": {"name": "test.mp4", "fps": 24.0, "duration_s": 12.5, "n_shots": 4},
  "gates": {
    "1": {
      "motion_peak_div": 1.8,
      "ssim3d_self": 0.91,
      "ssim_long_range": 0.85,
      "flicker_score": 4.5,
      "identity_within_shot": 0.08,
      "identity_within_shot_dtw": 0.12,
      "verdict": "ACCEPT",
      "failed_gates": []
    },
    "2": { ... }
  },
  "modules": {
    "motion_coherence": {"version": "1.0", ...},
    "identity_consistency": {"version": "1.1", ...},
    ...
  },
  "videoscore_axes": { ... },
  "composite_score": 0.671,
  "timing": {"total": 8.4, "decode": 2.1, ...}
}
```

### 6.2 The phase-4 verdict

Each shot gets one of:

- **`ACCEPT`** — all enabled gates pass their thresholds
- **`REVIEW`** — at least one gate fell into the warning band
- **`REJECT`** — at least one gate exceeded the rejection threshold
- **`BLOCKED`** — safety gate triggered (e.g. NSFW classifier)

The `failed_gates` array lists which gates contributed to the
non-ACCEPT verdict — this is CineInfini's signature feature versus
black-box ranking tools.

---

## 7. Configuration

### 7.1 The `Config` singleton

Everything in CineInfini reads from a single `Config` object accessible
via `get_config()`. The defaults are in `core/config.py`; YAML files
override them.

```python
from cineinfini.core.config import get_config
cfg = get_config()
print(cfg.paths.models_dir)     # → "~/.cineinfini/models"
print(cfg.modules)               # → all 21 module configs
```

### 7.2 YAML structure

The full schema:

```yaml
paths:
  models_dir: "~/.cineinfini/models"
  reports_dir: "~/.cineinfini/reports"
  test_videos_dir: "~/.cineinfini/test_videos"
  datasets_dir: "~/.cineinfini/datasets"
  cache_dir: "~/.cineinfini/cache"

device: "auto"                   # cpu | cuda | auto

processing:
  n_frames_per_shot: 16
  shot_threshold: 0.2
  min_shot_duration_s: 0.5
  max_duration_s: null
  num_workers: 4
  batch_size: 8
  use_amp: true

thresholds:
  motion_peak_div_max: 25.0
  ssim3d_min: 0.78
  videoscore_weights:
    visual_quality: 1.0
    ...

model_urls: { arcface: { url: "...", filename: "..." }, ... }

modules:
  motion_coherence: {enabled: true, threshold: 25.0}
  identity_consistency: {enabled: true, use_dtw: true, ...}
  ... # 21 entries

datasets: { ... }                # 5 entries
optional_models: { ... }         # 4 entries

reporting:
  formats: ["json", "html", "pdf", "markdown"]
  emit_per_shot: true

logging:
  level: "INFO"
```

### 7.3 The `extends:` mechanism

Profiles in `cfg/profiles/*.yaml` inherit from `cfg/config.yaml`:

```yaml
extends: "../config.yaml"

processing:
  n_frames_per_shot: 8           # only override what changes
```

### 7.4 Test config isolation

`cfg/config.test.yaml` mirrors the production schema but routes every
path to `/tmp/cineinfini_test/...` so tests never pollute the user
home directory.

---

## 8. Cross-benchmark integrations

CineInfini's strategic stance: instead of competing with established
AIGC-VQA tools, **subsume their output**. After enabling the wrappers
plus `cineinfini score` and `cineinfini export-vbench`, a single audit
run produces:

- 19 native CineInfini metrics
- 16 VBench dimensions (7 measured, 9 condition-dimensions need prompts)
- 5 VideoScore axes + composite
- DOVER aesthetic + technical
- FAST-VQA score

### 8.1 VBench export

```bash
cineinfini audit video.mp4 --config cfg/profiles/postproduction.yaml
cineinfini export-vbench ~/.cineinfini/reports/video/
# → ~/.cineinfini/reports/video/audit.vbench.json
```

The 7 measured Quality dimensions (subject_consistency,
background_consistency, temporal_flickering, motion_smoothness,
dynamic_degree, aesthetic_quality, imaging_quality) come straight from
our native metrics. The 9 Condition-Consistency dimensions
(object_class, color, scene, ...) require a prompt suite and stay null
in the export — this is documented in the JSON itself.

### 8.2 VideoScore-style fusion

```bash
cineinfini score ~/.cineinfini/reports/video/
```

Produces:

```
visual_quality           0.687
temporal_consistency     0.838
dynamic_degree           0.440
text_to_video_alignment  n/a (no prompt)
factual_consistency      0.720
composite_score          0.671
```

The composite is a weighted sum (default uniform) of measured axes.
Customise via `cfg.thresholds.videoscore_weights`.

### 8.3 DOVER + FAST-VQA wrappers

These are stubs by default — they register and report
`available: false`. To activate:

1. Install the upstream package: `pip install dover-vqa fast-vqa`
2. Fetch the weights: `cineinfini bootstrap --include-optional`
3. Edit the 5 lines in `_run_dover_inference()` /
   `_run_fastvqa_inference()` in the wrapper modules (the documentation
   in those functions shows exactly what to write).
4. Enable in config: `cfg.modules.dover_score.enabled = true`

After this the wrappers' scores appear in `data.json` exactly like our
native metrics.

---

## 9. Datasets and validation

CineInfini ships with metadata for 5 datasets used in calibration and
benchmarking. Most are auto-fetchable; one (BVI-VFI) requires
registration.

```bash
cineinfini datasets                           # list
cineinfini datasets --info bvi_vfi            # registration form URL
cineinfini datasets --fetch bvi_hfr --only "*.mp4"   # partial-ZIP fetch
```

| Key | Type | Auto? | Purpose |
|---|---|---|---|
| `bvi_hfr` | source videos | ✅ | High-frame-rate references for VFI training |
| `bvi_vfi` | DMOS labels | 🔒 (form) | Calibration ground truth |
| `lfw` | faces | ✅ | ArcFace identity validation |
| `videofeedback` | T2V + MOS | ✅ | VideoScore training corpus |
| `vbench_eval` | prompts | ✅ | VBench reference prompts |

### 9.1 Calibration workflow (when datasets are present)

1. Fetch the dataset to `~/.cineinfini/datasets/<NAME>/`
2. Run audits on each video with the same config
3. Compute Spearman/Pearson between native metrics and human MOS
4. Adjust `cfg.thresholds.*` accordingly

A `cineinfini calibrate` command is on the v0.5.0 roadmap to automate
this loop. For now, see `notebooks/04_user_walkthrough.ipynb` for an
example.

---

## 10. Programmatic API

### 10.1 Audit a video from Python

```python
from cineinfini.pipeline import run_audit
from cineinfini.core.config import load_config, set_config

cfg = load_config("cfg/profiles/postproduction.yaml")
set_config(cfg)

audit_data, output_dir = run_audit("my_video.mp4")
print(f"Verdict per shot: {[v['verdict'] for v in audit_data['gates'].values()]}")
print(f"Composite score:  {audit_data.get('composite_score')}")
```

### 10.2 Score an existing audit

```python
from cineinfini.aggregators import attach_videoscore_to_audit
from cineinfini.io.exporters import export_vbench_json

audit_data = json.load(open("audit/data.json"))
attach_videoscore_to_audit(audit_data)            # adds composite + axes
export_vbench_json(audit_data, "audit/vbench.json")
```

### 10.3 Add a custom module

See [section 5.5](#55-adding-a-new-module-extending-cineinfini).

### 10.4 Use the partial-ZIP downloader directly

```python
from cineinfini.core.partial_zip import extract_from_remote_zip

extract_from_remote_zip(
    "https://example.com/big.zip",
    target_dir="./extracted",
    only=["*.json", "labels/*"],
)
```

---

## 11. Troubleshooting

### `ffmpeg not found`

CineInfini doesn't bundle ffmpeg. Install it via your package manager:

```bash
sudo apt install ffmpeg          # Debian/Ubuntu
brew install ffmpeg              # macOS
choco install ffmpeg             # Windows (Chocolatey)
```

Pass `--no-ffmpeg-check` to `bootstrap` if you have an unusual install.

### `Tier-1 weights download fails`

GitHub LFS rate-limit or corporate firewall. The bootstrap report
prints the failed URL — fetch it manually with `wget`/`curl` and drop
it at the printed target path. Re-run `cineinfini bootstrap` to
verify the SHA-256.

### `dover_score reports available: false`

Check the `reason` field in the module output. Common causes:

- `weights missing at .../DOVER.pth` → run `cineinfini bootstrap --include-optional`
- `torch not installed` → `pip install torch torchvision`
- `DOVER model definition not importable` → `pip install dover-vqa`
- `Plug your DOVER inference call here` → edit
  `src/cineinfini/modules/dover_score.py:_run_dover_inference()`

### Audit takes too long

Switch to a lighter profile:

```bash
cineinfini audit video.mp4 --config cfg/profiles/realtime.yaml
```

Or cap the input duration:

```bash
cineinfini audit video.mp4 --duration 30
```

### Memory pressure on small GPU

Use `cfg/profiles/low_memory.yaml`. Reduces batch size, enables AMP,
disables DINOv2-based modules.

---

## 12. Going deeper

- **Architecture**: see [`STATUS.md`](STATUS.md) §10 for the v0.5.0
  refactor proposal (formal DAG, typed pipeline, sub-folders).
- **Comparison vs other tools**: [`benchmarking/COMPARISON.md`](benchmarking/COMPARISON.md).
- **Per-profile performance**: [`benchmarking/PROFILES.md`](benchmarking/PROFILES.md).
- **Competitor install one-liners**: [`benchmarking/COMPETITORS.md`](benchmarking/COMPETITORS.md).
- **Asset inventory** (what to download/train): [`MISSING_ASSETS.md`](MISSING_ASSETS.md).
- **Runnable notebooks**: see [`notebooks/`](../notebooks/).

If you find a missing piece in this manual, open an issue — the manual
is meant to be authoritative.
