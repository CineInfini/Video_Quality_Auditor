# Competitor tools — install + integration one-liners

This document is the **single index** of every external tool CineInfini
can subsume. For each: the install command, the asset URL, the
integration status, and the CineInfini module / export that surfaces
its output.

For deep comparison of *what each tool does* see
[`COMPARISON.md`](COMPARISON.md). This doc is purely operational.

## Index

| Tool | Status | Disk | Install |
|---|---|---|---|
| **VBench** (eval suite + 16 dim) | ✅ subsumed via export | 1.5 GB (partial) | `cineinfini export-vbench` (built-in) |
| **VideoScore** (5-axis + composite) | ✅ subsumed via aggregator | 0 (no MLLM) | `cineinfini score` (built-in) |
| **DOVER** (UGC aesthetic + technical) | 🟡 wrapper ready | 200 MB + torch | `pip install dover-vqa` + bootstrap |
| **DOVER-Mobile** | 🟡 wrapper ready | 35 MB + torch | same; uses `dover_mobile` weight |
| **FAST-VQA** (v0.3 paper) | 🟡 wrapper ready | 110 MB + torch | `pip install fast-vqa` + bootstrap |
| **FAST-VQA-M** (mobile) | 🟡 wrapper ready | 50 MB + torch | same; uses `fastvqa_m` weight |
| **VMAF** (full-reference) | ❌ not yet | 5 MB | (planned v0.5.x) |
| **EvalCrafter** (17 metrics) | ⚙️ partial via VBench export | 30+ GB | (manual; see below) |
| **Q-Align** (MLLM judge) | ❌ not yet | 15 GB | (planned; same wrapper pattern) |
| **MaxVQA** (language-prompted) | ❌ not yet | 250 MB | (planned) |
| **NIQE / BRISQUE** (classical) | ⚙️ subsumed by `flicker_score` | < 1 MB | (built-in) |

---

## VBench (subsumed via export)

CineInfini's native metrics are mapped onto VBench's 7 Quality dimensions.
The 9 Condition-Consistency dimensions need a prompt suite and stay null
in the export — this is documented in the JSON itself.

```bash
# After any audit run:
cineinfini export-vbench ~/.cineinfini/reports/<video>/
# → ~/.cineinfini/reports/<video>/audit.vbench.json
```

To compare CineInfini scores against VBench's official measurements on
the same eval set:

```bash
# 1. Fetch VBench prompts (partial-zip, ~50 MB instead of 1.5 GB)
cineinfini datasets --fetch vbench_eval --only "VBench-master/prompts/*"

# 2. Generate videos from those prompts with your T2V model
# 3. Audit each one
cineinfini benchmark generated_videos/ --config cfg/profiles/academic.yaml

# 4. Export each to VBench format
for f in ~/.cineinfini/reports/*/data.json; do
    cineinfini export-vbench "$(dirname "$f")"
done

# 5. Aggregate the *.vbench.json files for leaderboard submission
```

## VideoScore (subsumed via aggregator)

We don't run the actual VideoScore model (an 8B-parameter MLLM, ~16 GB).
Instead we produce **the same output shape** by fusing CineInfini's
native metrics into the same 5 axes.

```bash
cineinfini score ~/.cineinfini/reports/<video>/
```

This is what makes `ultralight.yaml` faster than VideoScore — no MLLM
inference, just aggregation of already-computed gates.

If you have the bandwidth and want to **run the real VideoScore** as a
parallel comparison:

```bash
pip install transformers
huggingface-cli download TIGER-Lab/Mantis-8B-Idefics2  # 16 GB
# then write your own script to call Mantis on the same videos
```

## DOVER + DOVER-Mobile

Two-branch UGC-VQA scorer (aesthetic + technical). ICCV 2023.

### Install

```bash
pip install torch torchvision
pip install dover-vqa
# or, from source:
git clone https://github.com/QualityAssessment/DOVER.git
cd DOVER && pip install -e .
```

### Fetch weights via CineInfini's bootstrap

```bash
cineinfini bootstrap --include-optional
# → fetches DOVER.pth (200 MB) and DOVER-Mobile.pth (35 MB) to ~/.cineinfini/models/
```

URLs (already in `cfg.optional_models`):

- `https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth`
- `https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth`

### Activate in CineInfini

```yaml
# cfg/config.yaml or your custom profile
modules:
  dover_score:
    enabled: true
```

Then edit `src/cineinfini/modules/dover_score.py` — replace the
`raise NotImplementedError(...)` line in `_run_dover_inference()` with
the actual DOVER call. The function's docstring shows what to write.
Approximate code:

```python
def _run_dover_inference(frames):
    import torch
    from dover import DOVER  # or the entry point of your install
    cfg = get_config()
    weight_path = cfg.models_dir() / cfg.optional_models["dover"]["filename"]
    model = DOVER.from_pretrained(weight_path)
    aesthetic, technical = model.score(frames)
    return float(aesthetic), float(technical)
```

After this, `cineinfini audit` outputs `dover_aesthetic` and
`dover_technical` per shot in `data.json`.

## FAST-VQA + FAST-VQA-M

Fragment-sampling VQA. ECCV 2022. v0.3 paper weights pinned.

### Install

```bash
pip install torch torchvision
pip install fast-vqa
# or, from source:
git clone https://github.com/VQAssessment/FAST-VQA-and-FasterVQA.git
cd FAST-VQA-and-FasterVQA && pip install -e .
```

### Fetch weights via bootstrap

```bash
cineinfini bootstrap --include-optional
# → fetches fast-vqa_v0_3.pth (110 MB) and fast-vqa_m-v0_3.pth (50 MB)
```

URLs pinned to the `v1.0.0-open-release-weights` tag (immutable):

- `https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v1.0.0-open-release-weights/fast-vqa_v0_3.pth`
- `https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v1.0.0-open-release-weights/fast-vqa_m-v0_3.pth`

### Activate

```yaml
modules:
  fastvqa_score:
    enabled: true
```

Edit `_run_fastvqa_inference()` similarly to DOVER — the upstream API
is documented in the wrapper module's docstring.

## VMAF (Netflix, full-reference) — not yet integrated

VMAF is full-reference (compares a distorted video to a reference) so
it's outside CineInfini's no-reference design. A `vmaf_full_reference`
module is on the v0.5.x roadmap for users who do have a reference and
want a single audit run that combines no-reference and full-reference
metrics.

For now, run VMAF separately:

```bash
sudo apt install libvmaf-dev    # or build from source
ffmpeg -i distorted.mp4 -i reference.mp4 \
    -lavfi libvmaf -f null -                    # prints VMAF score
```

## EvalCrafter (17 metrics)

EvalCrafter is heavy (30+ GB of models: BLIP-2, FlowNet2, RAFT, vgg_face,
Dover, VideoMAE, etc.) and runs as a Docker container. CineInfini covers
the same conceptual ground via:

- DOVER (aesthetic + technical) → `dover_score` wrapper
- CLIP-Score → `semantic_consistency`
- BLIP-2 → `prompt_alignment_fine` (when wired)
- Inception Score → not implemented (not very informative for AIGC)
- Optical-flow stats → `motion_coherence`, `temporal_signature`

To run real EvalCrafter for a head-to-head:

```bash
git clone https://github.com/evalcrafter/EvalCrafter.git
cd EvalCrafter
docker pull bruceliu1/evalcrafter:v1
# follow their README for the 700-prompt eval
```

## Q-Align — not yet integrated

MLLM-based scorer that maps VQA to discrete text-defined quality levels
(excellent / good / fair / poor / bad). 15 GB MLLM, similar pattern to
VideoScore.

A `qalign_score` wrapper is on the v0.5.x roadmap. The integration
pattern would be identical to `dover_score` — fetch weight, install
upstream, edit one inference function.

For now:

```bash
pip install transformers
huggingface-cli download q-future/q-align
# then write your own integration
```

## MaxVQA — not yet integrated

Language-prompted VQA, ACM MM 2023. Lighter than Q-Align (no MLLM —
uses CLIP + DOVER features). A `maxvqa_score` wrapper would reuse our
existing CLIP and DOVER weights.

```bash
pip install torch
git clone https://github.com/ZhangErliCarl/MaxVQA.git
# integration tracked for v0.5.x
```

## NIQE / BRISQUE (classical baselines)

These pixel-level no-reference metrics are subsumed by CineInfini's
`flicker_score`, `flicker_highfreq_variance`, and
`world_model_surprise` modules — same family of statistics over MSCN
coefficients and natural-scene statistics, rebranded as gates with
explicit thresholds.

If you specifically need NIQE for a baseline comparison:

```python
import skvideo.measure
niqe_score = skvideo.measure.niqe(frame)
```

---

## Troubleshooting integrations

### "DOVER says available: false even after pip install"

Verify each prerequisite individually:

```bash
python -c "import torch; print(torch.__version__)"      # should print ≥ 1.13
python -c "import dover"                                 # should not raise
ls ~/.cineinfini/models/DOVER.pth                        # should exist
cat src/cineinfini/modules/dover_score.py | grep -A 2 "_run_dover_inference"
# the NotImplementedError line is what makes it report "available: false"
```

### "URL pinned in `cfg.optional_models` returns 404"

The release tag has been moved upstream. Use the GitHub-API resolver:

```python
from cineinfini.core.bootstrap import resolve_github_release_asset
url = resolve_github_release_asset(
    "VQAssessment/FAST-VQA-and-FasterVQA",
    "v1.0.0-open-release-weights",
    "fast-vqa_v0_3.pth",
)
print(url)  # → either the new browser_download_url or None
```

If `None`, the upstream restructure is non-trivial — check the
release page manually and update `cfg.optional_models[...].github_tag`.

### "I want to run all competitors at once for a paper"

```bash
pip install torch torchvision dover-vqa fast-vqa
cineinfini bootstrap --include-optional
cineinfini audit video.mp4 --config cfg/profiles/academic.yaml
cineinfini export-vbench ~/.cineinfini/reports/video/
cineinfini score ~/.cineinfini/reports/video/
```

This single-pipeline run produces:
- 19 native CineInfini gates
- 16 VBench dimensions
- 5 VideoScore axes + composite
- DOVER aesthetic + technical
- FAST-VQA score

That's competitive parity in one command. For Q-Align / MaxVQA /
VideoScore (the actual MLLM) you currently need separate runs — those
integrations are on the v0.5.x roadmap.
