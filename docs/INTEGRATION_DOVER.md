# Integration guide — DOVER and FAST-VQA wrappers

This is the concrete how-to for wiring CineInfini's **`dover_score`**
and **`fastvqa_score`** wrappers to actually produce scores. Out of the
box both report `available: false` because we deliberately do not
vendor third-party model code. After this guide they return real
scores in the same `gates` and `modules` structure as native metrics.

> **Time**: ~10 minutes per tool, plus weight download.
> **Skill**: ability to follow `pip install` and edit one Python function.

## What you'll have when this is done

After both integrations, a single `cineinfini audit` produces in `data.json`:

```json
{
  "modules": {
    "dover_score": {
      "available": true, "version": "dover@v0.1.0",
      "mean_aesthetic": 0.687, "mean_technical": 0.812,
      "mean_dover_fused": 0.749,
      "per_shot": {"1": {"dover_aesthetic": 0.71, "dover_technical": 0.83}}
    },
    "fastvqa_score": {
      "available": true, "version": "fastvqa@v0.3",
      "mean_fastvqa": 0.654,
      "per_shot": {"1": {"fastvqa_score": 0.65}}
    }
  }
}
```

These flow through `cineinfini score` (VideoScore-style fusion) and
`cineinfini export-vbench` (16-dim leaderboard JSON) automatically.

---

## Part 1 — DOVER (UGC aesthetic + technical, ICCV 2023)

### Step 1. Install

```bash
pip install torch torchvision
pip install dover-vqa
# OR install from source if pip release is stale:
git clone https://github.com/QualityAssessment/DOVER.git
cd DOVER && pip install -e . && cd ..
```

Verify: `python -c "import dover; print('OK')"`

### Step 2. Fetch DOVER weights

```bash
cineinfini bootstrap --include-optional
```

Downloads `DOVER.pth` (200 MB) and `DOVER-Mobile.pth` (35 MB) to
`~/.cineinfini/models/`. Idempotent. SHA-256 verified.

If GitHub is unreachable (corporate firewall), fetch manually:

```bash
curl -L -o ~/.cineinfini/models/DOVER.pth \
  https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
```

### Step 3. Edit `_run_dover_inference()`

Open `src/cineinfini/modules/dover_score.py`. Find:

```python
def _run_dover_inference(frames):
    raise NotImplementedError("Plug your DOVER inference call here; ...")
```

Replace with one of:

**Option A — official DOVER package (most common)**

```python
def _run_dover_inference(frames):
    import torch
    from dover.models import DOVER
    from dover.datasets import UnifiedFrameSampler
    cfg = get_config()
    weight_path = cfg.models_dir() / cfg.optional_models["dover"]["filename"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hasattr(_run_dover_inference, "_model"):
        m = DOVER().to(device).eval()
        m.load_state_dict(torch.load(weight_path, map_location=device))
        _run_dover_inference._model = m

    sampler = UnifiedFrameSampler(fragments_h=7, fragments_w=7,
                                   fsize_h=32, fsize_w=32, num_clips=4)
    video_tensor = sampler(frames).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = _run_dover_inference._model(video_tensor)
    aesthetic = float(torch.sigmoid(scores[0]))
    technical = float(torch.sigmoid(scores[1]))
    return aesthetic, technical
```

**Option B — pip `dover-vqa` package (different API)**

```python
def _run_dover_inference(frames):
    from dover_vqa import DOVERScorer
    cfg = get_config()
    weight_path = cfg.models_dir() / cfg.optional_models["dover"]["filename"]

    if not hasattr(_run_dover_inference, "_scorer"):
        _run_dover_inference._scorer = DOVERScorer(weights=weight_path)
    return _run_dover_inference._scorer.score(frames)
```

The model is cached on the function attribute so the cold-start cost
(weight load) only happens once per process.

### Step 4. Enable in config

```yaml
# cfg/config.yaml or your profile
modules:
  dover_score:
    enabled: true
```

Or use the academic profile which has it pre-enabled:

```bash
cineinfini audit my_video.mp4 --config cfg/profiles/academic.yaml
```

### Step 5. Verify

```bash
cineinfini audit my_video.mp4 --config cfg/profiles/academic.yaml
python -c "
import json
d = json.load(open('~/.cineinfini/reports/my_video/data.json'.replace('~','/home/'+__import__('os').getlogin())))
print(d['modules'].get('dover_score'))
"
```

Expected: `{'available': True, 'mean_aesthetic': 0.687, ...}`.

If `available: False`, the `reason` field tells you exactly which
prerequisite failed.

---

## Part 2 — FAST-VQA (fragment-sampling VQA, ECCV 2022)

### Step 1. Install

```bash
pip install fast-vqa
# OR from source:
git clone https://github.com/VQAssessment/FAST-VQA-and-FasterVQA.git
cd FAST-VQA-and-FasterVQA && pip install -e . && cd ..
```

### Step 2. Weights are already pinned (paper v0.3)

```bash
cineinfini bootstrap --include-optional
```

Fetches `fast-vqa_v0_3.pth` (110 MB) and `fast-vqa_m-v0_3.pth` (50 MB).

> We pin the v0.3 paper weights via the `v1.0.0-open-release-weights`
> GitHub tag, which is immutable. The upstream main branch may have
> moved on; pinning preserves reproducibility.

### Step 3. Edit `_run_fastvqa_inference()`

Open `src/cineinfini/modules/fastvqa_score.py`. Replace the
`NotImplementedError` body with:

```python
def _run_fastvqa_inference(frames):
    import torch
    import numpy as np
    from fastvqa.models import DiViDeAddEvaluator
    cfg = get_config()
    weight_path = cfg.models_dir() / cfg.optional_models["fastvqa"]["filename"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hasattr(_run_fastvqa_inference, "_model"):
        m = DiViDeAddEvaluator().to(device).eval()
        sd = torch.load(weight_path, map_location=device)
        m.load_state_dict(sd.get("state_dict", sd), strict=False)
        _run_fastvqa_inference._model = m

    # FAST-VQA expects (1, 3, T, H, W) tensor with values in [0, 1]
    video = torch.from_numpy(np.stack(frames)).float() / 255.0
    video = video.permute(3, 0, 1, 2).unsqueeze(0).to(device)
    with torch.inference_mode():
        score = _run_fastvqa_inference._model(video)
    return float(torch.sigmoid(score.squeeze()))
```

### Step 4. Enable + verify

Same as DOVER — `cfg.modules.fastvqa_score.enabled = true`, then
`cineinfini audit ...` and check the JSON.

---

## Part 3 — Common pitfalls

### `torch.load` fails with `_pickle.UnpicklingError`

Some checkpoints predate `weights_only` security:

```python
sd = torch.load(weight_path, map_location=device, weights_only=False)
```

Safe because the pinned URLs are official signed releases.

### GPU OOM on 4 GB cards

Force CPU in the inference function: `device = "cpu"`. Use
`cfg/profiles/low_memory.yaml` for the rest of the pipeline.

### FAST-VQA dimensions don't match

The fragment sampler expects multiple-of-7 dimensions. If your video
is too small (< 224 px), the official repo's
`fastvqa.datasets.FastVQAFragmentSampler` handles padding/upscaling.

### Inference is slow on the first call

Expected. The wrapper caches the model on the function attribute
(`_run_dover_inference._model`); subsequent calls reuse it. First
call pays the weight-load cost (~1-3s for DOVER, ~0.5s for FAST-VQA).

---

## Part 4 — End-to-end verification

After both wrappers are wired:

```bash
cineinfini audit my_video.mp4 --config cfg/profiles/academic.yaml

# Inspect outputs
ls ~/.cineinfini/reports/my_video/
# Expected: dashboard.html, dashboard.md, data.json, audit.vbench.json

# Confirm both wrappers ran
python -m json.tool < ~/.cineinfini/reports/my_video/data.json | \
    grep -A1 '"available"' | head -20

# VideoScore composite reflects the new signals
cineinfini score ~/.cineinfini/reports/my_video/

# VBench export includes the new dimensions
cat ~/.cineinfini/reports/my_video/audit.vbench.json
```

If `dover_score.available == true` and `fastvqa_score.available == true`,
the integration is complete. The dashboard's Module Status table now
shows them with green badges, and they contribute to the Composite KPI.

---

## Part 5 — Submitting to leaderboards

| Leaderboard | Submission format | How |
|---|---|---|
| [VBench](https://github.com/Vchitect/VBench) | 16-dim JSON | `cineinfini export-vbench` produces exact schema |
| [VideoScore-Bench](https://huggingface.co/TIGER-Lab/VideoFeedback) | 5-axis CSV | `cineinfini score --output videoscore.json`, convert |
| Q-Align Bench | TBD | needs Q-Align wrapper (v0.5.x roadmap) |

The whole point of v0.4.8.x's competitive-parity work was making this
pipeline frictionless: two `pip install` + 30 lines of edits and
your CineInfini audit is leaderboard-ready.
