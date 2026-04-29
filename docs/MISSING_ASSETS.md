# Missing assets inventory — CineInfini v0.4.8.1.5

This document is the **single source of truth** for every external file
CineInfini may need. Files are grouped by purpose. For each file we
give: the URL (when public), the path it must land at, the SHA-256
when known, the audited file size, the licence, and which CineInfini
module(s) actually use it.

`cineinfini bootstrap` already automates Tier 1 + Tier 2. Tier 3 is
manual: those weights either don't exist publicly (must be trained) or
require a click-through accept on the publisher's site.

---

## Tier 1 — ML model weights (downloaded by `cineinfini bootstrap`)

These are declared in `cfg.model_urls` and fetched automatically. Land
in `paths.models_dir` (default `~/.cineinfini/models/`).

| Key | Filename | Size | License | Used by | Status |
|---|---|---|---|---|---|
| `arcface` | `arcface.onnx` | ~166 MB | MIT (yakhyo facial-analysis fork) | `identity_consistency`, `subject_consistency_long` | ✅ URL configured |
| `yunet` | `yunet.onnx` | ~232 KB | Apache-2.0 (OpenCV Zoo) | face detection in `identity_consistency` | ✅ URL configured |
| `clip_vit_b32` | `ViT-B-32.pt` | ~338 MB | MIT (OpenAI) | `semantic_consistency`, `multi_modal_safety`, `prompt_alignment_fine` | ✅ URL + SHA-256 configured |
| `dinov2_vitb14` | `dinov2_vitb14.pth` | ~330 MB | Apache-2.0 (Meta) | `long_term_narrative`, `origin_detection` | ✅ URL configured |

Total: **~835 MB** for full Tier 1.

Verify with `cineinfini bootstrap --models-only`. If a download fails
(typical: corporate firewall, GitHub LFS rate limit), the report shows
the URL and target path so you can drop the file there manually.

---

## Tier 2 — Test videos (downloaded by `cineinfini bootstrap --videos-only`)

Royalty-free Blender Foundation films, used for quick smoke-tests.
Declared in `cfg.test_videos`, land in `paths.test_videos_dir`.

| Key | Filename | Size | License | Source |
|---|---|---|---|---|
| `BBB` | `BBB.mp4` | ~62 MB (320×180) | CC-BY 3.0 | Big Buck Bunny |
| `tears_of_steel` | `tears_of_steel.mov` | ~734 MB (1080p) | CC-BY 3.0 | Tears of Steel |
| `elephants_dream` | `elephants_dream.avi` | ~830 MB (1024×576) | CC-BY 2.5 | Elephants Dream |
| `sintel_trailer` | `sintel_trailer.mp4` | ~62 MB (1080p) | CC-BY 3.0 | Sintel trailer |

Total: **~1.7 GB** for the full test suite. For quick smoke tests, only
`BBB.mp4` (~62 MB) is needed — pass `--only BBB` to bootstrap.

---

## Tier 3 — Optional assets (manual or to-be-trained)

These are the **gaps** between what we ship and an end-to-end production
deployment. Fill them in as needed.

### How datasets are wired

As of v0.4.8.1.6, every validation dataset is a first-class entry in
`cfg.datasets` with `homepage`, `registration_url`, `paper`, `license`,
`expected_layout`, and a target directory under `paths.datasets_dir`.

```bash
cineinfini datasets --list                   # show all registered datasets
cineinfini datasets --info bvi_vfi           # full details + download URL
cineinfini datasets --check                  # which ones are present locally
```

When you have downloaded a dataset, drop it at the path printed by
`datasets --info` (e.g. `~/.cineinfini/datasets/BVI-VFI/`) and the
calibration / benchmark scripts will pick it up automatically.

### 3a. `origin_classifier.npz` (real vs AI-generated)

**Used by:** `origin_detection`
**Format:** `np.savez(file, coef=<1×N float32>, intercept=<float>)` where
N is the DINOv2 ViT-B/14 embedding dim (768).
**Status:** ❌ **No public version exists — must be trained.**

This is a linear logistic-regression head on top of frozen DINOv2 features.
Training recipe (one-shot, ~1 hour on a single A100):

1. Collect ~5000 frames of real footage (Kinetics-400, AVA, anything natural)
   labelled 0, and ~5000 frames of AI-generated video (Sora, Runway Gen-3,
   Veo, Kling, Pika, ModelScope, Stable Video Diffusion, Mochi, etc.) labelled 1.
2. Extract DINOv2 ViT-B/14 features per frame (use
   `cineinfini.core.embedding.load_dinov2()`).
3. Fit `sklearn.linear_model.LogisticRegression(max_iter=1000)`.
4. `np.savez(models_dir/'origin_classifier.npz', coef=clf.coef_, intercept=clf.intercept_)`.

A reasonable training corpus you could assemble:
- **Real:** 1000 clips × 5 frames each from Kinetics-400 (Apache-2.0).
- **AI:** GenAI-Bench (the public split has 800+ AI videos with model
  labels, MIT-like licence).

If you don't want to train one yourself: drop the `models_dir/` path
empty and `origin_detection` will report `available: false`. The other
17 modules continue to work.

### 3b. BLIP-2 / VLM weights (optional alignment)

**Used by:** advanced `prompt_alignment_fine` (when installed instead
of CLIP fallback).
**Status:** Public via HuggingFace, large.

| Asset | URL | Size |
|---|---|---|
| BLIP-2 OPT-2.7B | `Salesforce/blip2-opt-2.7b` | ~14 GB |
| BLIP-2 FlanT5-XL | `Salesforce/blip2-flan-t5-xl` | ~16 GB |

CineInfini does **not** require BLIP-2 today: the `prompt_alignment_fine`
module uses CLIP-only zero-shot scoring. BLIP-2 is on the roadmap as a
high-precision option behind `cfg.modules.prompt_alignment_fine.model = "blip2"`.

### 3c. NSFW / violence classifier (optional safety hardening)

**Used by:** `multi_modal_safety` (currently zero-shot CLIP — workable but
not moderation-grade).
**Status:** Public public weights exist; we don't ship them by default
because of policy risk.

| Asset | URL | Size | Notes |
|---|---|---|---|
| NSFW image classifier (Falconsai) | `Falconsai/nsfw_image_detection` | ~340 MB | Vision Transformer; high precision on stills, can be applied per-frame |
| Open-NSFW2 (Yahoo, ResNet-50) | `notAI-tech/NudeNet` | ~95 MB | Older but widely benchmarked |

Both have permissive licences but **independent legal review is recommended
before deploying any safety classifier**. Zero-shot CLIP via the existing
`multi_modal_safety` module is the safer choice for ad-hoc audits.

### 3d. Validation datasets (for calibration / Spearman / Pearson)

These aren't run-time assets — they're how you would *validate* CineInfini
against human MOS scores.

| Dataset | Purpose | Size | Access |
|---|---|---|---|
| **BVI-VFI** | Frame-interpolation MOS (108 ref + 540 distorted, DMOS from 189 subjects) | ~10 GB | **Registration form (gated):** https://forms.office.com/e/gtKpYriSMJ — homepage at https://github.com/danier97/BVI-VFI-database |
| **VideoFeedback (VideoScore)** | 37.6k T2V videos × 5 dimensions × 3 raters | ~50 GB | HuggingFace `TIGER-Lab/VideoFeedback` |
| **VBench eval set** | 16 dimensions × prompt suite | ~30 GB | GitHub `Vchitect/VBench` |
| **EvalCrafter prompts + videos** | 2541 videos, 4 aspects × 7 raters | ~5 GB | GitHub `evalcrafter/EvalCrafter` |
| **GenAI-Bench** | Human preference pairs on T2V | ~8 GB | HuggingFace `TIGER-Lab/GenAI-Bench` |
| **T2VQA-DB** | 10k AIGC videos with MOS | ~6 GB | GitHub `QMME/T2VQA` |
| **FETV** | Fine-grained T2V eval set | ~3 GB | HuggingFace `liuhaotian/FETV` |
| **LFW** | Face recognition benchmark (ArcFace validation) | ~170 MB | http://vis-www.cs.umass.edu/lfw |

### 3e. External eval toolkits (for cross-benchmarking, not bundled)

These are the systems CineInfini's `benchmark_fusion` module exports to.
You don't *need* them to run CineInfini, but you do need them to **compare
your scores to industry rankings**.

| Tool | Purpose | URL |
|---|---|---|
| VMAF (Netflix) | Industrial full-reference quality (compression QC) | https://github.com/Netflix/vmaf |
| DOVER | Aesthetic + Technical UGC scoring | https://github.com/VQAssessment/DOVER |
| FAST-VQA / FasterVQA | Fragment-sampling VQA (ECCV 2022) | https://github.com/VQAssessment/FAST-VQA-and-FasterVQA |
| VBench | 16-dimension AIGC eval | https://github.com/Vchitect/VBench |
| EvalCrafter | 17-metric T2V eval | https://github.com/evalcrafter/EvalCrafter |
| VideoScore | MLLM-based scorer | https://huggingface.co/TIGER-Lab/VideoScore |

---

## Quick reference: what to upload

If you want to help CineInfini reach 100% feature coverage with the
**minimum** human effort, here is the priority order:

1. ✅ **Already automated** — Tier 1 + Tier 2 via `cineinfini bootstrap`.
2. **High-leverage, easy:** `BVI-VFI` calibration dataset (~10 GB) — unlocks
   threshold calibration via `cineinfini calibrate` against human MOS.
3. **High-leverage, harder:** train + drop `origin_classifier.npz` (300 KB)
   in `~/.cineinfini/models/` — unlocks the `origin_detection` module.
4. **Optional, large:** VBench / EvalCrafter / VideoScore eval sets — needed
   only if you want CineInfini benchmarked head-to-head against them in a
   paper.

Anything you upload to this chat I can pick up from `/mnt/user-data/uploads/`
and integrate.
