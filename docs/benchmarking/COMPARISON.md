# CineInfini vs the Video-Quality-Assessment landscape

A factual, head-to-head comparison of CineInfini against the major
industrial and academic video auditors as of April 2026. Every claim
about another tool is sourced from its public repository or paper;
where I make a value judgment it's flagged as **(opinion)**.

The point of this document is not to argue CineInfini is "best" — it
isn't, on most axes. The point is to **be honest about what fills which
gap**, so you (and reviewers) can pick the right tool for the right job.

---

## TL;DR positioning

CineInfini sits in a gap between three established families:

* **Compression-QC tools** (VMAF, PSNR, SSIM): full-reference, industrial,
  no semantic awareness. Useless for AIGC because there is no reference.
* **UGC-VQA tools** (DOVER, FAST-VQA, MaxVQA): no-reference, deep-learning,
  trained on human-rated YouTube/UGC. Great single-score predictors,
  weak per-shot diagnostic, designed for *real* video.
* **AIGC-VQA benchmarks** (VBench, EvalCrafter, VideoScore): no-reference,
  multi-dimensional, designed for AI-generated video. Heavy (multi-GB),
  monolithic, hard to extend without forking.

CineInfini's positioning: **a modular, hackable, no-reference, multi-dimensional
auditor with explicit shot-level diagnostics, human-readable per-metric
gates, and a registry-based architecture that lets you add a new module
in 100 lines.** It is *not* state-of-the-art on any single benchmark; it
is designed to be the framework you use when you need to *know why* a
video failed, not just *that* it did.

---

## 1. Feature matrix — twelve tools side by side

Legend: ✅ supported · ⚙️ partial · ❌ missing · 🛠 manual

| Capability | VMAF | DOVER | FAST-VQA | Q-Align | VideoScore | VBench-1 | VBench-2 | EvalCrafter | MaxVQA | NIQE | DOVER-Mobile | **CineInfini v0.4.8.1.5** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **No-reference** | ❌ (FR) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Designed for AIGC** | ❌ | ❌ | ❌ | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ❌ | ❌ | ✅ |
| **Per-shot output** | ⚙️ (per-frame) | ❌ | ❌ | ❌ | ❌ | ⚙️ | ⚙️ | ⚙️ | ⚙️ | ⚙️ | ❌ | ✅ (gates per shot) |
| **Number of dimensions** | 1 (+ per-feature) | 2 | 1 | 1 | 5 | 16 | 18 | 17 | 13 | 1 | 2 | 19 modules |
| **Human-aligned (Spearman ≥ 0.7)** | ✅ (compression) | ✅ (UGC, ~0.85) | ✅ (UGC, ~0.85) | ✅ (~0.81) | ✅ (~0.77) | ⚙️ (per dim) | ⚙️ (per dim) | ✅ (~0.65) | ✅ (~0.80) | ❌ (~0.4) | ✅ (UGC, ~0.83) | ❌ **not yet validated** |
| **Open weights** | ✅ | ✅ | ✅ | ✅ | ✅ (large) | ⚙️ (calls many) | ⚙️ | ⚙️ (calls many) | ✅ | ✅ | ✅ | ✅ (4 models bootstrapped) |
| **Causal / physics modules** | ❌ | ❌ | ❌ | ❌ | ⚙️ | ❌ | ✅ | ⚙️ | ❌ | ❌ | ❌ | ✅ (`causal_reasoning`, `physics_plausibility`, `world_model_surprise`) |
| **Identity tracking** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (subject_consistency) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ (ArcFace + DTW intra/inter/long) |
| **Aesthetic scoring** | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ (aesthetic_quality) | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ (`aesthetic_cinematic`: thirds + harmony + contrast) |
| **Plug-in / extensible** | 🛠 (C lib) | 🛠 (fork) | 🛠 (fork) | 🛠 (fork) | 🛠 (fine-tune) | 🛠 (eval suite) | 🛠 | 🛠 | 🛠 | 🛠 | 🛠 | ✅ (`@register_module`) |
| **YAML-driven config** | ❌ | ⚙️ | ⚙️ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚙️ | ❌ | ⚙️ | ✅ (100% via `Config` singleton) |
| **CPU-only deployable** | ✅ | ✅ (Mobile) | ⚙️ | ❌ | ❌ (8B MLLM) | ❌ | ❌ | ❌ | ⚙️ | ✅ | ✅ | ✅ (degraded but functional) |
| **GPU acceleration** | ✅ (CUDA in v3) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ | ⚙️ (via torch CUDA) |
| **HTML/PDF dashboard** | ❌ (CLI) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚙️ (EvalBoard) | ❌ | ❌ | ❌ | ✅ (HTML + PDF + MD + JSON) |
| **Comparison report (A/B)** | ✅ | ❌ | ❌ | ❌ | ⚙️ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ (`cineinfini compare`) |
| **Forensic / re-encoding** | ✅ (its purpose) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (`benchmark_forensic`) |
| **Disk footprint** | ~5 MB | ~200 MB | ~50 MB | ~15 GB (MLLM) | ~16 GB | ~10+ GB | ~15+ GB | ~30+ GB | ~250 MB | <1 MB | ~10 MB | **835 MB** (full Tier 1) |
| **Lines of source code** | ~150k (C+Py) | ~5k | ~3k | ~10k | ~50k (Mantis) | ~20k | ~25k | ~15k | ~6k | ~500 | ~5k (CineInfini core) | ~12k |

---

## 2. Tool-by-tool deep dives

### 2.1 VMAF (Netflix, 2016 → v3.0 with CUDA in 2024)

**What it does:** Full-reference perceptual quality. Combines VIF, ADM,
and motion features via SVM regression. The industrial gold standard
for compression QC.

**Strengths:**
- Battle-tested at petabyte scale by Netflix and FFmpeg.
- CUDA backend (4.4× throughput at 4K).
- VMAF NEG mode resists score gaming via post-processing.

**Weaknesses for our use case:**
- **Full-reference**: needs the original. Useless for AIGC where no
  reference exists.
- Single scalar output (`mean: 62.0967` and friends).
- No semantic, identity, or causal awareness.

**vs CineInfini:** complementary, not competitive. Use VMAF for
"how badly did my CRF=28 transcode hurt the BBB.mp4 source?". Use
CineInfini for "is this Sora output internally consistent across shots?".
CineInfini's `benchmark_forensic` module uses ffmpeg re-encoding and
SSIM (a VMAF building block) to do an analogous thing without a reference.

### 2.2 DOVER (ICCV 2023) and DOVER-Mobile

**What it does:** Disentangled aesthetic + technical scores from frozen
backbones. Two regression heads (aesthetic, technical) trained on
LIVE-VQC, KoNViD-1k, YouTube-UGC.

**Strengths:**
- Excellent Spearman vs MOS on UGC (~0.85 on KoNViD-1k).
- DOVER-Mobile is 5.7× smaller, runs on CPU at 1.4s/video.
- Public weights, ONNX export, well-maintained.

**Weaknesses for AIGC:**
- Trained on real videos; AIGC artifacts (impossible physics, identity
  drift across shots, anti-gravity) are out-of-distribution.
- Single per-video score; doesn't tell you *what* is wrong.

**vs CineInfini:** DOVER answers "is this video aesthetically and
technically fine?" with one number. CineInfini answers "*which* shots
have problems and *why*?" with 19 sub-metrics. **Roadmap:** add a
`dover_score` module that wraps DOVER as one of CineInfini's metrics.
The weights are 200 MB and the API is stable.

### 2.3 FAST-VQA / FasterVQA (ECCV 2022, TPAMI 2023)

**What it does:** Fragment sampling instead of dense frame processing.
Swin-T backbone over fragmentary clips → single quality score.

**Strengths:**
- 210× FLOP reduction vs dense baselines.
- Real-time on Apple M1 CPU (FasterVQA-MT: 14× real-time).
- 99.5% relative accuracy of dense sampling.

**Weaknesses:** Same as DOVER — UGC-trained, single score, no per-shot.

**vs CineInfini:** Could be wrapped as a fast pre-filter. If FasterVQA
gives a video < 0.3 (extremely bad), CineInfini's deeper analysis is
probably wasted compute. **Roadmap:** add an opt-in `fastvqa_prefilter`
module.

### 2.4 Q-Align (2023)

**What it does:** Reframes VQA as text-token prediction over discrete
quality levels ("excellent", "good", "fair", "poor", "bad") via a
multimodal LLM.

**Strengths:** Spearman ~0.81; state-of-the-art on KoNViD/LIVE-VQC at
the time.

**Weaknesses:** The MLLM is heavy (~15 GB). No per-shot. No diagnosis.

**vs CineInfini:** completely different philosophy. Q-Align replaces
hand-crafted features with an LLM judge; CineInfini keeps every metric
inspectable. We are aimed at users who need to *show their work*, not
just rank models.

### 2.5 VideoScore (HuggingFace TIGER, 2024)

**What it does:** Mantis-Idefics2-8B fine-tuned on VideoFeedback
(37.6k T2V samples × 5 dims × 3 raters). Outputs five regression scores
per video: Visual Quality, Temporal Consistency, Dynamic Degree,
Text-to-Video Alignment, Factual Consistency.

**Strengths:**
- Best human alignment to date: Spearman 77.1 on VideoFeedback-test.
- 72.1 average on VBench dimensions.
- Single model handles 5 dimensions.

**Weaknesses:**
- 16 GB checkpoint, requires 8B-parameter MLLM inference.
- Black-box: low Spearman doesn't tell you *which gate* failed.
- Trained on a specific corpus — generalisation to new T2V models
  (Sora 2, Veo 3) requires re-fine-tuning.

**vs CineInfini:** VideoScore is the better single-number predictor of
human preference today. CineInfini gives you 19 numbers + a registry
where you can plug in VideoScore as the 20th. **Roadmap:** add a
`videoscore_module` that calls Mantis when GPU is available, else
returns `available: false`.

### 2.6 VBench (NeurIPS 2024) and VBench-2.0 (March 2025)

**What it does:** 16 dimensions in v1, 18 in v2.0. Each dimension has
its own evaluator: subject_consistency uses DINO, motion_smoothness uses
AMT, aesthetic_quality uses LAION aesthetic predictor, dynamic_degree
uses RAFT, etc. v2.0 adds "intrinsic faithfulness": Human Fidelity,
Controllability, Creativity, Physics, Commonsense.

**Strengths:**
- The de-facto AIGC-VQA standard.
- Each dimension separately validated against human MOS.
- Public leaderboard with ~50 models compared.
- VBench-2.0's Physics dimension is the closest published analogue to
  CineInfini's `causal_reasoning`.

**Weaknesses:**
- Monolithic eval suite — adding a 17th dimension means forking.
- Heavy (calls 10+ models). Hard to deploy lightly.
- Per-prompt scoring; not designed for arbitrary user content.

**vs CineInfini:** VBench is the **benchmark to publish against**;
CineInfini is the **framework to develop against**. CineInfini's
`benchmark_fusion` module already maps 11 of our metrics to VBench
dimension names (`subject_consistency`, `temporal_flickering`,
`motion_smoothness`, `dynamic_degree`, `aesthetic_quality`,
`background_consistency`, `imaging_quality`) for cross-comparison.

A faithful integration test would be:
1. Run VBench's 946-prompt eval set through any T2V model.
2. Score each output with VBench's official toolkit *and* with
   CineInfini's `benchmark_fusion`.
3. Compare per-dimension Spearman. **(Pending — needs the eval set.)**

### 2.7 EvalCrafter (CVPR 2024)

**What it does:** 17 metrics over 700 prompts, 4 categories. Combines
DOVER (aesthetic + technical), CLIP-Score, BLIP-2 captioning, VGG-Face,
RAFT, FlowNet2, VideoMAE.

**Strengths:**
- Extensive prompt coverage (filtered from 600k community submissions).
- Human-aligned via linear-regression fit to user opinions.

**Weaknesses:**
- **30+ GB of pretrained checkpoints required**: BLIP-2-OPT-2.7B,
  RAFT, FlowNet2, vgg_face_weights, Dover, VideoMAE, ViT-B/32, etc.
- Docker-only deployment is the path of least resistance.
- No per-shot decomposition.

**vs CineInfini:** EvalCrafter is the closest in spirit — an "everything
toolkit" — but heavier and less hackable. CineInfini cuts the asset
weight by 35× (835 MB vs 30+ GB) by leaving the heaviest models
(BLIP-2, FlowNet2, VideoMAE) **out of the default install** and only
loading them when the corresponding optional module is enabled.

### 2.8 MaxVQA (ACM MM 2023)

**What it does:** Language-prompted VQA — user supplies a prompt
"Score this video on sharpness", model returns a Likert score per
language dimension.

**Strengths:** Most flexible single-tool VQA.
**Weaknesses:** Not designed for AIGC; needs DOVER+FAST-VQA pre-features.

**vs CineInfini:** MaxVQA's prompt-driven approach inspires our
`prompt_alignment_fine` module, which uses CLIP zero-shot scoring as a
lightweight stand-in until a proper VLM is wired up.

### 2.9 NIQE / BRISQUE / classical no-reference image quality

**What they do:** Statistics over MSCN coefficients (NIQE) or natural
scene statistics (BRISQUE). Pure CV, no learning.

**Strengths:** Tiny, fast, deterministic.
**Weaknesses:** Spearman ~0.4–0.5 on modern benchmarks. Outclassed by
FAST-VQA on every UGC dataset.

**vs CineInfini:** NIQE-like statistics are buried inside CineInfini's
`flicker_score`, `flicker_highfreq_variance`, `world_model_surprise`.
Where NIQE gives you one number, CineInfini gives you the full
breakdown.

### 2.10 The smaller brothers (CRAVE, AIGV-Assessor, T2VQA, FETV, …)

These are mostly **datasets** with companion baseline models. Useful
for validation, not for deployment. CineInfini will adopt their
splits as they reach Spearman > 0.85 against human MOS.

---

## 3. Where CineInfini wins, and where it doesn't

### Wins (objective)

1. **Modularity.** Add a 20th module in ~100 lines via `@register_module`.
   Every other tool requires forking.
2. **Per-shot diagnostic.** Phase-4 aggregator emits one
   ACCEPT/REVIEW/REJECT verdict *per shot* with the list of failing
   gates. None of the others do this.
3. **No-reference + multi-dimensional + lightweight.** Tier-1 install
   is 835 MB total — vs EvalCrafter's 30+ GB or VideoScore's 16 GB.
4. **Identity-aware causal reasoning combined.** No other open tool
   ships ArcFace identity-DTW *and* gravity violation detection in
   one pipeline.
5. **Configuration discipline.** 100% YAML-driven; tests use
   `test_config()` for /tmp isolation. None of the comparison tools
   has this clean separation.

### Doesn't win

1. **Human alignment.** The closest CineInfini has come is the v0.4.5
   BVI-VFI calibration. We have **no published Spearman number** vs
   human MOS on a modern T2V dataset. VideoScore (77.1), DOVER (~85
   on UGC) and Q-Align (81) are all empirically validated; CineInfini
   isn't, yet.
2. **Single-number ranking accuracy.** If you need *one* score that
   correlates with human preference for ranking T2V models, use
   VideoScore.
3. **Industrial penetration.** VMAF is in FFmpeg, every CDN, every
   transcoder. CineInfini is a research framework.
4. **Pretrained breadth.** EvalCrafter ships BLIP-2 + FlowNet2 +
   VideoMAE + DOVER out of the box. We ship 4 models and
   `available: false` for the rest.

---

## 4. Honest validation roadmap

To stop being unvalidated, the minimum is:

1. **Calibrate on BVI-VFI** (already wired up via `cineinfini calibrate`,
   needs the dataset uploaded).
2. **Publish Spearman/Pearson on VideoFeedback-test, VBench, EvalCrafter**.
   This requires running our 19 modules on those test sets. Once we have
   numbers, drop them in a new `docs/BENCHMARKS.md`.
3. **Train `origin_classifier.npz`** so `origin_detection` joins the
   active default modules instead of reporting `available: false`.
4. **Wire DOVER / FAST-VQA / VideoScore as opt-in modules** so users
   can compose CineInfini's diagnostics with their predictions.

After steps 1–3 the tool becomes citable. Step 4 makes it the
"Swiss-army knife" the README claims.

---

## 5. Summary: which tool to pick

| If you need to… | Use |
|---|---|
| QC a transcode against an original | **VMAF** |
| Rank T2V models by overall human preference | **VideoScore** |
| Submit to a workshop with a standard 16-dim eval | **VBench** |
| Score UGC quickly with one number | **DOVER-Mobile** or **FasterVQA-MT** |
| Diagnose *why* a generated shot looks wrong | **CineInfini** |
| Run on a laptop without a GPU | **DOVER-Mobile**, **NIQE**, or **CineInfini** |
| Compare two videos side-by-side with HTML report | **CineInfini compare** |
| Detect physically impossible motion | **CineInfini** (`causal_reasoning`) or **VBench-2.0** (Physics) |
| Audit identity drift across a 10-min film | **CineInfini** (`subject_consistency_long`) |
| Add a custom metric in 30 minutes | **CineInfini** (`@register_module`) |

In one line: **CineInfini is the only no-reference, modular, per-shot,
multi-dimensional auditor that runs in under 1 GB of weights and
exposes every gate to the user.** That's the gap it fills. Whether
that gap is worth filling for your project is the question this
document is meant to help you answer honestly.
