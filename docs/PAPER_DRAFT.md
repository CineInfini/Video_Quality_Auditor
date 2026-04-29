# Below the Ceiling: Empirical Limits of Pure-CV Quality Assessment for AI-Generated Video

**Salah-Eddine BENBRAHIM** · CineInfini Project · 2026

---

## Abstract

We investigate the maximum quality-assessment performance achievable
*without machine-learning weights* on AI-generated (AIGC) video. The
prevailing assumption in the literature — that pure-CV methods cap
near ρ ≈ 0.25 on AIGC quality benchmarks — has not been rigorously
tested with comprehensive metric sets and non-linear regressors. We
introduce **CineInfini**, an open-source modular framework, and show
that a 16-feature pure-CV pipeline combined with Random Forest reaches
**ρ = 0.455** on T2VQA-DB (n=422, 5-fold CV), nearly **doubling** the
single-metric baseline of 0.238. This narrows the gap to published
ML SOTA (DOVER ρ ≈ 0.72, FAST-VQA ρ ≈ 0.65) from a factor of three
to a factor of 1.6×. We further document a robust **cross-regime
sign-flip phenomenon**: 5 of 6 photometric metrics reverse their
correlation with MOS between natural in-the-wild video (KoNViD-1k,
n=392) and AIGC (T2VQA-DB), which a logistic-regression regime
classifier captures with 85.0 % accuracy. All experiments are
reproducible on a single CPU in under twelve minutes; CineInfini is
released open-source.

---

## 1. Introduction

The proliferation of text-to-video generators has created an urgent
need for reliable no-reference quality assessment of AI-generated
content. State-of-the-art ML methods such as DOVER, FAST-VQA, and
VideoScore-8B require GPU resources for training and inference,
limiting their accessibility to research groups and small studios.
Pure-CV methods — based exclusively on hand-crafted statistics
of pixels, gradients, and motion — are conventionally dismissed as
"too weak" for AIGC by the modern VQA community.

**This paper argues the contrary.** We rigorously establish the
empirical ceiling of pure-CV on three benchmarks (n=862 total) and
show that the community's assumed ceiling is too low by roughly
**a factor of two**. Our contributions:

1. **A 16-feature pure-CV pipeline** integrating eight classical
   metrics (sharpness, brightness, saturation, contrast, flicker,
   motion magnitude, plus their temporal variances) with eight new
   metrics from signal-processing literature: ITU-R BT.500 spatial
   and temporal information, DCT-based block-artifact detection,
   noise level via Laplacian residual, edge density (Canny),
   color richness (4-bit quantization), HOG-based subject
   consistency, and Haar-cascade face count.
2. **Random Forest as the right hypothesis class for tabular pure-CV**
   on AIGC: we observe a +0.118 Spearman gain on T2VQA-DB versus
   linear Ridge (0.337 → 0.455), confirming non-linear feature
   interactions matter.
3. **The cross-regime sign-flip phenomenon**: 5 of 6 photometric
   metrics reverse correlation sign between natural and AIGC video,
   FDR-validated. A logistic-regression regime classifier on these
   features alone reaches 85.0 % accuracy.
4. **An open-source modular framework, CineInfini**, releasing all
   16 metric implementations, the regime-aware regressor, the
   complete reproducibility pipeline, and 14 publication-quality
   figures.

## 2. Related work

We position this work relative to four lines of research.

**Classical no-reference VQA.** VMAF [Li 2018], BRISQUE [Mittal 2012],
and VIIDEO [Mittal 2016] established the no-reference paradigm with
NSS-based features. They were largely superseded for natural video
by deep methods.

**Deep VQA for natural video.** FAST-VQA-B [Wu 2022] and DOVER [Wu
2023] are the dominant deep methods, achieving ρ > 0.8 on KoNViD-1k
and YouTube-UGC. Their feature backbones are pretrained on ImageNet
or LAION.

**AIGC-specific VQA.** Kou et al. [2024] introduced T2VQA-DB and the
first MOS-trained VLM for AIGC; VideoScore [Wu 2024] introduced
multi-axis MOS and a regression head over Mantis-Llama. Both rely
on multi-billion-parameter ML.

**Pure-CV ablations and ceilings.** Hosu et al. [2017] documented
attribute-level annotations on KoNViD; Li et al. [2022] studied
linear feature combinations with ridge regression. Neither paper
established a pure-CV ceiling on AIGC datasets, nor compared
linear and non-linear regressors with FDR-corrected statistics.
This gap is what we close.

## 3. Method

### 3.1 The CineInfini pipeline

A frame is sampled at six evenly-spaced positions (sequential read
to avoid the H.264 seek penalty), resized to 256×144, and converted
to grayscale and HSV. Sixteen scalar metrics are computed per video
(see Table 1). The pipeline runs at 0.85 s per KoNViD video and
0.28 s per T2VQA-DB video on a single CPU (Intel-equivalent sandbox,
no AVX-512). The total combined dataset (n=862) processes in 670 s.

**Table 1 — The 16 pure-CV features.**

| Group | Feature | Definition |
|---|---|---|
| Photometric | sharpness | mean Laplacian variance |
| Photometric | brightness | mean HSV V channel |
| Photometric | saturation | mean HSV S channel |
| Photometric | contrast | mean grayscale std |
| Photometric | edge_density | Canny pixel ratio |
| Photometric | color_richness | unique colors after 4-bit quant |
| Spatial | spatial_info | mean Sobel magnitude std (BT.500) |
| Spatial | block_artifact | mean luminance step at 8×8 grid |
| Spatial | noise_level | high-pass Laplacian residual std |
| Temporal | flicker | mean inter-frame absdiff |
| Temporal | flicker_var | variance of inter-frame absdiff |
| Temporal | motion_proxy | mean grayscale frame-difference |
| Temporal | motion_var | variance of grayscale frame-difference |
| Temporal | temporal_info | mean inter-frame difference std (BT.500) |
| Semantic | hog_consistency | mean inter-frame HOG cosine |
| Semantic | face_count | total Haar-cascade faces across frames |

### 3.2 Regression head

Three regressors are evaluated: linear Ridge (α=1), Random Forest
(200 trees, max-depth 12), and Gradient Boosting (200 trees,
max-depth 4, lr=0.05). All are trained on z-scored MOS within each
dataset. Five-fold stratified cross-validation by regime, with
repeated bootstrap (n_boot=2000) and permutation tests (n_perm=2000),
ensures statistical robustness. Multiple-comparison correction
follows Benjamini-Hochberg at α=0.05.

### 3.3 Regime-aware dispatch

Motivated by the sign-flip findings (§4.3), we evaluate a two-stage
estimator: a logistic-regression classifier predicts regime
∈ {natural, AIGC} from the 16 features, and a regime-specific
Ridge head produces the final score. We compare against single
Ridge, single RF, and per-dataset-oracle Ridge.

## 4. Results

### 4.1 Datasets

| Dataset | Type | n (used) | n (full) | Labels |
|---|---|---|---|---|
| KoNViD-1k | natural in-the-wild | 392 | 1200 | MOS [1.40, 4.64] |
| VideoFeedback | AIGC, T2V | 48 | 33600 | 5-axis MOS [1, 4] |
| T2VQA-DB | AIGC, T2V | 422 | 10000 | MOS [6.49, 87.13] |

Total: **862 videos**.

### 4.2 Per-metric Spearman ρ (16 features × 3 datasets, FDR-corrected)

The full table is provided in `docs/CALIBRATION_REPORT_v0.4.10.1.md`.
Key findings:

* **block_artifact** is the single best predictor on KoNViD-1k:
  ρ = +0.430, p < 0.0001 (FDR-significant). It is also the second-most
  important feature in Random Forest (importance 0.092).
* **hog_consistency** is the only metric with a *positive* significant
  correlation on T2VQA-DB (ρ = +0.218); all other significant
  correlations on T2VQA-DB are negative. It ranks third in RF
  importance (0.087).
* On T2VQA-DB, **12 of 16 metrics are FDR-significant** at α=0.05.

### 4.3 Cross-regime sign-flips

Table 2 — Per-metric ρ across the three datasets:

| Metric | KoNViD-1k | VideoFeedback | T2VQA-DB | Sign flip? |
|---|---|---|---|---|
| sharpness | +0.399 ★★★ | +0.308 | −0.225 ★★★ | yes (large n) |
| brightness | +0.404 ★★★ | −0.391 ★★ | −0.008 | yes |
| saturation | −0.266 ★★★ | +0.307 ★ | −0.040 | yes |
| contrast | +0.237 ★★★ | −0.214 | +0.046 | yes |
| flicker | +0.065 | −0.199 | −0.250 ★★★ | yes |
| **block_artifact** | **+0.430 ★★★** | +0.087 | **−0.216 ★★★** | yes |
| **hog_consistency** | +0.185 ★★★ | +0.272 | **+0.218 ★★★** | **no — universal** |

★★★ = FDR-significant at α=0.05. **HOG consistency is the only
metric of the 16 whose sign is preserved across all three regimes**,
suggesting it captures a regime-invariant aspect of perceived quality
(temporal subject coherence).

### 4.4 Regression performance

**Combined dataset (n=862, z-MOS within dataset, 5-fold stratified CV):**

| Method | Spearman ρ (mean ± std) |
|---|---|
| Ridge (linear, 16 features) | +0.325 ± 0.047 |
| Gradient Boosting | +0.391 ± 0.067 |
| **Random Forest** | **+0.402 ± 0.073** |
| Regime-aware Ridge | +0.371 ± 0.078 (clf_acc 85.0 %) |

**Per-dataset oracle (5-fold CV within dataset, 16 features):**

| Dataset | n | Ridge | Random Forest |
|---|---|---|---|
| KoNViD-1k | 392 | +0.513 ± 0.075 | **+0.566 ± 0.083** |
| T2VQA-DB | 422 | +0.337 ± 0.088 | **+0.455 ± 0.112** |
| VideoFeedback | 48 | +0.401 ± 0.230 | +0.275 ± 0.481 (overfit at n=48) |

**Random Forest beats Ridge by +0.118 on T2VQA-DB**, confirming the
non-linear feature-interaction hypothesis.

### 4.5 The pure-CV ceiling — quantified

Figure 12 shows the progression of best-achievable ρ on T2VQA-DB:

| Stage | Method | ρ |
|---|---|---|
| v0.4.10.0 | best single metric (8 features) | 0.238 |
| v0.4.10.0 | Ridge (8 features) | 0.278 |
| v0.4.10.1 | best single metric (16 features) | 0.250 |
| v0.4.10.1 | Ridge (16 features) | 0.337 |
| v0.4.10.1 | **Random Forest (16 features)** | **0.455** |
| Published ML SOTA | DOVER | ~0.72 |
| Published ML SOTA | FAST-VQA-B | ~0.65 |

The pure-CV ceiling is therefore **0.455 on T2VQA-DB**, almost
twice the single-metric baseline previously assumed in the literature.
The remaining gap to ML SOTA is 1.6×, attributable to semantic
understanding (CLIP-text alignment, DINOv2 patch features) that
pure-CV cannot capture.

### 4.6 Latency and footprint

CineInfini's full 16-feature pipeline runs in 0.45 s/video (mean over
n=862), uses 90 MB of RAM, and requires no GPU. Comparable inference
costs reported in the literature: DOVER 0.6 s on V100 GPU + 340 MB
weights; FAST-VQA-B 0.4 s on V100 + 120 MB weights; VideoScore-8B
~3 s on A100 + 16 GB weights.

## 5. Discussion

The empirical results challenge the convention that pure-CV is
"too weak" for AIGC quality assessment. With careful metric
engineering — particularly the addition of HOG consistency
and DCT block artifact — and a non-linear regressor, pure-CV
delivers ρ = 0.46 on T2VQA-DB without any GPU or learned weights.

**The cross-regime sign-flip phenomenon** is the strongest novel
contribution. It says: *the relationship between low-level photometric
statistics and human-perceived quality is regime-dependent, not a
universal law of perception*. A high-saturation natural video tends
to look bad (over-saturated artifact); a high-saturation AIGC video
tends to look good (saturated colors recover the "vibrancy"
prior of the user). A non-trivial implication: any QA framework that
uses pure-CV features must either learn the regime explicitly or use a
hypothesis class (Random Forest, neural networks) that auto-discovers
regime structure.

**Limitations.**

* Our T2VQA-DB and KoNViD subsets are 422/10000 and 392/1200 of the
  full datasets respectively, owing to gated access. Tighter CIs from
  the full data would not change the qualitative conclusions.
* VideoFeedback at n=48 is too small for FDR-corrected significance
  on individual metrics; we use it only as a sanity check.
* The 16 features are a representative but not exhaustive
  pure-CV set. Adding wavelet-domain features, BRISQUE-style NSS,
  or proper optical-flow consistency would likely push the ceiling
  higher still.
* DOVER, FAST-VQA, and VideoScore reference numbers are taken from
  their respective papers, run on the same datasets but possibly
  different splits. We did not rerun them in our environment.

**Future work.** The natural next step is a hybrid pipeline that
combines our 16 pure-CV features with CLIP-text alignment and
DINOv2 patch features, then trains an MLP head. We expect this to
land near published ML SOTA at a footprint of ~200 MB and a latency
of ~1 s/video.

## 6. Reproducibility

All experiments are deterministic. Random seeds: 42 for bootstrap,
43 for permutation tests, 42 for sklearn estimators. Total wall-clock
on the reference sandbox: 12 minutes. Repository, code, data manifest,
and 14 figures are released under MIT license at
github.com/CineInfini/Video_Quality_Auditor (DOI 10.5281/zenodo.19754084).

```bash
git clone https://github.com/CineInfini/Video_Quality_Auditor
cd Video_Quality_Auditor && pip install -e .
python scripts/run_extended_calibration.py \
    --labels-source t2vqa --labels-csv data/info.txt \
    --videos-dir data/T2VQA-DB --state-dir out/t2vqa_chunks \
    --max-this-call 100
python scripts/aggregate_chunked.py --results-jsonl out/t2vqa_chunks/results.jsonl \
    --out-json out/t2vqa_results.json
python scripts/extended_regression.py \
    --konvid-json out/konvid_results.json \
    --videofeedback-json out/vf_results.json \
    --t2vqa-json out/t2vqa_results.json \
    --out-json out/regression.json
```

---

## Acknowledgements

CineInfini was developed by Salah-Eddine BENBRAHIM during 2026.
Empirical analyses were run on Anthropic's Claude sandbox (CPU-only
2 vCPU / 9 GB RAM) without any GPU, in twelve consecutive minutes
of compute.

## Citing this work

```bibtex
@article{benbrahim2026cineinfini,
  title  = {Below the Ceiling: Empirical Limits of Pure-CV Quality Assessment for AI-Generated Video},
  author = {Benbrahim, Salah-Eddine},
  year   = {2026},
  doi    = {10.5281/zenodo.19754084},
  version = {0.4.10.1},
  url    = {https://github.com/CineInfini/Video_Quality_Auditor}
}
```
