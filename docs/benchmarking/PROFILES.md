# Performance profiles — characteristics and trade-offs

CineInfini ships five pre-baked configurations. This document
characterises each one in detail: what's enabled, what's disabled,
expected wall-clock cost, and which competitor it's positioned
against.

> **Note on numbers**: the wall-clock times below are *targets*, not
> measurements on your hardware. Run the
> [`notebooks/02_integration_tests.ipynb`](../../notebooks/02_integration_tests.ipynb)
> notebook on your machine for actual numbers; CineInfini doesn't ship
> with a reference benchmark suite (yet).

## Comparison matrix

| Profile | Modules ON | Frames/shot | Use case | Competitive parity |
|---|---|---|---|---|
| `realtime.yaml` | 3 | 8 | Live monitoring | FasterVQA-MT (14× real-time on M1) |
| `ultralight.yaml` | 3 | 4 | CI gates, prompt loops | VideoScore (but no MLLM load) |
| `postproduction.yaml` | 9 | 16 | Studio QC | DOVER + per-shot diagnostic |
| `academic.yaml` | 21 | 32 | Paper benchmarks | EvalCrafter coverage + VBench dims |
| `low_memory.yaml` | 8 | 12 | Free-tier Colab, M1 | DOVER-Mobile footprint |

## Detailed targets per profile

### `realtime.yaml` — < 2s of audit per minute of video on CPU

**Goal**: live monitoring of a generation service or streaming QC
pipeline. Trade depth for speed — one number per shot, no rendering,
no heavy modules.

| Aspect | Value |
|---|---|
| Modules enabled | 3 (`motion_coherence`, `identity_consistency`, `semantic_consistency`) |
| Frames per shot | 8 |
| Mixed precision | yes |
| DTW for identity | **disabled** (slow) |
| Reports | JSON only |
| Logging | WARNING |

**Anti-goals**: don't use this profile to write a paper or to surface
subtle artefacts. It's tuned for "is this video obviously broken?"
gating decisions.

**Compares to**: FasterVQA-MT in pure-CV form. CineInfini wins on
shot-level decomposition (FasterVQA gives one number per video).

### `ultralight.yaml` — < 5 seconds total CPU

**Goal**: CI gate or prompt-engineering loop. Beat VideoScore on speed.

| Aspect | Value |
|---|---|
| Modules enabled | 3 (same as realtime) |
| Frames per shot | 4 |
| Max input duration | 30 s |
| Mixed precision | yes |
| Logging | ERROR |

**How it beats VideoScore**: VideoScore loads an 8B-parameter MLLM
(~16 GB) and runs it; total cold-start + inference is typically
~30s for a 5s video. CineInfini's ultralight profile uses only
vectorised numpy/cv2 + cached CLIP, no large model loaded.

**Compares to**: VideoScore on speed; DOVER-Mobile on footprint.
Doesn't compete on accuracy — that's not the use case.

### `postproduction.yaml` — ~5-10 s/min on GPU

**Goal**: the default for serious users. Studio QC of generated shots,
daily review pipelines.

| Aspect | Value |
|---|---|
| Modules enabled | 9 |
| Frames per shot | 16 (default) |
| DTW for identity | **enabled** |
| Output formats | json + html |
| Logging | INFO |

Enabled modules: `motion_coherence`, `identity_consistency`,
`semantic_consistency`, `background_consistency`, `aesthetic_cinematic`,
`causal_reasoning`, `subject_consistency_long`, `long_term_narrative`,
`benchmark_fusion`.

**Compares to**: DOVER (with our per-shot decomposition added).
Includes our differentiating modules (`causal_reasoning`,
`subject_consistency_long`) that no UGC-VQA tool has.

### `academic.yaml` — no time budget

**Goal**: maximum precision for paper benchmarks and leaderboard
submissions. Activates everything.

| Aspect | Value |
|---|---|
| Modules enabled | 21 (all, including DOVER + FAST-VQA wrappers) |
| Frames per shot | 32 |
| Shot threshold | 0.18 (catch micro-cuts) |
| Mixed precision | **disabled** (FP32 for reproducibility) |
| Output formats | json + html + pdf + markdown |
| Per-shot detail | yes |
| VBench export | yes |
| Logging | INFO |

**Compares to**: EvalCrafter coverage (17 metrics → CineInfini
ships 21) + VBench 16 dimensions in one run.

### `low_memory.yaml` — < 4 GB VRAM

**Goal**: fit on consumer hardware (free Colab, T4, M1/M2 unified
memory).

| Aspect | Value |
|---|---|
| Modules enabled | 8 |
| Frames per shot | 12 |
| Batch size | 2 |
| Mixed precision | yes (FP16 halves VRAM) |
| Workers | 2 |
| DINOv2-based modules | **disabled** (330 MB model + activations) |

Enabled modules: `motion_coherence`, `identity_consistency` (no DTW),
`semantic_consistency`, `background_consistency`, `aesthetic_cinematic`,
`causal_reasoning`, `benchmark_fusion`.

**Compares to**: DOVER-Mobile (35 MB model). Same memory footprint
class while keeping CineInfini's per-shot diagnostic.

---

## Choosing a profile

Use this decision tree:

```
Do you need the audit to finish in < 5 seconds?
├── Yes → ultralight.yaml
└── No
    Do you need real-time / streaming?
    ├── Yes → realtime.yaml
    └── No
        Are you submitting to a research benchmark?
        ├── Yes → academic.yaml
        └── No
            Are you on consumer GPU / Colab / M1?
            ├── Yes → low_memory.yaml
            └── No → postproduction.yaml (default)
```

## Customising a profile

A profile is a thin YAML override of `cfg/config.yaml`. The cleanest
way to adjust is to copy the closest profile and edit only what
changes:

```bash
cp cfg/profiles/postproduction.yaml cfg/profiles/my_custom.yaml
# edit my_custom.yaml
cineinfini audit video.mp4 --config cfg/profiles/my_custom.yaml
```

Or override a single field at the CLI:

```bash
cineinfini audit video.mp4 \
    --config cfg/profiles/postproduction.yaml \
    -- modules.dover_score.enabled=true \
    -- processing.n_frames_per_shot=24
```

## What's NOT a profile

Three configurations are deliberately not pre-baked because they need
domain context:

| Use case | Why no canned profile |
|---|---|
| Streaming 4K compression QC | Needs `vmaf_full_reference` module (not yet implemented) |
| Compliance / safety audit | Depends heavily on local regulation; users assemble |
| Cinema theatrical QC | Requires colour-managed pipeline + frame-accurate timing |

These are tracked in [`STATUS.md`](../STATUS.md) §4.
