# CineInfini — Installation Guide

Three install tiers, picked to match your use case:

| Tier | Disk | What you get | When to use |
|---|---|---|---|
| **Minimal** | ~10 MB | 16 pure-CV modules | Quick prototyping, CI gates, profiling |
| **Standard** | ~835 MB | All 19 native modules | Default for most users |
| **Full** | ~1.2 GB | + DOVER + FAST-VQA + cross-benchmark | Research, leaderboard submission, paper benchmarks |

---

## 1. System prerequisites

| Tool | Version | How to install |
|---|---|---|
| Python | ≥ 3.9 | [python.org](https://www.python.org/downloads/) |
| ffmpeg | ≥ 4.0 | `apt install ffmpeg` / `brew install ffmpeg` / `choco install ffmpeg` |
| pip | latest | `python -m pip install --upgrade pip` |

CineInfini does not bundle ffmpeg. The `bootstrap` command will warn
if it's missing.

---

## 2. Tier 1 — Minimal install

```bash
pip install cineinfini-audit
```

This gives you the framework + 16 pure-CV modules:

- motion_coherence, identity_consistency, semantic_consistency
- background_consistency, aesthetic_cinematic, causal_reasoning
- temporal_signature, physics_plausibility, trustworthiness
- world_model_surprise, creative_composition, long_term_narrative
- subject_consistency_long, benchmark_forensic, explainability
- benchmark_fusion

The 3 default-on modules (motion + identity + semantic) work without
any extra weights, so you can immediately run:

```bash
cineinfini audit test.mp4 --config cfg/profiles/ultralight.yaml
```

For `identity_consistency` and `semantic_consistency` to use real
ArcFace and CLIP weights instead of degraded fallbacks, proceed to
Tier 2.

---

## 3. Tier 2 — Standard install (with ML weights)

```bash
pip install cineinfini-audit
cineinfini bootstrap                    # downloads ~835 MB
```

This adds 4 weight files to `~/.cineinfini/models/`:

| Weight | Size | License | Used by |
|---|---|---|---|
| `arcface.onnx` | 166 MB | MIT (yakhyo) | identity_consistency, subject_consistency_long |
| `yunet.onnx` | 232 KB | Apache-2.0 | face detection |
| `ViT-B-32.pt` | 338 MB | MIT (OpenAI) | semantic_consistency, multi_modal_safety, prompt_alignment_fine |
| `dinov2_vitb14.pth` | 330 MB | Apache-2.0 (Meta) | long_term_narrative, origin_detection |

After this you have **all 19 native modules functional**.

### 3.1 If `bootstrap` fails

Common issues:

- **GitHub LFS rate-limit** → wait an hour or retry with `--force`
- **Corporate firewall** → fetch the URL manually (the report shows
  the exact URL + target path), drop the file there, re-run
  `bootstrap` to verify SHA-256
- **Disk space** → models go to `~/.cineinfini/models/`; configure a
  different path in `cfg/config.yaml` `paths.models_dir`

### 3.2 Verifying the install

```bash
cineinfini bootstrap --skip-download    # just verify
cineinfini config --check               # validate the active config
cineinfini audit ~/.cineinfini/test_videos/BBB.mp4
```

If `bootstrap` prints "✓" for all 4 models, you're set.

---

## 4. Tier 3 — Full install (cross-benchmark wrappers)

```bash
# 1. Tier 2 first
pip install cineinfini-audit
cineinfini bootstrap

# 2. PyTorch (DOVER + FAST-VQA both need it)
pip install torch torchvision

# 3. Either upstream packages or local clones
pip install dover-vqa                                       # OR
git clone https://github.com/QualityAssessment/DOVER.git && cd DOVER && pip install -e .

pip install fast-vqa                                        # OR
git clone https://github.com/VQAssessment/FAST-VQA-and-FasterVQA.git && \
    cd FAST-VQA-and-FasterVQA && pip install -e .

# 4. Fetch the optional weights (~395 MB)
cineinfini bootstrap --include-optional
```

After this **`dover_score` and `fastvqa_score` modules report
`available: true`** and their scores appear alongside our native
metrics in `data.json`.

> **Note:** the wrapper modules ship as stubs — they detect the weights
> + dependencies and call documented placeholder functions. To make
> them produce real scores, edit 5 lines in `_run_dover_inference()` /
> `_run_fastvqa_inference()` (the docstrings show exactly what to write
> based on the upstream API). This is by design: we don't vendor third-
> party model code.

---

## 5. Dataset install (validation)

Datasets are not auto-downloaded by `bootstrap` (they're large or
gated). Use `cineinfini datasets`:

```bash
cineinfini datasets                              # list registered datasets
cineinfini datasets --info bvi_hfr               # full details + URL

# Public, auto-downloadable
cineinfini datasets --fetch lfw                                # 170 MB
cineinfini datasets --fetch bvi_hfr --only "*.mp4"             # partial-zip
cineinfini datasets --fetch vbench_eval --only "VBench-master/prompts/*"

# Gated (BVI-VFI requires registration form)
# 1. Fill: https://forms.office.com/e/gtKpYriSMJ
# 2. Wait for the email with the download link from Bristol
# 3. Drop the ZIP at ~/.cineinfini/datasets/BVI-VFI/
# 4. Verify: cineinfini datasets --check
```

See [`MISSING_ASSETS.md`](MISSING_ASSETS.md) for the full list and
[`benchmarking/COMPETITORS.md`](benchmarking/COMPETITORS.md) for one-
liner installs of every supported competitor tool.

---

## 6. Optional GPU acceleration

CineInfini auto-detects CUDA via `torch.cuda.is_available()`. To force
CPU:

```yaml
# cfg/config.yaml
device: "cpu"
```

For `cfg/profiles/low_memory.yaml` users on Apple Silicon, `device: "mps"`
works for the CLIP-based modules.

---

## 7. Containerised install (Docker)

A Dockerfile is on the v0.5.0 roadmap. For now, the recommended
isolation is a venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install cineinfini-audit
cineinfini bootstrap
```

---

## 8. Updating

```bash
pip install --upgrade cineinfini-audit
cineinfini bootstrap                    # idempotent, only fetches new weights
```

If a release introduces a new optional model (rare), it will be in
`cfg.optional_models` and you re-run `--include-optional`.

---

## 9. Uninstalling

```bash
pip uninstall cineinfini-audit
rm -rf ~/.cineinfini                    # removes models, reports, datasets
```
