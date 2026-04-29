# CineInfini — 5-minute Quickstart

```bash
# 1. Install
pip install cineinfini-audit

# 2. One-time bootstrap of ~835 MB of weights (skip if you'll only use pure-CV)
cineinfini bootstrap

# 3. Audit a video
cineinfini audit my_video.mp4 --config cfg/profiles/postproduction.yaml

# 4. Read the report
xdg-open ~/.cineinfini/reports/my_video/dashboard.html

# 5. Cross-tool exports
cineinfini export-vbench  ~/.cineinfini/reports/my_video/    # → vbench.json
cineinfini score          ~/.cineinfini/reports/my_video/    # → 5 axes + composite
```

That's the whole loop. For details:

- [`USER_MANUAL.md`](USER_MANUAL.md) — complete reference
- [`INSTALLATION.md`](INSTALLATION.md) — install variants + competitor deps
- [`STATUS.md`](STATUS.md) — what's done / not done / what to install
- [`../notebooks/04_user_walkthrough.ipynb`](../notebooks/04_user_walkthrough.ipynb) — same flow in Jupyter

## Five common one-liners

```bash
# Audit ALL videos in a directory and aggregate
cineinfini benchmark ./videos/ --config cfg/profiles/postproduction.yaml

# Real-time monitoring (< 2s/min CPU, 3 modules only)
cineinfini audit live.mp4 --config cfg/profiles/realtime.yaml

# Maximum precision — all 21 modules + DOVER + FAST-VQA
pip install torch torchvision dover-vqa fast-vqa
cineinfini bootstrap --include-optional
cineinfini audit video.mp4 --config cfg/profiles/academic.yaml

# Side-by-side comparison
cineinfini compare original.mp4 generated.mp4

# Inspect a remote ZIP without downloading the whole thing
cineinfini datasets --list-files bvi_hfr
cineinfini datasets --fetch bvi_hfr --only "*.mp4"
```

## What the output looks like

```
~/.cineinfini/reports/my_video/
├── data.json              # all 21 modules' raw output
├── dashboard.html         # interactive dashboard
├── dashboard.md           # markdown summary
├── report.pdf             # print-ready
├── audit.vbench.json      # 16-dim VBench format (after export-vbench)
└── videoscore.json        # 5-axis + composite (after score)
```

A typical `data.json` excerpt:

```json
{
  "video": {"name": "ai_clip.mp4", "duration_s": 5.0, "n_shots": 3},
  "gates": {
    "1": {
      "verdict": "ACCEPT",
      "motion_peak_div": 1.8,
      "ssim3d_self": 0.91,
      "identity_within_shot_dtw": 0.12,
      "failed_gates": []
    },
    "2": {
      "verdict": "REVIEW",
      "motion_peak_div": 4.1,
      "flicker_score": 18.0,
      "failed_gates": ["flicker_score"]
    },
    "3": {"verdict": "REJECT", "failed_gates": ["motion_peak_div", "ssim3d_self"]}
  },
  "composite_score": 0.671
}
```

The "verdict" + "failed_gates" combo is CineInfini's signature feature —
you know not just *that* a video has issues but *which* shot, *which* gate,
and by *how much*.
