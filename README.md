# 🎬 CineInfini – Video Quality Auditor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**CineInfini** is an open‑source, modular pipeline for automatic, explainable video quality auditing. It goes beyond traditional fidelity metrics (PSNR, SSIM) by assessing **temporal stability, identity consistency, semantic coherence, and narrative flow** – all in a single GPU‑accelerated pipeline.

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

---

## ✨ Features

- **Adaptive shot detection** – HSV histogram with percentile‑based threshold  
- **7 intra‑shot metrics** – motion peak (optical flow), 3D‑SSIM, flicker, identity drift (ArcFace), long‑range SSIM, high‑frequency flicker, CLIP temporal consistency  
- **Inter‑shot coherence** – structure (SSIM), style (histogram), semantic (CLIP)  
- **Narrative coherence** – DINOv2 cosine similarity between shots  
- **Two‑stage audit** – automatically optimises processing parameters and composite weights  
- **GPU acceleration** – CUDA for CLIP/DINOv2, parallel shot processing (4 workers)  
- **Exhaustive reports** – Markdown dashboards with bar charts, radar plots, improvement suggestions, and JSON exports  
- **Benchmark mode** – compare multiple videos (synthetic / real) with inter‑video dashboards  

---

## 🛠 Installation

### From PyPI (recommended)
```bash
pip install cineinfini-audit
