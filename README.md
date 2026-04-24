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
```

### From source
```bash
git clone https://github.com/CineInfini/Video_Quality_Auditor.git
cd Video_Quality_Auditor
pip install -e .
```

### Dependencies
- Python 3.9+
- OpenCV, PyTorch, ONNX Runtime, transformers, open_clip_torch, scikit‑image, pandas, matplotlib, weasyprint, etc.  
  (All are installed automatically with the package.)

---

## 🚀 Quick Start

### 1. Basic audit (first 60 seconds)
```python
from cineinfini import audit_video

report_dir, metrics = audit_video("path/to/your_video.mp4")
print(f"Report saved to {report_dir}")
```

### 2. Adaptive audit (two‑stage optimisation)
```python
from cineinfini import adaptive_multi_stage_audit

report_dir = adaptive_multi_stage_audit("video.mp4", force_full_video=False)
```

### 3. Command‑line interface (after installation)
```bash
cineinfini audit video.mp4 --output reports/
cineinfini compare --vids video1.mp4 video2.mp4 --benchmark
```

---

## ⚙️ Configuration

You can modify the global `CONFIG` dictionary before running an audit, or pass a custom parameter dict to `audit_video`:

```python
custom_params = {
    "max_duration_s": 120,          # analyse 2 minutes
    "n_frames_per_shot": 8,         # faster, less precise
    "narrative_coherence": False,   # disable DINOv2 (CPU‑friendly)
    "thresholds": {"motion": 30.0, "ssim3d": 0.4}
}
audit_video("video.mp4", video_params=custom_params)
```

---

## 📊 Outputs

After an audit, the pipeline creates:

- **`reports/intra/<video_name>/`**  
  - `data.json` – raw per‑shot metrics  
  - `dashboard.md` – Markdown report with bar charts, radar, alerts, improvement suggestions  
  - `figures/` – PNG bar charts and radar chart  

- **`reports/inter/<benchmark_name>/`** (benchmark mode)  
  - Inter‑video comparison dashboard with tables, bar charts, and radar  

---

## 📖 Documentation

Full documentation (including API reference, tutorials, and parameter justifications) is available at [https://cineinfini.readthedocs.io](https://cineinfini.readthedocs.io) (coming soon).

---

## 🤝 Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.  
Main areas:  
- Adding new metrics (e.g., TULIP, animal face detection)  
- Improving performance (GPU‑accelerated SSIM, lighter CLIP models)  
- Enhancing narrative coherence (object tracking over time)  

---

## 📜 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

---

## 📄 Citation

If you use CineInfini in your research, please cite it as:

```bibtex
@software{cineinfini2025,
  author = {Salah-Eddine BENBRAHIM},
  title = {CineInfini: Adaptive Multi‑Stage Video Quality Audit Pipeline},
  year = {2025},
  url = {https://github.com/CineInfini/Video_Quality_Auditor},
  doi = {10.xxxx/xxxx}
}
```

---

## 🙏 Acknowledgments

- Blender Foundation for open‑source movies (Big Buck Bunny, Sintel, Tears of Steel)  
- InsightFace (ArcFace), OpenAI (CLIP), Meta AI (DINOv2)  
- OpenCV, scikit‑image, ONNX Runtime, PyTorch, Hugging Face Transformers  

---

**Star ⭐ this repository if you find it useful!**  
For questions or suggestions, open an [issue](https://github.com/CineInfini/Video_Quality_Auditor/issues).
