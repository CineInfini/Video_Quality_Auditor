# 🎬 CineInfini – Video Quality Auditor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/cineinfini-audit.svg)](https://pypi.org/project/cineinfini-audit/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**CineInfini** is an open‑source pipeline for automatic, explainable video quality auditing.

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation)

## ✨ Features
- Adaptive shot detection, 7 intra‑shot metrics, inter‑shot coherence, narrative coherence.
- Two‑stage audit, GPU acceleration, exhaustive dashboards, benchmark mode.

## 🛠 Installation
```bash
pip install cineinfini-audit
```

## 🚀 Quick Start
```python
from cineinfini import audit_video
metrics, report_dir = audit_video("video.mp4")
```

### Command line
```bash
cineinfini audit video.mp4
cineinfini compare --vids v1.mp4 --vids v2.mp4
cineinfini benchmark v1.mp4 v2.mp4 v3.mp4
```

## 📖 Documentation
- **API reference**: [docs/index.html](docs/index.html) (generated with pdoc)
- Example notebooks in `notebooks/`

## 📄 Citation
```bibtex
@software{cineinfini2026,
  author = {Salah-Eddine BENBRAHIM},
  title = {CineInfini},
  year = {2026},
  url = {https://github.com/CineInfini/Video_Quality_Auditor},
  version = {0.1.2}
}
```

## 🙏 Acknowledgments
Blender Foundation, InsightFace, OpenAI, Meta AI, OpenCV, PyTorch.

**Star ⭐ this repository if you find it useful!**
