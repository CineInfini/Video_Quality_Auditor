# CineInfini notebooks

Four runnable Jupyter notebooks. Each one is **self-contained** —
starts from scratch, cleans up after itself, doesn't require any
external download (except notebook 01 which runs the test suite).

| Notebook | Purpose | Runtime |
|---|---|---|
| [`01_unit_tests.ipynb`](01_unit_tests.ipynb) | Run the entire pytest suite from a notebook (CI-ready) | ~40s |
| [`02_integration_tests.ipynb`](02_integration_tests.ipynb) | End-to-end audit pipeline on a synthetic video | ~10s |
| [`03_deployment_tests.ipynb`](03_deployment_tests.ipynb) | Validate `deploy_cineinfini.py` pre-flight checks | ~60s |
| [`04_user_walkthrough.ipynb`](04_user_walkthrough.ipynb) | Typical user journey: 3 videos, audit, score, export | ~20s |

## How to run

```bash
# from the repo root
pip install jupyter nbformat nbclient
export CINEINFINI_REPO=$(pwd)
jupyter notebook notebooks/
```

Or from the command line, without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute notebooks/02_integration_tests.ipynb \
    --ExecutePreprocessor.timeout=120
```

## What each notebook validates

### 01_unit_tests.ipynb
- The package imports cleanly
- All 232 unit tests pass
- The v0.4.8.x deltas (registry, datasets, partial-ZIP, FAST-VQA pinning,
  competitive-parity exporters, profiles) are individually green

### 02_integration_tests.ipynb
- Config loads + module registration works
- Synthetic video is decoded and shot-detected
- Default-on modules produce per-shot scores
- VideoScore aggregator composes 5 axes + composite
- VBench exporter maps to 16 dimensions
- The full pipeline composes end-to-end

### 03_deployment_tests.ipynb
- Deploy script is in place
- Version is consistent across `__init__.py` / `CITATION.cff` / `README.md`
- All 232 tests pass (the pre-deploy gate)
- Every required release artifact (README, CHANGELOG, profiles, docs) exists
- The package builds cleanly with `python -m build`

### 04_user_walkthrough.ipynb
- Three videos with different quality (good/mediocre/broken) get
  ranked correctly by composite score
- Side-by-side comparison via VideoScore axes
- VBench export of the best video produces valid JSON

If any notebook fails, the failure is in the framework — not in the
notebook. Open an issue with the cell output.
