# CineInfini v0.4.5 — Bundle consolidé v0.3.0 → v0.4.5

**Bundle unique** qui regroupe **TOUS** les changements depuis le repo v0.3.0
publié sur GitHub. À appliquer en une seule fois.

## Validation

```
Baseline v0.3.0 (repo GitHub):  24/27 tests passent
Après v0.4.5:                   59/59 tests passent (sandbox sans modèles)
                                33/33 tests core (sans nouveaux modules)
                                26/26 tests engineering (config + calibrate + reader)
```

Tous les tests sont passés sur le vrai code v0.3.0 du ZIP que vous avez fourni.

## Contenu (14 fichiers)

### Nouveaux fichiers (5)

| Fichier | Lignes | Rôle |
|---|---|---|
| `src/cineinfini/core/config.py` | 363 | Singleton YAML centralisé — `get_config()`, `test_config()`, registre URLs |
| `src/cineinfini/core/calibrate.py` | 404 | Calibration scientifique seuils — grid ROC, logistique, Optuna |
| `scripts/download_models.py` | 179 | Téléchargement modèles avec SHA256, reprise |
| `tests/test_engineering_v040.py` | 274 | 26 tests unitaires sur les nouveaux modules |
| `FILES_TO_DELETE.txt` | 1 | Liste des fichiers à `git rm` |

### Fichiers modifiés (9)

| Fichier | Type | Pourquoi |
|---|---|---|
| `src/cineinfini/__init__.py` | RÉÉCRIT | Version 0.2.0 → 0.4.5, exports complets, dataclasses pipeline |
| `src/cineinfini/pipeline/audit.py` | REFACTORÉ | God function 247L (CC=25) → 6 fonctions ≤5 chacune, cv2.cuda fix |
| `src/cineinfini/core/coherence.py` | RÉÉCRIT | Bug v0.2.0 (TypeError sur 2+ shots) — signature multi-compatible |
| `src/cineinfini/core/metrics.py` | ANNOTÉ | Type annotations sur 12 fonctions |
| `src/cineinfini/io/reader.py` | RÉÉCRIT | `detect_shot_boundaries` CC 26 → 3 fonctions pures CC ≤7 |
| `tests/conftest.py` | RÉÉCRIT | Fixture `isolate_global_paths` empêche pollution `~/.cineinfini` |
| `tests/test_advanced.py` | RÉÉCRIT | Suppression chemins Colab `/content/cineinfini-dev/`, suppression xfail masqué |
| `pyproject.toml` | MODIFIÉ | Markers pytest, ruff/mypy config |
| `.github/workflows/ci.yml` | MODIFIÉ | Job lint séparé, exclusion intégration |

### Fichier à supprimer (1)

| Fichier | Pourquoi |
|---|---|
| `src/cineinfini/pipeline/audit_todelete.py` | Zombie de 282 lignes dans le wheel PyPI |

## Application — un seul script

```bash
cd Video_Quality_Auditor/

# 1. Supprimer le fichier zombie
git rm src/cineinfini/pipeline/audit_todelete.py

# 2. Copier les fichiers du bundle (extracted to ./cineinfini-v0_4_5/)
BUNDLE=./cineinfini-v0_4_5

# Nouveaux fichiers
cp $BUNDLE/src/cineinfini/core/config.py     src/cineinfini/core/
cp $BUNDLE/src/cineinfini/core/calibrate.py  src/cineinfini/core/
cp $BUNDLE/scripts/download_models.py        scripts/
cp $BUNDLE/tests/test_engineering_v040.py    tests/

# Fichiers modifiés
cp $BUNDLE/src/cineinfini/__init__.py        src/cineinfini/
cp $BUNDLE/src/cineinfini/pipeline/audit.py  src/cineinfini/pipeline/
cp $BUNDLE/src/cineinfini/core/coherence.py  src/cineinfini/core/
cp $BUNDLE/src/cineinfini/core/metrics.py    src/cineinfini/core/
cp $BUNDLE/src/cineinfini/io/reader.py       src/cineinfini/io/
cp $BUNDLE/tests/conftest.py                 tests/
cp $BUNDLE/tests/test_advanced.py            tests/
cp $BUNDLE/pyproject.toml                    .
cp $BUNDLE/.github/workflows/ci.yml          .github/workflows/

# 3. Reinstaller et tester
pip install -e ".[dev,dtw]"
pip install pyyaml pandas scikit-learn      # required for config + calibrate
pytest tests/ -v -m "not integration and not slow"
# Expected: 59 passed, 2 skipped
```

## Strategy de commits suggérée

Au lieu d'un seul gros commit, faites 4 commits logiques :

```bash
git checkout -b feat/v0.4.5-engineering

# Commit 1: critical bug fixes
git rm src/cineinfini/pipeline/audit_todelete.py
cp $BUNDLE/src/cineinfini/pipeline/audit.py  src/cineinfini/pipeline/
cp $BUNDLE/src/cineinfini/core/coherence.py  src/cineinfini/core/
git commit -m "fix: cv2.cuda crash, audit_video god function, coherence TypeError"

# Commit 2: engineering tests
cp $BUNDLE/tests/conftest.py    tests/
cp $BUNDLE/tests/test_advanced.py tests/
cp $BUNDLE/pyproject.toml       .
cp $BUNDLE/.github/workflows/ci.yml .github/workflows/
git commit -m "test: isolate_global_paths fixture; remove Colab paths"

# Commit 3: centralized config
cp $BUNDLE/src/cineinfini/core/config.py    src/cineinfini/core/
cp $BUNDLE/scripts/download_models.py       scripts/
cp $BUNDLE/src/cineinfini/__init__.py       src/cineinfini/
git commit -m "feat: YAML config singleton + URL registry + download script"

# Commit 4: scientific calibration + reader decomposition
cp $BUNDLE/src/cineinfini/core/calibrate.py src/cineinfini/core/
cp $BUNDLE/src/cineinfini/io/reader.py      src/cineinfini/io/
cp $BUNDLE/src/cineinfini/core/metrics.py   src/cineinfini/core/
cp $BUNDLE/tests/test_engineering_v040.py   tests/
git commit -m "feat: threshold calibration; reader.py CC 26→7; metrics.py type hints"

git push origin feat/v0.4.5-engineering
```

## Notes importantes

1. **Backward compatibility** : tout code existant qui utilise `CONFIG`, `set_global_paths()`,
   `audit_video()` continue de fonctionner. Les nouveautés (`get_config()`, etc.)
   sont additives.

2. **Ordre des migrations futures (v0.5.0)** :
   - `face_detection.py` doit appeler `get_config().models_dir()` au lieu de
     `MODELS_DIR` global
   - `compare.py` doit utiliser le registre d'URLs au lieu d'URLs en dur
   - `report.py::generate_inter_report` (CC=26) à décomposer comme reader.py

3. **Validation perceptuelle BVI-VFI** : le lien que vous avez donné
   (https://data.bris.ac.uk/datasets/tar/k8bfn0qsj9fs1rwnc2x75z6t7.zip) est
   accessible. Le ZIP contient frames pré-extraites + scores DMOS humains.
   Code de calibration `core/calibrate.py` peut directement consommer ces
   scores via `calibrate_from_csv("bvi_vfi_dmos.csv", method="logistic")`.
