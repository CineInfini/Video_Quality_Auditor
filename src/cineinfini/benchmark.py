"""
CineInfini – Utilitaires de benchmark et audit multiple
"""

import json
import time
from pathlib import Path
from typing import List, Union, Dict, Any
import numpy as np
import pandas as pd

from cineinfini import audit_video, generate_synthetic_video
from cineinfini.io.report import generate_inter_report
from cineinfini.pipeline.audit import CONFIG, REPORTS_DIR


def audit_multiple_videos(
    video_paths: List[Union[str, Path]],
    output_subdir: str = "multi_audit",
    max_duration_s: int = 10,
    force_full_video: bool = False,
) -> Path:
    """
    Lance un audit sur plusieurs vidéos et génère un rapport inter-vidéo global.

    Paramètres
    ----------
    video_paths : liste de chemins (str ou Path)
    output_subdir : nom du sous-dossier dans reports/inter
    max_duration_s : durée max analysée par vidéo
    force_full_video : analyser toute la vidéo

    Retourne
    -------
    Path du dossier contenant le dashboard comparatif.
    """
    reports = []
    for i, vp in enumerate(video_paths):
        vp = Path(vp)
        print(f"[{i+1}/{len(video_paths)}] Audit de {vp.name}...")
        _, report_dir = audit_video(
            str(vp),
            video_params={} if force_full_video else {"max_duration_s": max_duration_s},
            force_full_video=force_full_video
        )
        reports.append(report_dir)

    inter_root = REPORTS_DIR / "inter"
    inter_root.mkdir(parents=True, exist_ok=True)
    generate_inter_report(
        intra_report_dirs=reports,
        output_dir=inter_root,
        thresholds=CONFIG["thresholds"],
        comparison_name=output_subdir
    )
    return inter_root / output_subdir


def run_benchmark(
    video_path: Union[str, Path],
    output_file: Union[str, Path] = None,
    repeats: int = 3,
) -> Dict[str, Any]:
    """
    Exécute un benchmark de performance sur une vidéo (temps d'exécution, mémoire, etc.)

    Paramètres
    ----------
    video_path : chemin de la vidéo
    output_file : si spécifié, sauvegarde le rapport JSON
    repeats : nombre de répétitions pour mesurer la stabilité

    Retourne
    -------
    Dictionnaire contenant les métriques de performance (temps moyen, std, etc.)
    """
    video_path = Path(video_path)
    times = []
    for i in range(repeats):
        print(f"  Run {i+1}/{repeats}...")
        start = time.time()
        _, _ = audit_video(str(video_path), video_params={"max_duration_s": 5}, force_full_video=False)
        elapsed = time.time() - start
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)
    result = {
        "video": str(video_path),
        "repeats": repeats,
        "mean_duration_s": mean_time,
        "std_duration_s": std_time,
        "min_s": min(times),
        "max_s": max(times),
        "times": times,
    }
    print(f"⏱️  Temps moyen : {mean_time:.2f}s (±{std_time:.2f})")

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Benchmark sauvegardé dans {output_file}")
    return result


def generate_test_dataset(
    output_dir: Union[str, Path],
    num_videos: int = 5,
    duration_s: int = 2,
    fps: int = 24,
    resolution: tuple = (320, 240),
    patterns: List[str] = None,
) -> List[Path]:
    """
    Génère un jeu de vidéos synthétiques pour les tests ou les benchmarks.

    Paramètres
    ----------
    output_dir : dossier de sortie
    num_videos : nombre de vidéos à créer
    duration_s : durée en secondes de chaque vidéo
    fps : images par seconde
    resolution : (largeur, hauteur)
    patterns : liste de motifs à utiliser en cycle ("circle", "color_switch", "noise")
                Par défaut : cycle sur les trois.

    Retourne
    -------
    Liste des chemins des vidéos générées.
    """
    if patterns is None:
        patterns = ["circle", "color_switch", "noise"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    for i in range(num_videos):
        pattern = patterns[i % len(patterns)]
        name = output_dir / f"test_video_{i+1}_{pattern}.mp4"
        generate_synthetic_video(str(name), duration_s, fps, resolution, pattern)
        videos.append(name)
        print(f"✅ Créé : {name}")
    return videos


def benchmark_models(
    video_path: Union[str, Path],
    model_names: List[str] = None,
    output_file: Union[str, Path] = None,
) -> Dict[str, float]:
    """
    Compare les temps de chargement et d'inférence des modèles (DINOv2, CLIP, ArcFace, etc.)

    Paramètres
    ----------
    video_path : vidéo de test (utilisée pour une petite inférence)
    model_names : liste des modèles à tester ("dinov2", "clip", "arcface", "yunet")
                  Par défaut : tous.
    output_file : fichier JSON optionnel pour sauvegarder les résultats.

    Retourne
    -------
    Dictionnaire {nom_modele: temps_en_secondes}
    """
    import time
    from cineinfini.core.embedding import load_dinov2, get_dinov2, CLIPSemanticScorer
    from cineinfini.core.face_detection import ArcFaceEmbedder, FaceDetector

    if model_names is None:
        model_names = ["dinov2", "clip", "arcface", "yunet"]

    results = {}
    video_path = str(video_path)

    for model in model_names:
        print(f"Benchmarking {model}...")
        start = time.time()
        if model == "dinov2":
            load_dinov2("cpu")
            _ = get_dinov2()
        elif model == "clip":
            scorer = CLIPSemanticScorer(device="cpu")
            _ = scorer.score([], "dummy")  # initialisation
        elif model == "arcface":
            embedder = ArcFaceEmbedder()
            _ = embedder  # déclenche le chargement
        elif model == "yunet":
            detector = FaceDetector()
            _ = detector
        else:
            print(f"  Modèle inconnu : {model}, ignoré")
            continue
        elapsed = time.time() - start
        results[model] = elapsed
        print(f"  -> {elapsed:.2f}s")

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    return results


def compare_multiple_videos(
    video_list: List[Union[str, Path]],
    output_subdir: str = "comparison_group",
    max_duration_s: int = 10,
) -> Path:
    """
    Alias de audit_multiple_videos – comparer plus de deux vidéos.
    """
    return audit_multiple_videos(video_list, output_subdir, max_duration_s)
