# src/cineinfini/core/config.py
"""Centralised configuration singleton.

All modules read state through ``get_config()``. Bootstrap helpers expose
typed accessors for ``model_urls`` and the new ``test_videos`` section so
the bootstrap module can resolve assets without hard-coded URLs.
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# Validation framework  (NEW v0.4.8.8)
# ---------------------------------------------------------------------------
class ConfigValidationError(Exception):
    """Raised when :meth:`Config.validate` finds structural errors.

    The message contains a bullet-listed report of every missing key and
    every type mismatch. Used by ``deploy_cineinfini.py`` to fail fast
    before attempting a release.
    """


# Required (key, type) pairs per **flat** section. Sections that are
# dict-of-dicts (model_urls, test_videos, datasets, optional_models, modules)
# are validated separately for "is a dict" only — the inner schema is the
# responsibility of each module.
_VALIDATION_SCHEMA: Dict[str, List[Tuple[str, Union[type, Tuple[type, ...]]]]] = {
    "paths": [
        ("models_dir", str),
        ("reports_dir", str),
        ("benchmark_dir", str),
        ("test_videos_dir", str),
        ("cache_dir", str),
        ("logs_dir", str),
        ("temp_dir", str),
        ("output_root", str),
    ],
    "device": [
        ("gpu_device", str),
        ("torch_dtype", str),
    ],
    "processing": [
        ("max_duration_s", (int, float)),
        ("shot_threshold", (int, float)),
        ("n_frames_per_shot", int),
        ("num_workers", int),
        ("embedder", str),
        ("semantic_scorer", str),
    ],
    "thresholds": [
        ("motion", (int, float)),
        ("ssim3d", (int, float)),
        ("flicker", (int, float)),
        ("trust_score", (int, float)),
    ],
    "reporting": [
        ("generate_markdown", bool),
        ("generate_plots", bool),
    ],
    "logging": [
        ("level", str),
        ("file_enabled", bool),
        ("console_enabled", bool),
    ],
}

# Sections that must exist as dicts but whose internals are not schema-checked.
_DICT_SECTIONS: Tuple[str, ...] = (
    "model_urls", "test_videos", "datasets", "optional_models", "modules",
)


# ---------------------------------------------------------------------------
# Deep-merge utility  (NEW v0.4.8.8)
# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive deep merge — *override* always wins on conflicting keys.

    Sub-dicts are merged recursively. Lists are replaced (not concatenated)
    to match standard YAML override semantics. Inputs are not mutated.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_D_PATHS = {
    "models_dir": "~/.cineinfini/models",
    "reports_dir": "~/.cineinfini/reports",
    "benchmark_dir": "~/.cineinfini/benchmark",
    "test_videos_dir": "~/.cineinfini/test_videos",
    "datasets_dir": "~/.cineinfini/datasets",
    "cache_dir": "~/.cineinfini/cache",
    "logs_dir": "~/.cineinfini/logs",
    "temp_dir": "/tmp/cineinfini",
    "output_root": "~/.cineinfini/output",
}

_D_DEVICE = {"gpu_device": "auto", "torch_dtype": "float16", "use_amp": True}

_D_PROCESSING = {
    "max_duration_s": 60, "shot_threshold": 0.2, "min_shot_duration_s": 0.5,
    "downsample_to": [320, 180], "n_frames_per_shot": 16,
    "frame_resize": [320, 180], "step": 2,
    "adaptive_threshold": True, "threshold_percentile": 85,
    "num_workers": 4, "parallel_shots": True,
    "inter_shot_subsample": 5, "narrative_coherence": True,
    "compute_dtw_self": True, "compute_dtw_inter": True,
    "dtw_max_samples": 16, "benchmark_mode": True,
    "embedder": "arcface_onnx", "semantic_scorer": "clip",
    "hist_resize": [160, 90], "hist_bins_hue": 20, "hist_bins_sat": 20,
}

_D_THRESHOLDS = {
    "motion": 25.0, "ssim3d": 0.45, "flicker": 0.10,
    "identity_drift": 0.60, "ssim_long_range": 0.45,
    "clip_temp": 0.25, "flicker_hf": 0.01,
    "narrative_coherence": 0.70, "temporal_coherence": 0.75,
    "physics_overall": 0.65, "aesthetic_score": 0.60,
    "causal_violation": 0.35, "trust_score": 0.70,
    "background_ssim": 0.55,
}

_D_MODEL_URLS = {
    "arcface": {
        "url": "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx",
        "filename": "arcface.onnx",
        "sha256": None,
        "required": False,
    },
    "yunet": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "filename": "yunet.onnx",
        "sha256": None,
        "required": False,
    },
    "clip_vit_b32": {
        "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "filename": "ViT-B-32.pt",
        "sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
        "required": False,
    },
    "dinov2_vitb14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "filename": "dinov2_vitb14.pth",
        "sha256": None,
        "required": False,
    },
}

# Royalty-free open-licence test videos used by the audit / benchmark suite.
_D_TEST_VIDEOS = {
    "bbb_sunflower": {
        "url": "https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_60fps_normal.mp4",
        "filename": "bbb_sunflower.mp4",
        "size_mb": 340,
        "sha256": None,
        "license": "CC-BY-3.0 (Blender Foundation)",
        "description": "Big Buck Bunny - Sunflower demo, 1080p 60fps, 10:34",
        "required": False,
    },
    "BBB": {
        "url": "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
        "filename": "BBB.mp4",
        "size_mb": 50,
        "sha256": None,
        "license": "CC-BY-3.0 (Blender Foundation)",
        "description": "Big Buck Bunny - small 320x180 (smoke-test only)",
        "required": False,
    },
    "tears_of_steel": {
        "url": "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        "filename": "tears_of_steel.mov",
        "size_mb": 365,
        "sha256": None,
        "license": "CC-BY-3.0 (Blender Foundation)",
        "description": "Tears of Steel - 720p MOV, 12:14",
        "required": False,
    },
    "sintel": {
        "url": "https://archive.org/download/sintel-open-movie-by-blender/Sintel%20-%20Open%20Movie%20by%20Blender.mp4",
        "filename": "sintel.mp4",
        "size_mb": 224,
        "sha256": None,
        "license": "CC-BY-3.0 (Blender Foundation)",
        "description": "Sintel - full short film, 14:48 (Internet Archive mirror)",
        "required": False,
    },
    "sintel_trailer": {
        "url": "https://download.blender.org/durian/trailer/sintel_trailer-1080p.mp4",
        "filename": "sintel_trailer.mp4",
        "size_mb": 50,
        "sha256": None,
        "license": "CC-BY-3.0 (Blender Foundation)",
        "description": "Sintel - 1080p trailer (smoke-test)",
        "required": False,
    },
    "elephants_dream": {
        "url": "https://download.blender.org/ED/ED_1024.avi",
        "filename": "elephants_dream.avi",
        "size_mb": 410,
        "sha256": None,
        "license": "CC-BY-2.5 (Blender Foundation)",
        "description": "Elephants Dream - 1024p AVI, 10:54",
        "required": False,
    },
}

# Validation / calibration datasets. Distinct from model_urls and test_videos
# because they typically require human registration, are large, and are used
# for benchmarking / threshold calibration rather than at audit run-time.
# The bootstrap module will NOT auto-download these — it prints instructions.
_D_DATASETS = {
    # ----- BVI-HFR (PUBLIC, direct download — source videos for BVI-VFI) -----
    "bvi_hfr": {
        "name": "BVI-HFR",
        "description": (
            "High-Frame-Rate source video dataset from University of Bristol. "
            "200+ source sequences at multiple frame rates. PUBLIC — direct "
            "download (no registration). The reference videos used to "
            "generate BVI-VFI."
        ),
        "purpose": "training_source",
        "url": "https://data.bris.ac.uk/datasets/tar/k8bfn0qsj9fs1rwnc2x75z6t7.zip",
        "homepage": "https://data.bris.ac.uk/data/dataset/k8bfn0qsj9fs1rwnc2x75z6t7",
        "license": "Academic use only (University of Bristol)",
        "size_gb": 40.0,
        "format": "zip",
        "partial_patterns": ["*.mp4", "*.yuv", "*.txt", "*.csv", "README*"],
        "auto_downloadable": True,
        "required": False,
    },
    # ----- BVI-VFI removed in v0.4.8.9 — labels gated behind registration form
    # impedes reproducibility. Use T2VQA-DB or VideoFeedback for calibration.
    # ----- VFIPS (paired-comparison VFI, validates the new VFI artifact module) -----
    "vfips": {
        "name": "VFIPS",
        "description": (
            "Video Frame Interpolation Perceptual Similarity — 2AFC "
            "paired comparisons rather than scalar MOS. The right "
            "validation target for the temporal_smoothness_via_interp_"
            "artifacts module."
        ),
        "purpose": "calibration_vfi_artifacts",
        "homepage": "https://github.com/laulampaul/VFIPS",
        "paper": "Hou et al., 'VFIPS: Video Frame Interpolation Perceptual Similarity Metric', 2022",
        "license": "Research only — citation required",
        "label_format": "2AFC pairs",
        "size_gb": 5.0,
        "auto_downloadable": False,
        "required": False,
    },
    # ----- T2VQA-DB (largest AIGC-VQA dataset with MOS) -----
    "t2vqa_db": {
        "name": "T2VQA-DB",
        "description": (
            "10K T2V-generated videos from 9 models with single MOS [1-100] "
            "from 27 annotators. Standard AIGC-VQA benchmark — Kou et al. ACM MM 2024."
        ),
        "purpose": "calibration_primary",
        "homepage": "https://github.com/QMME/T2VQA",
        "paper": "https://arxiv.org/abs/2403.11956",
        "license": "Research only — citation required",
        "label_field": "mos_overall",
        "label_range": [1, 100],
        "n_videos": 10000,
        "expected_spearman_target": 0.65,
        "auto_downloadable": False,
        "required": False,
    },
    # ----- CRAVE-DB (next-gen AIGC, harder than T2VQA-DB) -----
    "crave_db": {
        "name": "CRAVE-DB",
        "description": (
            "1228 next-gen AIGC videos (Sora/Hunyuan-era), MOS from 29 "
            "annotators. Harder than T2VQA-DB — the previous-generation "
            "artifacts (flickering, weak motion, short content) are "
            "rare in this set."
        ),
        "purpose": "calibration_secondary",
        "homepage": "https://github.com/CRAVE-Bench/CRAVE",
        "paper": "https://arxiv.org/abs/2502.04076",
        "license": "Research only",
        "n_videos": 1228,
        "auto_downloadable": False,
        "required": False,
    },
    # ----- KoNViD-1k (sanity check on natural videos) -----
    "konvid_1k": {
        "name": "KoNViD-1k",
        "description": (
            "1200 natural in-the-wild videos with crowdsourced MOS. "
            "Standard NR-VQA sanity-check on natural content (not AIGC). "
            "If CineInfini scores poorly here, it's miscalibrated."
        ),
        "purpose": "calibration_natural_videos",
        "homepage": "http://database.mmsp-kn.de/konvid-1k-database.html",
        "paper": "Hosu et al., QoMEX 2017",
        "license": "CC-BY-NC 4.0",
        "n_videos": 1200,
        "expected_spearman_target": 0.75,
        # Videos are gated behind a registration form on the official site;
        # however, the MOS labels (1200 rows) are mirrored on the CNN-TLVQM
        # GitHub repo by Korhonen and can be fetched without auth.
        "labels_url": (
            "https://github.com/jarikorhonen/cnn-tlvqm/raw/refs/heads/master/"
            "KoNViD_mos_fr.csv"
        ),
        "labels_filename": "KoNViD_mos_fr.csv",
        "labels_format": "csv_two_column",  # file_name, mos
        "videos_url": None,  # gated — user must register
        "auto_downloadable": False,  # videos are gated; labels are downloadable
        "labels_auto_downloadable": True,  # labels yes
        "required": False,
    },
    # ----- LFW (PUBLIC, direct download — face validation) -----
    "lfw": {
        "name": "LFW",
        "description": (
            "Labelled Faces in the Wild — 13,233 face images of 5,749 "
            "people. Used to validate ArcFace identity embeddings."
        ),
        "purpose": "validation",
        "url": "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
        "homepage": "http://vis-www.cs.umass.edu/lfw",
        "license": "Public domain (research use)",
        "size_gb": 0.17,
        "format": "tgz",
        "auto_downloadable": True,
        "required": False,
    },
    # ----- VideoFeedback (HuggingFace) -----
    "videofeedback": {
        "name": "TIGER-Lab/VideoFeedback",
        "description": (
            "37,661 AIGC video samples (33.6K annotated + 4.08K real) "
            "with 5-dimension MOS [1-4] from 27 annotators. The standard "
            "validation set for VideoScore-style metrics. The frames + "
            "labels are 12.6 MB; actual MP4 videos live at the "
            "videos_mirror URL and are downloaded per-sample on demand."
        ),
        "purpose": "calibration_primary",
        "homepage": "https://huggingface.co/datasets/TIGER-Lab/VideoFeedback",
        "videos_mirror": "https://huggingface.co/datasets/hexuan21/VideoFeedback-videos-mp4",
        "paper": "https://arxiv.org/abs/2406.15252",
        "license": "Apache-2.0",
        "size_mb": 12.6,
        "n_samples": 37661,
        "n_annotated": 33600,
        "n_real": 4080,
        "label_fields": [
            "visual quality", "temporal consistency", "dynamic degree",
            "text-to-video alignment", "factual consistency",
        ],
        "label_range": [1, 4],
        "expected_spearman_target": 0.70,
        "auto_downloadable": True,
        "fetch_method": "huggingface_datasets",
        "fetch_command": "load_dataset('TIGER-Lab/VideoFeedback', name='annotated')",
        "required": False,
    },
    # ----- VBench eval suite (GitHub) -----
    "vbench_eval": {
        "name": "VBench eval suite",
        "description": (
            "16-dimension AIGC evaluation prompts + reference videos. "
            "Standard for cross-paper comparison."
        ),
        "purpose": "benchmark",
        "url": "https://github.com/Vchitect/VBench/archive/refs/heads/master.zip",
        "homepage": "https://github.com/Vchitect/VBench",
        "paper": "https://arxiv.org/abs/2311.17982",
        "license": "Apache-2.0",
        "size_gb": 1.5,  # the source code + prompts; full eval set is larger
        "format": "zip",
        "partial_patterns": ["VBench-master/prompts/*", "VBench-master/*.json", "VBench-master/*.txt"],
        "auto_downloadable": True,
        "required": False,
    },
}

# Additional ML-weight URLs that can be downloaded directly (not videos but
# weight files for the optional cross-benchmark modules).
_D_OPTIONAL_MODELS = {
    "dover": {
        "url": "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth",
        "filename": "DOVER.pth",
        "description": "DOVER aesthetic + technical UGC scorer (ICCV 2023)",
        "license": "MIT",
        "size_mb": 200,
        # Future-proofing: GitHub release coordinates for resolver fallback.
        "github_repo": "QualityAssessment/DOVER",
        "github_tag": "v0.1.0",
        "asset_name": "DOVER.pth",
        "required": False,
    },
    "dover_mobile": {
        "url": "https://github.com/QualityAssessment/DOVER/releases/download/v0.5.0/DOVER-Mobile.pth",
        "filename": "DOVER-Mobile.pth",
        "description": "DOVER-Mobile (5.7× smaller, CPU-friendly)",
        "license": "MIT",
        "size_mb": 35,
        "github_repo": "QualityAssessment/DOVER",
        "github_tag": "v0.5.0",
        "asset_name": "DOVER-Mobile.pth",
        "required": False,
    },
    # FAST-VQA / FAST-VQA-M from the v1.0.0-open-release-weights tag.
    # The author confirmed in the release notes that these are the v0.3 paper
    # weights. The tag itself is stable once published; the URL is therefore
    # safe to hard-code. We keep github_repo/github_tag/asset_name in the
    # entry so `resolve_github_release_asset()` can fall back to the GitHub
    # API if the hard-coded URL ever 404s (Strategy #2 of the user's note).
    "fastvqa": {
        "url": (
            "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/"
            "releases/download/v1.0.0-open-release-weights/fast-vqa_v0_3.pth"
        ),
        "filename": "fast-vqa_v0_3.pth",
        "description": (
            "FAST-VQA base model, v0.3 weights from the ECCV 2022 paper. "
            "Fragment-sampling VQA with Swin-T backbone, ~210x FLOP reduction "
            "vs dense baselines."
        ),
        "license": "Apache-2.0",
        "size_mb": 110,
        "github_repo": "VQAssessment/FAST-VQA-and-FasterVQA",
        "github_tag": "v1.0.0-open-release-weights",
        "asset_name": "fast-vqa_v0_3.pth",
        "paper": "https://arxiv.org/abs/2207.02595",
        "required": False,
    },
    "fastvqa_m": {
        "url": (
            "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/"
            "releases/download/v1.0.0-open-release-weights/fast-vqa_m-v0_3.pth"
        ),
        "filename": "fast-vqa_m-v0_3.pth",
        "description": (
            "FAST-VQA-M (mobile) model, v0.3 weights. Smaller backbone for "
            "edge / CPU deployment, faster than FAST-VQA-B with similar SRCC."
        ),
        "license": "Apache-2.0",
        "size_mb": 50,
        "github_repo": "VQAssessment/FAST-VQA-and-FasterVQA",
        "github_tag": "v1.0.0-open-release-weights",
        "asset_name": "fast-vqa_m-v0_3.pth",
        "paper": "https://arxiv.org/abs/2207.02595",
        "required": False,
    },
}

_D_MODULES = {
    "motion_coherence": {"enabled": True, "threshold": 25.0},
    "identity_consistency": {
        "enabled": True, "model": "arcface", "threshold": 0.60,
        "n_samples": 5, "use_dtw": True,
    },
    "semantic_consistency": {"enabled": True, "model": "clip", "threshold": 0.25},
    "background_consistency": {
        "enabled": False, "method": "ssim", "threshold": 0.55, "subsample": 5,
    },
    "origin_detection": {"enabled": False, "confidence_threshold": 0.65},
    "temporal_signature": {
        "enabled": False, "max_frames": 30,
        "optical_flow_params": {"pyr_scale": 0.5, "levels": 3, "winsize": 15, "iterations": 3},
    },
    "physics_plausibility": {"enabled": False, "min_contour_area": 300, "iou_threshold": 0.3},
    "trustworthiness": {"enabled": False, "noise_level": 0.05, "n_samples": 5},
    "explainability": {"enabled": False, "method": "proportional", "n_permutations": 100},
    "benchmark_fusion": {"enabled": False, "vbench_path": None},
    "benchmark_forensic": {"enabled": False, "compression_crfs": [23, 28, 35]},
    "causal_reasoning": {"enabled": False, "gravity_penalty": 8.0, "vertical_flow_threshold": 0.1},
    "long_term_narrative": {"enabled": False, "segment_size": 16, "similarity_metric": "cosine"},
    "aesthetic_cinematic": {
        "enabled": False, "use_rule_of_thirds": True,
        "color_harmony_weight": 0.4, "contrast_weight": 0.3, "composition_weight": 0.3,
    },
    "prompt_alignment_fine": {"enabled": False, "model": "blip2", "vlm_device": "cpu"},
    "world_model_surprise": {"enabled": False, "use_divergence": True, "surprise_threshold": 0.7},
    "subject_consistency_long": {"enabled": False, "window_size": 30, "dtw_enabled": True},
    "multi_modal_safety": {"enabled": False, "nsfw_threshold": 0.8, "violence_threshold": 0.7},
    "creative_composition": {"enabled": False, "rhythm_variance_weight": 0.5},
    # ---- Cross-benchmark wrappers (opt-in; need pretrained weights) ----
    "dover_score": {"enabled": False},
    "fastvqa_score": {"enabled": False},
}

_D_REPORTING = {
    "active_renderers": ["markdown", "json"],
    "figure_format": "png", "figure_dpi": 150,
    "theme": "dark", "interactive": True,
    "generate_markdown": True, "generate_html": False,
    "generate_plots": True, "save_raw_data": True,
    "include_shapley": False, "dashboard_theme": "dark",
}

_D_LOGGING = {
    "level": "INFO", "file_enabled": True, "console_enabled": True,
    "format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
}


# ---------------------------------------------------------------------------
@dataclass
class Config:
    paths: Dict[str, str] = field(default_factory=lambda: dict(_D_PATHS))
    device: Dict[str, Any] = field(default_factory=lambda: dict(_D_DEVICE))
    processing: Dict[str, Any] = field(default_factory=lambda: dict(_D_PROCESSING))
    thresholds: Dict[str, float] = field(default_factory=lambda: dict(_D_THRESHOLDS))
    model_urls: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _D_MODEL_URLS.items()}
    )
    test_videos: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _D_TEST_VIDEOS.items()}
    )
    datasets: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _D_DATASETS.items()}
    )
    optional_models: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _D_OPTIONAL_MODELS.items()}
    )
    modules: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _D_MODULES.items()}
    )
    reporting: Dict[str, Any] = field(default_factory=lambda: dict(_D_REPORTING))
    logging: Dict[str, Any] = field(default_factory=lambda: dict(_D_LOGGING))

    # ---- path helpers -----------------------------------------------------
    def resolve_path(self, key: str) -> Path:
        return Path(self.paths.get(key, f"~/.cineinfini/{key}")).expanduser().resolve()

    def models_dir(self) -> Path:
        return self.resolve_path("models_dir")

    def reports_dir(self) -> Path:
        return self.resolve_path("reports_dir")

    def benchmark_dir(self) -> Path:
        return self.resolve_path("benchmark_dir")

    def test_videos_dir(self) -> Path:
        return self.resolve_path("test_videos_dir")

    def datasets_dir(self) -> Path:
        return self.resolve_path("datasets_dir")

    def cache_dir(self) -> Path:
        return self.resolve_path("cache_dir")

    def logs_dir(self) -> Path:
        return self.resolve_path("logs_dir")

    def model_path(self, key: str) -> Optional[Path]:
        e = self.model_urls.get(key)
        return self.models_dir() / e["filename"] if e else None

    def test_video_path(self, key: str) -> Optional[Path]:
        e = self.test_videos.get(key)
        return self.test_videos_dir() / e["filename"] if e else None

    def dataset_dir(self, key: str) -> Optional[Path]:
        """Return the expected directory for a registered dataset."""
        e = self.datasets.get(key)
        if not e:
            return None
        return self.datasets_dir() / e.get("name", key)

    def dataset_present(self, key: str) -> bool:
        """Return True iff the dataset directory exists and contains files."""
        d = self.dataset_dir(key)
        if d is None or not d.exists() or not d.is_dir():
            return False
        try:
            return any(d.iterdir())
        except OSError:
            return False

    # ---- module helpers ---------------------------------------------------
    def is_module_enabled(self, name: str) -> bool:
        return bool(self.modules.get(name, {}).get("enabled", False))

    def is_enabled(self, name: str) -> bool:
        return self.is_module_enabled(name)

    def get_module_config(self, name: str) -> Dict[str, Any]:
        return self.modules.get(name, {})

    def enabled_modules(self) -> List[str]:
        return [n for n, c in self.modules.items() if c.get("enabled", False)]

    # ---- reporting helpers ------------------------------------------------
    def active_renderers(self) -> List[str]:
        # Accept either 'active_renderers' or the user-friendly 'formats' alias.
        explicit = self.reporting.get("active_renderers") or self.reporting.get("formats")
        if explicit is not None:
            return list(explicit)
        out = []
        if self.reporting.get("generate_markdown", True):
            out.append("markdown")
        if self.reporting.get("generate_html", False):
            out.append("html")
        if self.reporting.get("save_raw_data", True):
            out.append("json")
        return out

    def figure_format(self) -> str:
        return str(self.reporting.get("figure_format", "png")).lower()

    def figure_dpi(self) -> int:
        return int(self.reporting.get("figure_dpi", 150))

    def theme(self) -> str:
        return str(self.reporting.get("theme", self.reporting.get("dashboard_theme", "dark")))

    # ---- device -----------------------------------------------------------
    def effective_device(self) -> str:
        dev = str(self.device.get("gpu_device", "auto")).lower()
        if dev == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return dev

    @staticmethod
    def is_jupyter() -> bool:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is None:
                return False
            return type(ip).__name__ in {"ZMQInteractiveShell", "Shell"}
        except Exception:
            return "ipykernel" in sys.modules

    def to_audit_config(self) -> dict:
        cfg = dict(self.processing)
        cfg["thresholds"] = dict(self.thresholds)
        cfg["gpu_device"] = self.effective_device()
        cfg["enable_animal_face_detection"] = False
        return cfg

    # ---- (de)serialisation ----------------------------------------------
    @staticmethod
    def _merge_reporting(defaults: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user reporting on top of defaults, with `formats` translated.

        Users may write `formats: [...]` (more intuitive) or
        `active_renderers: [...]` (canonical). The first one provided wins.
        """
        merged = {**defaults, **(user or {})}
        # If the user explicitly set `formats` but not `active_renderers`,
        # treat it as the canonical override.
        if user and "formats" in user and "active_renderers" not in user:
            merged["active_renderers"] = list(user["formats"])
        return merged

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        dev_data = data.get("device", {})
        if isinstance(dev_data, str):
            device = {**_D_DEVICE, "gpu_device": dev_data}
        else:
            device = {**_D_DEVICE, **(dev_data or {})}
        merged_modules = {k: dict(v) for k, v in _D_MODULES.items()}
        for name, entry in (data.get("modules") or {}).items():
            base = merged_modules.get(name, {})
            if isinstance(entry, str):
                raise ValueError(
                    f"Config error: modules.{name} is a string ({entry!r}), "
                    f"expected a mapping. Likely cause: missing space after ':' "
                    f"in YAML inline mapping (write 'name: {{enabled: true}}' "
                    f"not 'name:{{enabled: true}}')."
                )
            merged_modules[name] = {**base, **(entry or {})}
        merged_videos = {k: dict(v) for k, v in _D_TEST_VIDEOS.items()}
        for name, entry in (data.get("test_videos") or {}).items():
            base = merged_videos.get(name, {})
            merged_videos[name] = {**base, **(entry or {})}
        merged_datasets = {k: dict(v) for k, v in _D_DATASETS.items()}
        for name, entry in (data.get("datasets") or {}).items():
            base = merged_datasets.get(name, {})
            merged_datasets[name] = {**base, **(entry or {})}
        merged_opt_models = {k: dict(v) for k, v in _D_OPTIONAL_MODELS.items()}
        for name, entry in (data.get("optional_models") or {}).items():
            base = merged_opt_models.get(name, {})
            merged_opt_models[name] = {**base, **(entry or {})}
        return cls(
            paths={**_D_PATHS, **(data.get("paths") or {})},
            device=device,
            processing={**_D_PROCESSING, **(data.get("processing") or {})},
            thresholds={**_D_THRESHOLDS, **(data.get("thresholds") or {})},
            model_urls={**{k: dict(v) for k, v in _D_MODEL_URLS.items()},
                        **(data.get("model_urls") or {})},
            test_videos=merged_videos,
            datasets=merged_datasets,
            optional_models=merged_opt_models,
            modules=merged_modules,
            reporting=cls._merge_reporting(_D_REPORTING, data.get("reporting") or {}),
            logging={**_D_LOGGING, **(data.get("logging") or {})},
        )

    def replace(self, **overrides) -> "Config":
        import dataclasses
        return dataclasses.replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ---- validation / diagnostics  (NEW v0.4.8.8) -------------------------
    def validate(self) -> None:
        """Validate this config against the structural schema.

        Walks :data:`_VALIDATION_SCHEMA` and the dict-section list, collecting
        every issue (missing required key, wrong type, non-dict section). If
        any issue is found, raises :class:`ConfigValidationError` with a
        bullet-listed report of *all* of them at once — so the deploy script
        can show every problem in a single failure rather than one per run.

        Examples
        --------
        >>> from cineinfini.core.config import default_config
        >>> default_config().validate()  # passes silently
        >>> bad = default_config()
        >>> bad.thresholds.pop("motion")
        25.0
        >>> bad.validate()
        Traceback (most recent call last):
            ...
        ConfigValidationError: ...thresholds.motion...
        """
        errors: List[str] = []
        section_map: Dict[str, Dict[str, Any]] = {
            "paths": self.paths,
            "device": self.device,
            "processing": self.processing,
            "thresholds": self.thresholds,
            "reporting": self.reporting,
            "logging": self.logging,
        }
        # Flat sections — schema-checked.
        for section, required in _VALIDATION_SCHEMA.items():
            data = section_map.get(section)
            if not isinstance(data, dict):
                errors.append(f"section '{section}' must be a dict, got {type(data).__name__}")
                continue
            for key, expected in required:
                if key not in data:
                    errors.append(f"{section}.{key}: required key missing")
                    continue
                value = data[key]
                if not isinstance(value, expected):
                    if isinstance(expected, tuple):
                        exp = " | ".join(t.__name__ for t in expected)
                    else:
                        exp = expected.__name__
                    errors.append(
                        f"{section}.{key}: expected {exp}, got "
                        f"{type(value).__name__} ({value!r})"
                    )
        # Dict-of-dicts sections — only check that they're dicts.
        for section in _DICT_SECTIONS:
            value = getattr(self, section, None)
            if not isinstance(value, dict):
                errors.append(
                    f"section '{section}' must be a dict, got "
                    f"{type(value).__name__}"
                )
        if errors:
            msg = "Configuration validation failed:\n" + "\n".join(
                f"  • {e}" for e in errors
            )
            raise ConfigValidationError(msg)

    def diff(self, other: "Config") -> Dict[str, Any]:
        """Return a nested dict of differences between *self* and *other*.

        The shape is ``{section: {key: (self_value, other_value)}}``. Only
        keys whose value differs are included. Useful to compare a YAML file
        on disk against the in-memory config, or to log what changed when a
        profile is applied on top of the production config.

        Examples
        --------
        >>> a = default_config()
        >>> b = default_config()
        >>> b.thresholds["motion"] = 30.0
        >>> a.diff(b)
        {'thresholds': {'motion': (25.0, 30.0)}}
        """
        diffs: Dict[str, Any] = {}
        a = self.to_dict()
        b = other.to_dict()
        for section in a:
            sa = a[section]
            sb = b.get(section)
            if not isinstance(sa, dict) or not isinstance(sb, dict):
                if sa != sb:
                    diffs[section] = (sa, sb)
                continue
            sec: Dict[str, Any] = {}
            for key in set(sa) | set(sb):
                va, vb = sa.get(key), sb.get(key)
                if va != vb:
                    sec[key] = (va, vb)
            if sec:
                diffs[section] = sec
        return diffs

    def summary(self) -> str:
        """Return a compact human-readable summary of the active config.

        Used by the deploy script and audit orchestrator to log the runtime
        configuration at startup. Never raises — even on a partly-broken
        config it returns its best effort.
        """
        try:
            active = self.enabled_modules()
        except Exception:
            active = []
        try:
            renderers = self.active_renderers()
        except Exception:
            renderers = []
        lines = [
            "=" * 64,
            "  CineInfini — active configuration",
            "=" * 64,
            f"  device           : {self.device.get('gpu_device')} "
            f"(effective={self.effective_device()})",
            f"  dtype            : {self.device.get('torch_dtype')}",
            f"  max_duration_s   : {self.processing.get('max_duration_s')}",
            f"  n_frames_per_shot: {self.processing.get('n_frames_per_shot')}",
            f"  num_workers      : {self.processing.get('num_workers')}",
            f"  embedder         : {self.processing.get('embedder')}",
            f"  semantic_scorer  : {self.processing.get('semantic_scorer')}",
            f"  reports_dir      : {self.resolve_path('reports_dir')}",
            f"  models_dir       : {self.resolve_path('models_dir')}",
            f"  active modules   : {', '.join(active) if active else '(none)'}",
            f"  active renderers : {', '.join(renderers) if renderers else '(none)'}",
            "=" * 64,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level merge  (NEW v0.4.8.8)
# ---------------------------------------------------------------------------
def merge_configs(base: Config, override: Config) -> Config:
    """Deep-merge two :class:`Config` instances; *override* wins on conflicts.

    Returns a brand-new :class:`Config` — neither *base* nor *override* is
    mutated. Sub-dicts are merged recursively; lists are replaced (matching
    standard YAML override semantics).

    Typical use case::

        prod = load_config("cfg/config.yaml")
        exp  = load_config("cfg/profiles/postproduction.yaml")
        cfg  = merge_configs(prod, exp)
    """
    merged = _deep_merge(base.to_dict(), override.to_dict())
    return Config.from_dict(merged)


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the active config singleton, loading on first call.

    Fixes a bug in v0.4.8.1.0 where the result of ``_default_config()``
    was not stored, so every call paid the file-system lookup cost.
    """
    global _config
    if _config is None:
        _config = _default_config()
    return _config


def set_config(cfg: Config) -> None:
    global _config
    _config = cfg


def reset_config() -> None:
    global _config
    _config = None


def default_config() -> Config:
    return Config()


def test_config() -> Config:
    """Fast, isolated configuration for pytest (everything under /tmp)."""
    tmp = Path(tempfile.mkdtemp(prefix="cineinfini_test_"))
    cfg = Config()
    cfg.paths.update({
        "reports_dir": str(tmp / "reports"),
        "benchmark_dir": str(tmp / "benchmark"),
        "test_videos_dir": str(tmp / "videos"),
        "cache_dir": str(tmp / "cache"),
        "logs_dir": str(tmp / "logs"),
        "temp_dir": str(tmp / "temp"),
        "output_root": str(tmp / "output"),
    })
    cfg.processing.update({
        "max_duration_s": 10, "n_frames_per_shot": 8, "num_workers": 2,
        "parallel_shots": False, "narrative_coherence": False,
        "benchmark_mode": False,
    })
    for n in cfg.modules:
        if n not in {"motion_coherence", "identity_consistency", "semantic_consistency"}:
            cfg.modules[n]["enabled"] = False
    cfg.reporting.update({
        "active_renderers": ["json"], "figure_format": "png",
        "figure_dpi": 72, "interactive": False, "generate_plots": False,
    })
    cfg.logging.update({"level": "WARNING", "file_enabled": False})
    cfg.device["gpu_device"] = "cpu"
    cfg.device["use_amp"] = False
    return cfg


def load_config(path: "str | Path") -> Config:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return Config.from_dict(data)


def save_config(cfg: Config, path: "str | Path") -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.dump(cfg.to_dict(), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _default_config() -> Config:
    cand = [
        Path(os.environ.get("CINEINFINI_CONFIG", "")).expanduser()
        if os.environ.get("CINEINFINI_CONFIG") else None,
        Path.home() / ".cineinfini" / "config.yaml",
        Path.cwd() / "cfg" / "config.yaml",
    ]
    for c in cand:
        if c and c.exists():
            try:
                return load_config(c)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[cineinfini.config] Warning: could not load {c}: {e}",
                    file=sys.stderr,
                )
    return default_config()


def compat_models_dir() -> Path:
    return get_config().models_dir()


def compat_reports_dir() -> Path:
    return get_config().reports_dir()
