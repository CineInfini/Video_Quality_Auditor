"""Test the datasets section of Config + helpers.

v0.4.8.9: BVI-VFI was removed (gated registration impedes
reproducibility). Tests now exercise T2VQA-DB and VideoFeedback as the
calibration datasets, and KoNViD-1k as the gated counterpoint to
BVI-HFR's auto-downloadable example.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cineinfini.core.config import default_config, Config


def test_datasets_section_present():
    cfg = default_config()
    assert "t2vqa_db" in cfg.datasets
    assert "videofeedback" in cfg.datasets
    assert "vbench_eval" in cfg.datasets
    assert "bvi_vfi" not in cfg.datasets


def test_bvi_vfi_definitively_absent():
    cfg = default_config()
    assert "bvi_vfi" not in cfg.datasets
    assert cfg.dataset_dir("bvi_vfi") is None


def test_t2vqa_db_has_registration_url():
    cfg = default_config()
    t = cfg.datasets["t2vqa_db"]
    assert t["homepage"].startswith("https://github.com/QMME/T2VQA")
    assert "Research" in t["license"] or "research" in t["license"]
    assert t["purpose"].startswith("calibration")


def test_videofeedback_uses_huggingface():
    cfg = default_config()
    v = cfg.datasets["videofeedback"]
    assert v["auto_downloadable"] is True
    assert v.get("fetch_method") == "huggingface_datasets"


def test_dataset_dir_resolves_under_datasets_dir(tmp_path):
    cfg = default_config()
    cfg.paths["datasets_dir"] = str(tmp_path / "ds")
    target = cfg.dataset_dir("t2vqa_db")
    assert target is not None
    assert str(tmp_path / "ds") in str(target)
    assert target.name == cfg.datasets["t2vqa_db"]["name"]


def test_dataset_dir_unknown_key_returns_none():
    cfg = default_config()
    assert cfg.dataset_dir("nonexistent") is None


def test_dataset_present_false_when_missing(tmp_path):
    cfg = default_config()
    cfg.paths["datasets_dir"] = str(tmp_path / "empty")
    assert cfg.dataset_present("t2vqa_db") is False


def test_dataset_present_true_when_files_exist(tmp_path):
    cfg = default_config()
    cfg.paths["datasets_dir"] = str(tmp_path)
    target = cfg.dataset_dir("t2vqa_db")
    target.mkdir(parents=True)
    (target / "dummy.mp4").write_bytes(b"x")
    assert cfg.dataset_present("t2vqa_db") is True


def test_datasets_dir_resolves_path():
    cfg = default_config()
    p = cfg.datasets_dir()
    assert isinstance(p, Path)
    assert "~" not in str(p)


def test_datasets_section_roundtrips_via_yaml(tmp_path):
    from cineinfini.core.config import save_config, load_config
    cfg = default_config()
    cfg.datasets["t2vqa_db"]["expected_spearman_target"] = 0.99
    out = tmp_path / "cfg.yaml"
    save_config(cfg, out)
    loaded = load_config(out)
    assert loaded.datasets["t2vqa_db"]["expected_spearman_target"] == 0.99


def test_required_datasets_default_to_false():
    cfg = default_config()
    for key, entry in cfg.datasets.items():
        assert entry.get("required", False) is False


def test_bvi_hfr_has_direct_url():
    cfg = default_config()
    assert "bvi_hfr" in cfg.datasets
    bvi_hfr = cfg.datasets["bvi_hfr"]
    assert bvi_hfr["url"] == "https://data.bris.ac.uk/datasets/tar/k8bfn0qsj9fs1rwnc2x75z6t7.zip"
    assert bvi_hfr["auto_downloadable"] is True
    assert bvi_hfr["format"] == "zip"
    assert "*.mp4" in bvi_hfr["partial_patterns"]


def test_lfw_has_direct_url():
    cfg = default_config()
    assert "lfw" in cfg.datasets
    assert cfg.datasets["lfw"]["url"].startswith("http")
    assert cfg.datasets["lfw"]["auto_downloadable"] is True


def test_optional_models_section():
    cfg = default_config()
    assert hasattr(cfg, "optional_models")
    assert "dover" in cfg.optional_models
    assert "dover_mobile" in cfg.optional_models
    assert cfg.optional_models["dover"]["url"].startswith(
        "https://github.com/QualityAssessment/DOVER"
    )


def test_auto_downloadable_split():
    cfg = default_config()
    auto = [k for k, v in cfg.datasets.items() if v.get("auto_downloadable")]
    gated = [k for k, v in cfg.datasets.items() if not v.get("auto_downloadable")]
    assert len(auto) >= 2
    assert len(gated) >= 1
    assert "bvi_hfr" in auto
    assert "t2vqa_db" in gated


def test_konvid_1k_has_labels_url():
    """v0.4.9.0: KoNViD-1k now has a public labels CSV URL on cnn-tlvqm GitHub."""
    cfg = default_config()
    konvid = cfg.datasets["konvid_1k"]
    assert konvid["labels_url"].startswith("https://github.com/jarikorhonen/cnn-tlvqm")
    assert konvid["labels_filename"] == "KoNViD_mos_fr.csv"
    assert konvid["labels_auto_downloadable"] is True
    # Videos themselves remain gated
    assert konvid["auto_downloadable"] is False
    assert konvid["videos_url"] is None
