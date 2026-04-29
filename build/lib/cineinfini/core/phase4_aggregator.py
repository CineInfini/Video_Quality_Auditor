"""Phase-4 aggregator: combines per-shot metrics into ACCEPT/REVIEW/REJECT."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass
class GateThresholds:
    motion: float = 25.0
    ssim3d: float = 0.45
    flicker: float = 0.10
    identity_drift: float = 0.60
    ssim_long_range: float = 0.45
    clip_temp: float = 0.25
    flicker_hf: float = 0.01
    background_ssim: float = 0.55
    accept_min: float = 0.6
    review_min: float = 0.3

    @classmethod
    def from_config(cls, cfg_thresholds: Mapping[str, float]) -> "GateThresholds":
        return cls(
            motion=float(cfg_thresholds.get("motion", 25.0)),
            ssim3d=float(cfg_thresholds.get("ssim3d", 0.45)),
            flicker=float(cfg_thresholds.get("flicker", 0.10)),
            identity_drift=float(cfg_thresholds.get("identity_drift", 0.60)),
            ssim_long_range=float(cfg_thresholds.get("ssim_long_range", 0.45)),
            clip_temp=float(cfg_thresholds.get("clip_temp", 0.25)),
            flicker_hf=float(cfg_thresholds.get("flicker_hf", 0.01)),
            background_ssim=float(cfg_thresholds.get("background_ssim", 0.55)),
            accept_min=float(cfg_thresholds.get("accept_min", 0.6)),
            review_min=float(cfg_thresholds.get("review_min", 0.3)),
        )


@dataclass
class ShotVerdict:
    shot_id: int
    composite: float
    verdict: str
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": int(self.shot_id),
            "composite": float(self.composite),
            "verdict": self.verdict,
            "gates_passed": list(self.gates_passed),
            "gates_failed": list(self.gates_failed),
            "metrics": dict(self.metrics),
        }


_GATES: List[Tuple[str, str, str]] = [
    ("motion_peak_div", "<=", "motion"),
    ("ssim3d_self", ">=", "ssim3d"),
    ("flicker", "<=", "flicker"),
    ("identity_intra", "<=", "identity_drift"),
    ("ssim_long_range", ">=", "ssim_long_range"),
    ("clip_temp_consistency", ">=", "clip_temp"),
    ("flicker_hf", "<=", "flicker_hf"),
    ("background_ssim", ">=", "background_ssim"),
]


def _gate_pass(value, op: str, threshold: float) -> Optional[bool]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v <= threshold if op == "<=" else v >= threshold


def aggregate_shot_verdict(
    shot_id: int,
    metrics: Mapping[str, Optional[float]],
    thresholds: GateThresholds,
) -> ShotVerdict:
    passed: List[str] = []
    failed: List[str] = []
    evaluated: List[bool] = []
    for metric_key, op, attr in _GATES:
        if metric_key not in metrics:
            continue
        thr = float(getattr(thresholds, attr))
        result = _gate_pass(metrics.get(metric_key), op, thr)
        if result is True:
            passed.append(metric_key)
            evaluated.append(True)
        elif result is False:
            failed.append(metric_key)
            evaluated.append(False)
    composite = float(sum(evaluated)) / float(len(evaluated)) if evaluated else 0.0
    if composite >= thresholds.accept_min:
        verdict = "ACCEPT"
    elif composite >= thresholds.review_min:
        verdict = "REVIEW"
    else:
        verdict = "REJECT"
    return ShotVerdict(
        shot_id=int(shot_id), composite=composite, verdict=verdict,
        gates_passed=passed, gates_failed=failed, metrics=dict(metrics),
    )


def build_phase4_report(
    shot_metrics: Mapping[int, Mapping[str, Optional[float]]],
    thresholds: Optional[GateThresholds] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    verdicts: Dict[int, Dict[str, Any]] = {}
    counts = {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0}
    composites: List[float] = []
    for sid, metrics in shot_metrics.items():
        v = aggregate_shot_verdict(int(sid), metrics, thresholds)
        verdicts[int(sid)] = v.to_dict()
        counts[v.verdict] += 1
        composites.append(v.composite)
    return {
        "thresholds": {
            "motion": thresholds.motion, "ssim3d": thresholds.ssim3d,
            "flicker": thresholds.flicker, "identity_drift": thresholds.identity_drift,
            "ssim_long_range": thresholds.ssim_long_range,
            "clip_temp": thresholds.clip_temp, "flicker_hf": thresholds.flicker_hf,
            "background_ssim": thresholds.background_ssim,
            "accept_min": thresholds.accept_min, "review_min": thresholds.review_min,
        },
        "verdicts": verdicts,
        "summary": {
            "n_shots": len(verdicts),
            "n_accept": counts["ACCEPT"],
            "n_review": counts["REVIEW"],
            "n_reject": counts["REJECT"],
            "mean_composite": (sum(composites) / len(composites)) if composites else 0.0,
        },
    }


__all__ = ["GateThresholds", "ShotVerdict", "aggregate_shot_verdict", "build_phase4_report"]
