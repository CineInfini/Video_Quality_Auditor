"""Typed dataclasses for CineInfini outputs (v0.5.0 preview).

These types describe the shape that every audit module, exporter, and
renderer agrees on. They're **structurally compatible** with the
existing dict-based outputs — every field corresponds 1:1 to a key in
the legacy dict — so adopting these types is incremental:

  - Modules can return either a `dict` or a `ModuleResult`; the
    orchestrator's `_merge_module_result` accepts both.
  - Existing tests / renderers that read from dicts continue to work
    unchanged.
  - New code can use `AuditResult.from_dict(audit_data)` to get a
    typed view without breaking anything.

Adoption path for v0.5.0:
  1. (this release) Define the types and provide `from_dict` /
     `to_dict` round-trip.
  2. (next minor) Switch one renderer at a time to consume the typed
     view internally.
  3. (v0.5.0) Make `run_audit` return `AuditResult` directly with a
     deprecation shim that yields a dict for legacy callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-metric result
# ---------------------------------------------------------------------------
@dataclass
class MetricResult:
    """A single named metric with optional confidence and metadata."""
    name: str
    value: Optional[float]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    @classmethod
    def from_value(cls, name: str, value: Any, version: str = "1.0") -> "MetricResult":
        if value is None:
            return cls(name=name, value=None, confidence=0.0, version=version)
        return cls(name=name, value=float(value), version=version)


# ---------------------------------------------------------------------------
# Per-module result
# ---------------------------------------------------------------------------
@dataclass
class ModuleResult:
    """The structured form of what every audit module returns.

    The legacy dict shape is:
        {
          "module": "<id>", "version": "<v>", "available": True/False,
          "per_shot": {1: {...}, 2: {...}, ...},
          "<custom-summary-fields>": ...
        }

    This dataclass is a 1:1 mirror plus a `to_dict()` for backward
    compatibility.
    """
    module: str
    version: str = "1.0"
    available: bool = True
    reason: Optional[str] = None
    per_shot: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "module": self.module,
            "version": self.version,
            "available": self.available,
        }
        if self.reason:
            out["reason"] = self.reason
        if self.per_shot:
            out["per_shot"] = self.per_shot
        out.update(self.summary)
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleResult":
        per_shot = d.get("per_shot") or {}
        summary = {
            k: v for k, v in d.items()
            if k not in {"module", "version", "available", "reason", "per_shot"}
        }
        return cls(
            module=d.get("module", "unknown"),
            version=str(d.get("version", "1.0")),
            available=bool(d.get("available", True)),
            reason=d.get("reason"),
            per_shot={int(k): v for k, v in per_shot.items()},
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Per-shot verdict
# ---------------------------------------------------------------------------
@dataclass
class ShotResult:
    shot_id: int
    metrics: Dict[str, float] = field(default_factory=dict)
    composite: Optional[float] = None
    verdict: str = "REVIEW"           # ACCEPT / REVIEW / REJECT / BLOCKED
    failed_gates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.metrics)
        if self.composite is not None:
            out["composite"] = self.composite
        out["verdict"] = self.verdict
        out["failed_gates"] = self.failed_gates
        return out

    @classmethod
    def from_dict(cls, shot_id: int, d: Dict[str, Any]) -> "ShotResult":
        composite = d.get("composite")
        verdict = str(d.get("verdict", "REVIEW"))
        failed = list(d.get("failed_gates") or [])
        metrics = {
            k: v for k, v in d.items()
            if k not in {"composite", "verdict", "failed_gates"}
            and isinstance(v, (int, float))
        }
        return cls(shot_id=shot_id, metrics=metrics,
                   composite=composite, verdict=verdict, failed_gates=failed)


# ---------------------------------------------------------------------------
# Whole-audit result
# ---------------------------------------------------------------------------
@dataclass
class AuditResult:
    video_name: str
    version: str
    duration_s: float
    fps: float
    n_shots: int
    shots: Dict[int, ShotResult] = field(default_factory=dict)
    modules: Dict[str, ModuleResult] = field(default_factory=dict)
    composite_score: Optional[float] = None
    videoscore_axes: Dict[str, Optional[float]] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    rendered: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "version": self.version,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "n_shots": self.n_shots,
            "gates": {sid: shot.to_dict() for sid, shot in self.shots.items()},
            "modules": {mid: m.to_dict() for mid, m in self.modules.items()},
            "composite_score": self.composite_score,
            "videoscore_axes": self.videoscore_axes,
            "timing": self.timing,
            "rendered": self.rendered,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditResult":
        shots = {
            int(sid): ShotResult.from_dict(int(sid), gate)
            for sid, gate in (d.get("gates") or {}).items()
        }
        modules = {
            mid: ModuleResult.from_dict(mod)
            for mid, mod in (d.get("modules") or {}).items()
        }
        return cls(
            video_name=str(d.get("video_name", "unknown")),
            version=str(d.get("version", "unknown")),
            duration_s=float(d.get("duration_s", 0.0)),
            fps=float(d.get("fps", 0.0)),
            n_shots=int(d.get("n_shots", 0)),
            shots=shots,
            modules=modules,
            composite_score=d.get("composite_score"),
            videoscore_axes=d.get("videoscore_axes") or {},
            timing=d.get("timing") or {},
            rendered=d.get("rendered") or {},
        )


__all__ = ["MetricResult", "ModuleResult", "ShotResult", "AuditResult"]
