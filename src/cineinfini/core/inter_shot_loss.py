"""Inter-shot coherence loss between successive shots."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class InterShotLossResult:
    a: int
    b: int
    visual_distance: Optional[float] = None
    semantic_distance: Optional[float] = None
    identity_distance: Optional[float] = None
    composite_loss: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": int(self.a), "b": int(self.b),
            "visual_distance": self.visual_distance,
            "semantic_distance": self.semantic_distance,
            "identity_distance": self.identity_distance,
            "composite_loss": float(self.composite_loss),
            "weights": dict(self.weights),
        }


class InterShotCoherenceLoss:
    """Visual + semantic + identity distance between successive shots."""

    def __init__(self, visual_weight: float = 0.4,
                 semantic_weight: float = 0.4, identity_weight: float = 0.2):
        self.weights = {
            "visual": float(visual_weight),
            "semantic": float(semantic_weight),
            "identity": float(identity_weight),
        }

    @staticmethod
    def _cosine_distance(a, b) -> Optional[float]:
        if a is None or b is None:
            return None
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return None
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return None
        return 1.0 - float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _ssim_distance(s: Optional[float]) -> Optional[float]:
        if s is None:
            return None
        return float(max(0.0, 1.0 - float(s)))

    def evaluate_pair(self, a_id: int, b_id: int,
                      visual_ssim: Optional[float] = None,
                      semantic_a=None, semantic_b=None,
                      identity_a=None, identity_b=None) -> InterShotLossResult:
        vis = self._ssim_distance(visual_ssim)
        sem = self._cosine_distance(semantic_a, semantic_b)
        ide = self._cosine_distance(identity_a, identity_b)
        components = []
        if vis is not None:
            components.append((self.weights["visual"], vis))
        if sem is not None:
            components.append((self.weights["semantic"], sem))
        if ide is not None:
            components.append((self.weights["identity"], ide))
        if not components:
            composite = 0.0
        else:
            wsum = sum(w for w, _ in components)
            composite = (
                float(sum(w * d for w, d in components) / wsum)
                if wsum > 1e-9 else float(np.mean([d for _, d in components]))
            )
        return InterShotLossResult(
            a=a_id, b=b_id,
            visual_distance=vis, semantic_distance=sem, identity_distance=ide,
            composite_loss=composite, weights=dict(self.weights),
        )

    def evaluate_sequence(
        self, shot_ids: Sequence[int],
        visual_ssims: Optional[Mapping[int, float]] = None,
        semantic_embs: Optional[Mapping[int, Sequence[float]]] = None,
        identity_embs: Optional[Mapping[int, Sequence[float]]] = None,
    ) -> List[InterShotLossResult]:
        visual_ssims = visual_ssims or {}
        semantic_embs = semantic_embs or {}
        identity_embs = identity_embs or {}
        out: List[InterShotLossResult] = []
        for a, b in zip(shot_ids[:-1], shot_ids[1:]):
            out.append(self.evaluate_pair(
                a, b,
                visual_ssim=visual_ssims.get(b),
                semantic_a=semantic_embs.get(a), semantic_b=semantic_embs.get(b),
                identity_a=identity_embs.get(a), identity_b=identity_embs.get(b),
            ))
        return out

    def aggregate(self, results: Sequence[InterShotLossResult]) -> Dict[str, Any]:
        if not results:
            return {"n_pairs": 0, "mean_loss": 0.0, "max_loss": 0.0,
                    "min_loss": 0.0, "std_loss": 0.0}
        losses = [r.composite_loss for r in results]
        return {
            "n_pairs": len(results),
            "mean_loss": float(np.mean(losses)),
            "max_loss": float(np.max(losses)),
            "min_loss": float(np.min(losses)),
            "std_loss": float(np.std(losses)),
        }


__all__ = ["InterShotCoherenceLoss", "InterShotLossResult"]
