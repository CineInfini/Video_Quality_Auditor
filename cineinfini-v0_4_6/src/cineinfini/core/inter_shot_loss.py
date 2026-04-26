"""
inter_shot_loss.py
==================
Loss de cohérence inter-plans composite avec pondération asymétrique.

Principe : la "cohérence" entre deux plans n'est pas une chose unique. Elle
se décompose en composantes qui doivent se comporter différemment :

  - IDENTITY    : visages du MÊME personnage doivent rester proches
                  (tolerance ~0.3 cos-dist avec ArcFace)
  - STRUCTURE   : composition globale doit varier modérément
                  (coupe = changement de plan, donc on S'ATTEND à ce que
                  ça change — une loss trop stricte serait incohérente)
  - STYLE       : palette colorimétrique et tonalité doivent être
                  cohérentes pour un même décor, variables entre décors
  - SEMANTIC    : contenu sémantique (CLIP) doit suivre le script

La loss composite retourne :
  - un score scalaire pour backpropagation
  - un dict détaillé pour diagnostic (quel composant contribue le plus)
  - une interprétation automatique (recommandations)

Usage typique :
  loss = InterShotCoherenceLoss(
      embedder_identity=ArcFaceEmbedder(),
      weights={"identity": 3.0, "structure": 0.5, "style": 1.0, "semantic": 1.5},
      asymmetry=True,  # identity est pénalisée plus fortement que style
  )
  value = loss.compute(shot_A_frames, shot_B_frames,
                       same_character=True, same_location=False)

Cette loss peut être utilisée :
  1. Comme quality gate post-rendu (agrégée dans Phase4)
  2. Comme signal d'entraînement pour fine-tuner un T2V (si torch.Tensor)
  3. Comme critère de re-sampling (rendre N variantes, garder meilleure)

Fondamentalement, il n'y a pas de "breakthrough" mathématique ici — c'est une
agrégation asymétrique calibrée de métriques connues. Ce qui la rend utile en
pratique, c'est :
  - elle expose CHAQUE composante séparément (pas une boîte noire)
  - elle permet de déclarer des priors (same_character, same_location)
  - elle produit des recommandations interprétables

Pour publier en conférence A, il faudrait compléter par :
  - un dataset d'annotations human-labeled (quels plans "cohérents")
  - un fit des poids par régression logistique sur ce dataset
  - une évaluation PLCC/SRCC vs jugements humains
  Rien de cela n'est fait ici. Mais la structure est prête.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Types et interfaces
# ---------------------------------------------------------------------------

@dataclass
class InterShotLossResult:
    """Résultat d'une comparaison entre deux plans."""
    identity: float | None        # distance cos ArcFace, None si pas de visage
    structure: float              # distance L2 sur histogramme spatial
    style: float                  # distance sur moments couleur
    semantic: float               # distance cosine sur features CNN
    weighted_total: float         # somme pondérée
    components: dict              # poids et valeurs détaillées
    recommendations: list[str] = field(default_factory=list)

    def is_coherent(self, threshold: float = 0.5) -> bool:
        return self.weighted_total < threshold


class _IdentityEmbedder(Protocol):
    def embed(self, face_crop: np.ndarray) -> np.ndarray: ...


class _FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list: ...


# ---------------------------------------------------------------------------
# Extracteurs de features par composante
# ---------------------------------------------------------------------------

def extract_identity_embedding(
    frames: list[np.ndarray],
    detector: _FaceDetector,
    embedder: _IdentityEmbedder,
    max_frames: int = 5,
) -> np.ndarray | None:
    """Extrait un embedding d'identité agrégé sur plusieurs frames.

    Stratégie : détecte le plus grand visage sur ≤5 frames équidistantes,
    moyenne les embeddings, L2-normalise. Retourne None si aucun visage
    n'est détecté (cas fréquent pour plans de décor sans personnage).
    """
    if not frames:
        return None
    n = min(len(frames), max_frames)
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    embs = []
    for i in idxs:
        f = frames[i]
        boxes = detector.detect(f)
        if not boxes:
            continue
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        crop = f[y:y + h, x:x + w]
        if crop.size > 0:
            embs.append(embedder.embed(crop))
    if not embs:
        return None
    mean = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 0 else mean


def extract_structure_histogram(
    frames: list[np.ndarray],
    grid: int = 4,
) -> np.ndarray:
    """Histogramme spatial : distribution d'intensité par cellule de grille.

    Plus rapide que CNN features et capte la composition globale. Invariant
    aux détails fins, sensible à la structure grossière (ciel, sol,
    premier plan, arrière-plan).
    """
    avg_frame = np.mean([f.astype(np.float32) for f in frames], axis=0)
    gray = cv2.cvtColor(avg_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY) \
        if avg_frame.ndim == 3 else avg_frame.astype(np.uint8)
    H, W = gray.shape
    feat = []
    for gy in range(grid):
        for gx in range(grid):
            cell = gray[gy * H // grid:(gy + 1) * H // grid,
                        gx * W // grid:(gx + 1) * W // grid]
            hist, _ = np.histogram(cell, bins=16, range=(0, 256), density=True)
            feat.append(hist)
    vec = np.concatenate(feat).astype(np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def extract_style_moments(frames: list[np.ndarray]) -> np.ndarray:
    """Moments couleur LAB : moyenne + stddev par canal = 6 features.

    Capte la "tonalité" d'un plan (chaud/froid, saturé/désaturé).
    """
    # Pour chaque frame, calculer (mean, std) par canal LAB → 6 valeurs
    # Puis moyenner sur les frames → 6 valeurs finales.
    per_frame_feats = []
    for f in frames[::max(1, len(frames) // 8)]:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        vec = []
        for c in range(3):
            ch = lab[..., c]
            vec.append(ch.mean())
            vec.append(ch.std())
        per_frame_feats.append(vec)
    feats = np.mean(per_frame_feats, axis=0)
    return feats.astype(np.float32)


def extract_semantic_signature(
    frames: list[np.ndarray],
    n_frames: int = 4,
) -> np.ndarray:
    """Signature sémantique légère via DCT basses fréquences.

    Pour une vraie application, remplacer par CLIP embeddings (import clip).
    Ici on utilise une approximation DCT + stats qui capte le contenu général
    sans charger un modèle lourd.
    """
    idxs = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    feats = []
    for i in idxs:
        f = frames[i]
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        small = cv2.resize(gray.astype(np.float32), (64, 64))
        dct = cv2.dct(small)
        # 8x8 basse fréquence = 64 coefficients
        low = dct[:8, :8].flatten()
        n = np.linalg.norm(low)
        feats.append(low / n if n > 0 else low)
    mean = np.mean(feats, axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 0 else mean


# ---------------------------------------------------------------------------
# Loss composite
# ---------------------------------------------------------------------------

class InterShotCoherenceLoss:
    """Loss asymétrique combinant 4 composantes.

    Parameters
    ----------
    embedder_identity : _IdentityEmbedder
        Embedder visage (ArcFace recommandé, ClassicV2 en fallback)
    detector : _FaceDetector
        Détecteur visage (YuNet recommandé)
    weights : dict
        Poids par composante. Defaults :
          identity=3.0, structure=0.5, style=1.0, semantic=1.5
        L'asymétrie est dans les valeurs : identity pèse 6× plus que
        structure, reflétant que c'est plus critique narrativement.
    asymmetry : bool
        Si True (défaut), on applique une pénalité quadratique sur identity
        (petites différences acceptées, grosses pénalisées fortement). Si
        False, pénalité linéaire pour tous les composants.
    """

    def __init__(
        self,
        embedder_identity: _IdentityEmbedder | None = None,
        detector: _FaceDetector | None = None,
        weights: dict | None = None,
        asymmetry: bool = True,
    ):
        self.embedder_identity = embedder_identity
        self.detector = detector
        self.weights = weights or {
            "identity": 3.0,
            "structure": 0.5,
            "style": 1.0,
            "semantic": 1.5,
        }
        self.asymmetry = asymmetry

    @staticmethod
    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b) /
                           (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    @staticmethod
    def _l2_normalized(a: np.ndarray, b: np.ndarray) -> float:
        """L2 normalisée par la norme de la concaténation."""
        d = np.linalg.norm(a - b)
        scale = (np.linalg.norm(a) + np.linalg.norm(b)) / 2 + 1e-9
        return float(d / scale / np.sqrt(len(a)))

    def compute(
        self,
        shot_a_frames: list[np.ndarray],
        shot_b_frames: list[np.ndarray],
        *,
        same_character: bool | None = None,
        same_location: bool | None = None,
    ) -> InterShotLossResult:
        """Calcule la loss entre deux plans.

        Les flags `same_character` et `same_location` ajustent les poids :
        - same_character=True → identity DOIT être très proche (poids ×2)
        - same_character=False → identity DOIT être différent (poids=0, on
          ne pénalise pas une différence, voire on récompenserait si
          asymmetry=True et différence>seuil)
        - same_location=True → style et structure attendus proches
        - same_location=False → style libre, structure libre
        """
        w = dict(self.weights)  # copy

        # Ajustements contextuels
        if same_character is True:
            w["identity"] *= 2.0
        elif same_character is False:
            w["identity"] = 0.0
        if same_location is False:
            w["style"] *= 0.3
            w["structure"] *= 0.3

        recs = []
        components = {}

        # 1. IDENTITY (si demandé et visages présents)
        id_dist = None
        if w["identity"] > 0 and self.embedder_identity and self.detector:
            emb_a = extract_identity_embedding(shot_a_frames, self.detector,
                                                self.embedder_identity)
            emb_b = extract_identity_embedding(shot_b_frames, self.detector,
                                                self.embedder_identity)
            if emb_a is not None and emb_b is not None:
                id_dist = self._cos_dist(emb_a, emb_b)
                # Asymétrie : pénalité quadratique au-delà de 0.3
                if self.asymmetry and id_dist > 0.3:
                    id_contribution = w["identity"] * (id_dist ** 2) * 2
                else:
                    id_contribution = w["identity"] * id_dist
                components["identity"] = {
                    "distance": id_dist,
                    "weight": w["identity"],
                    "contribution": id_contribution,
                }
                if same_character and id_dist > 0.5:
                    recs.append(
                        f"⚠ Identity drift élevée ({id_dist:.3f}) entre "
                        f"plans du MÊME personnage. Vérifier le rendu "
                        f"(risque de changement d'acteur perceptible)."
                    )
                elif same_character is False and id_dist < 0.3:
                    recs.append(
                        f"⚠ Les personnages devraient être différents mais "
                        f"les embeddings sont proches ({id_dist:.3f}). "
                        f"Risque de confusion narrative."
                    )

        # 2. STRUCTURE
        hist_a = extract_structure_histogram(shot_a_frames)
        hist_b = extract_structure_histogram(shot_b_frames)
        struct_dist = self._cos_dist(hist_a, hist_b)
        components["structure"] = {
            "distance": struct_dist,
            "weight": w["structure"],
            "contribution": w["structure"] * struct_dist,
        }

        # 3. STYLE
        style_a = extract_style_moments(shot_a_frames)
        style_b = extract_style_moments(shot_b_frames)
        # L2 normalisée sur les 6 moments (échelles LAB différentes)
        style_dist = float(np.linalg.norm((style_a - style_b) /
                                           (np.abs(style_a) + np.abs(style_b) + 1)))
        style_dist = min(style_dist, 2.0)  # cap
        components["style"] = {
            "distance": style_dist,
            "weight": w["style"],
            "contribution": w["style"] * style_dist,
        }
        if same_location and style_dist > 1.0:
            recs.append(
                f"⚠ Style (palette LAB) très différent ({style_dist:.2f}) "
                f"entre plans du même lieu. Vérifier color grading / heure."
            )

        # 4. SEMANTIC
        sem_a = extract_semantic_signature(shot_a_frames)
        sem_b = extract_semantic_signature(shot_b_frames)
        sem_dist = self._cos_dist(sem_a, sem_b)
        components["semantic"] = {
            "distance": sem_dist,
            "weight": w["semantic"],
            "contribution": w["semantic"] * sem_dist,
        }

        # Total pondéré
        total = sum(c["contribution"] for c in components.values())

        # Recommandations globales
        if id_dist is None and (same_character is True):
            recs.append(
                "ℹ Aucun visage détecté dans au moins un des plans — "
                "identité ne peut être vérifiée."
            )
        if total > 1.5:
            recs.append(
                f"⚠ Cohérence globale faible ({total:.2f}). Vérifier le "
                f"composant dominant : " +
                max(components.items(),
                    key=lambda kv: kv[1]["contribution"])[0]
            )

        return InterShotLossResult(
            identity=id_dist,
            structure=struct_dist,
            style=style_dist,
            semantic=sem_dist,
            weighted_total=total,
            components=components,
            recommendations=recs,
        )


# ---------------------------------------------------------------------------
# Démo
# ---------------------------------------------------------------------------

def demo():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, "/home/claude")

    from .face_detection import CascadeFaceDetector
    from .face_embedding import get_embedder

    detector = CascadeFaceDetector()
    embedder = get_embedder("auto")
    loss = InterShotCoherenceLoss(
        embedder_identity=embedder,
        detector=detector,
    )

    # Synthétise 2 paires de plans :
    #   Cas A : même décor, même personnage → devrait être cohérent
    #   Cas B : décors différents, personnages différents → incohérent
    rng = np.random.default_rng(42)

    def make_shot(bg_color, face_color, n=6, drift=0):
        frames = []
        for i in range(n):
            f = np.zeros((270, 480, 3), dtype=np.float32)
            f[:] = bg_color
            # Face
            cx = 240 + drift * i
            cy = 135
            for y in range(cy - 45, cy + 45):
                for x in range(cx - 35, cx + 35):
                    dx = (x - cx) / 35
                    dy = (y - cy) / 45
                    if dx * dx + dy * dy <= 1:
                        shade = max(0.6, 1 - 0.3 * (dx * 0.4 + dy * 0.3))
                        f[y, x] = np.array(face_color) * shade
            # Yeux
            for ex in (cx - 12, cx + 12):
                cv2.circle(f, (ex, cy - 10), 3, (50, 40, 30), -1)
            # Bouche
            cv2.ellipse(f, (cx, cy + 18), (8, 3), 0, 0, 180, (80, 70, 120), -1)
            f += rng.normal(0, 2, f.shape)
            frames.append(np.clip(f, 0, 255).astype(np.uint8))
        return frames

    # Cas A : même personnage, même décor
    print("=" * 70)
    print("Cas A : même personnage, même décor (devrait être cohérent)")
    print("=" * 70)
    shot_a1 = make_shot((140, 150, 130), (165, 185, 220), drift=2)
    shot_a2 = make_shot((140, 150, 130), (165, 185, 220), drift=3)
    result_A = loss.compute(shot_a1, shot_a2,
                             same_character=True, same_location=True)
    print(f"  identity    = {result_A.identity}")
    print(f"  structure   = {result_A.structure:.4f}")
    print(f"  style       = {result_A.style:.4f}")
    print(f"  semantic    = {result_A.semantic:.4f}")
    print(f"  TOTAL       = {result_A.weighted_total:.4f}")
    print(f"  cohérent?   = {result_A.is_coherent()}")
    for r in result_A.recommendations:
        print(f"  → {r}")

    # Cas B : personnages différents, décors différents
    print()
    print("=" * 70)
    print("Cas B : personnages différents, décors différents")
    print("=" * 70)
    shot_b1 = make_shot((140, 150, 130), (165, 185, 220))
    shot_b2 = make_shot((80, 60, 40), (130, 145, 180))
    result_B = loss.compute(shot_b1, shot_b2,
                             same_character=False, same_location=False)
    print(f"  identity    = {result_B.identity}")
    print(f"  structure   = {result_B.structure:.4f}")
    print(f"  style       = {result_B.style:.4f}")
    print(f"  semantic    = {result_B.semantic:.4f}")
    print(f"  TOTAL       = {result_B.weighted_total:.4f}")

    # Cas C : même personnage attendu mais en fait différent (erreur de rendu)
    print()
    print("=" * 70)
    print("Cas C : same_character=True mais visages différents (erreur)")
    print("=" * 70)
    shot_c1 = make_shot((140, 150, 130), (165, 185, 220))
    shot_c2 = make_shot((140, 150, 130), (130, 145, 180))  # visage différent
    result_C = loss.compute(shot_c1, shot_c2,
                             same_character=True, same_location=True)
    print(f"  identity    = {result_C.identity}")
    print(f"  structure   = {result_C.structure:.4f}")
    print(f"  TOTAL       = {result_C.weighted_total:.4f}")
    for r in result_C.recommendations:
        print(f"  → {r}")


if __name__ == "__main__":
    demo()
