"""
phase4_aggregator.py
====================
Agrégation des trois quality gates (motion, volumetric, identity) en un
rapport Phase 4 consolidé avec verdict par plan.

Principe :
  - Chaque gate produit un score normalisé dans [0, 1] où 1 = OK, 0 = rejet
  - Score global = moyenne géométrique (pénalise les mauvais scores plus que
    la moyenne arithmétique)
  - Verdict : ACCEPT / REVIEW / REJECT selon des seuils configurables
  - Tous les seuils sont paramétrables (pas hardcodés)

Les seuils par défaut viennent des mesures empiriques réalisées dans ce PoC,
pas de spec externe. Chaque seuil doit être recalibré sur le dataset cible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration des seuils (paramétrable)
# ---------------------------------------------------------------------------

@dataclass
class GateThresholds:
    """Tous les seuils du rapport, paramétrables.

    Les defaults viennent de la calibration synthétique du PoC. En production :
    recalibrer sur un dataset annoté (BVI-VFI pour motion, ConsIDVid-Bench
    pour identity, etc.).
    """
    # Motion coherence (peak |div| du champ de mouvement)
    motion_accept_below: float = 2.0    # plan propre
    motion_reject_above: float = 6.0    # plan manifestement cassé

    # Volumetric stability (3D-SSIM)
    ssim3d_accept_above: float = 0.95
    ssim3d_reject_below: float = 0.80

    # Identity drift (cosine distance vs ref shot)
    identity_accept_below: float = 0.15
    identity_reject_above: float = 0.40

    # Score global minimum pour ACCEPT automatique
    global_accept_above: float = 0.75
    global_reject_below: float = 0.40


# ---------------------------------------------------------------------------
# Normalisation score par gate
# ---------------------------------------------------------------------------

def _linear_score(value: float, accept_at: float, reject_at: float) -> float:
    """Interpole linéairement entre accept (score=1) et reject (score=0).

    accept_at < reject_at : plus c'est bas, mieux c'est (motion, identity)
    accept_at > reject_at : plus c'est haut, mieux c'est (3D-SSIM)
    """
    if value is None or np.isnan(value):
        return float("nan")
    if accept_at < reject_at:
        # « bas = bon »
        if value <= accept_at:
            return 1.0
        if value >= reject_at:
            return 0.0
        return 1.0 - (value - accept_at) / (reject_at - accept_at)
    else:
        # « haut = bon »
        if value >= accept_at:
            return 1.0
        if value <= reject_at:
            return 0.0
        return (value - reject_at) / (accept_at - reject_at)


def normalize_motion(peak_div: float, th: GateThresholds) -> float:
    return _linear_score(peak_div, th.motion_accept_below, th.motion_reject_above)


def normalize_ssim3d(ssim3d: float, th: GateThresholds) -> float:
    return _linear_score(ssim3d, th.ssim3d_accept_above, th.ssim3d_reject_below)


def normalize_identity(drift: float, th: GateThresholds) -> float:
    return _linear_score(drift, th.identity_accept_below, th.identity_reject_above)


# ---------------------------------------------------------------------------
# Agrégation par plan
# ---------------------------------------------------------------------------

@dataclass
class ShotVerdict:
    shot_id: int
    motion_peak_div: float | None = None
    ssim3d: float | None = None
    identity_drift: float | None = None
    motion_score: float | None = None
    ssim3d_score: float | None = None
    identity_score: float | None = None
    global_score: float | None = None
    verdict: str = "NO_DATA"
    safety_status: str = "unknown"
    notes: list[str] = field(default_factory=list)


def _geometric_mean(scores: list[float]) -> float:
    """Moyenne géométrique ignorant les NaN. Retourne NaN si aucun score valide."""
    valid = [s for s in scores if s is not None and not np.isnan(s)]
    if not valid:
        return float("nan")
    # Borne inférieure à 1e-6 pour éviter log(0)
    logs = [np.log(max(s, 1e-6)) for s in valid]
    return float(np.exp(np.mean(logs)))


def aggregate_shot_verdict(
    shot_id: int,
    motion_peak_div: float | None,
    ssim3d: float | None,
    identity_drift: float | None,
    safety_status: str,
    thresholds: GateThresholds | None = None,
) -> ShotVerdict:
    th = thresholds or GateThresholds()

    v = ShotVerdict(
        shot_id=shot_id,
        motion_peak_div=motion_peak_div,
        ssim3d=ssim3d,
        identity_drift=identity_drift,
        safety_status=safety_status,
    )

    # Si le plan est bloqué par safety : verdict final = BLOCKED, pas de score
    if safety_status == "blocked_for_review":
        v.verdict = "BLOCKED_SAFETY"
        v.notes.append("Shot bloqué par safety gate, gates de qualité non évaluées.")
        return v

    v.motion_score = normalize_motion(motion_peak_div, th) if motion_peak_div is not None else None
    v.ssim3d_score = normalize_ssim3d(ssim3d, th) if ssim3d is not None else None
    v.identity_score = normalize_identity(identity_drift, th) if identity_drift is not None else None

    v.global_score = _geometric_mean([v.motion_score, v.ssim3d_score, v.identity_score])

    if np.isnan(v.global_score):
        v.verdict = "NO_DATA"
        v.notes.append("Aucun score de gate disponible.")
    elif v.global_score >= th.global_accept_above:
        v.verdict = "ACCEPT"
    elif v.global_score <= th.global_reject_below:
        v.verdict = "REJECT"
    else:
        v.verdict = "REVIEW"

    # Notes ciblées si un gate particulier est en cause
    if v.motion_score is not None and v.motion_score < 0.3:
        v.notes.append(f"Motion peak div élevé ({motion_peak_div:.3f}) — vérifier singularités.")
    if v.ssim3d_score is not None and v.ssim3d_score < 0.3:
        v.notes.append(f"3D-SSIM bas ({ssim3d:.3f}) — flicker ou instabilité de texture.")
    if v.identity_score is not None and v.identity_score < 0.3:
        v.notes.append(f"Dérive d'identité élevée ({identity_drift:.3f}) — personnage instable.")

    return v


# ---------------------------------------------------------------------------
# Construction du rapport markdown
# ---------------------------------------------------------------------------

def build_phase4_report(
    verdicts: list[ShotVerdict],
    backend_info: dict,
    thresholds: GateThresholds,
    extra_context: dict | None = None,
) -> str:
    """Construit le markdown du rapport Phase 4 à partir de verdicts par plan."""
    extra = extra_context or {}

    # Statistiques globales
    by_verdict: dict[str, list[ShotVerdict]] = {}
    for v in verdicts:
        by_verdict.setdefault(v.verdict, []).append(v)

    global_scores = [v.global_score for v in verdicts
                     if v.global_score is not None and not np.isnan(v.global_score)]
    mean_global = float(np.mean(global_scores)) if global_scores else float("nan")

    lines = [
        "# Phase 4 — Consolidated Quality Report",
        "",
        "**Generated by CineInfini orchestration PoC**",
        "",
        "## Disclaimer",
        "",
        "Ce rapport mesure les frames *effectivement produites par le backend de rendu*.",
        "Il ne mesure pas la qualité d'un long-métrage sur l'intrigue ou l'émotion ; il",
        "mesure la cohérence pixel-level et la stabilité d'identité telle qu'on peut la",
        "calculer objectivement avec PSNR_DIV, 3D-SSIM et un embedder d'identité.",
        "",
        "## Rendering backend",
        "",
    ]
    for k, val in backend_info.items():
        lines.append(f"- **{k}** : {val}")
    lines.append("")

    if extra:
        lines.append("## Context")
        lines.append("")
        for k, val in extra.items():
            lines.append(f"- **{k}** : {val}")
        lines.append("")

    lines.extend([
        "## Thresholds used",
        "",
        "Ces seuils sont paramétrables (voir `GateThresholds`). Les valeurs ici sont",
        "les defaults du PoC, calibrés sur données synthétiques. À recalibrer sur le",
        "dataset cible avant usage en production.",
        "",
        "| Gate | Accept si ≤ | Reject si ≥ |",
        "|---|---|---|",
        f"| Motion peak \\|div\\| | {thresholds.motion_accept_below} | {thresholds.motion_reject_above} |",
        f"| 3D-SSIM | *≥* {thresholds.ssim3d_accept_above} | *≤* {thresholds.ssim3d_reject_below} |",
        f"| Identity drift | {thresholds.identity_accept_below} | {thresholds.identity_reject_above} |",
        f"| Global score (geo mean) | *≥* {thresholds.global_accept_above} | *≤* {thresholds.global_reject_below} |",
        "",
        "## Verdict summary",
        "",
        "| Verdict | Count |",
        "|---|---:|",
    ])
    for verdict_name in ("ACCEPT", "REVIEW", "REJECT", "BLOCKED_SAFETY", "NO_DATA"):
        lines.append(f"| {verdict_name} | {len(by_verdict.get(verdict_name, []))} |")
    lines.append(f"| **Total** | {len(verdicts)} |")
    lines.append("")
    lines.append(f"**Mean global score (verdicts with data)** : {mean_global:.3f}")
    lines.append("")

    # Table détaillée
    lines.extend([
        "## Per-shot detail",
        "",
        "| Shot | Safety | Motion peak | 3D-SSIM | Id drift | Global | Verdict |",
        "|---:|---|---:|---:|---:|---:|---|",
    ])
    for v in sorted(verdicts, key=lambda x: x.shot_id):
        def fmt(x):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
        lines.append(
            f"| {v.shot_id} | {v.safety_status} | "
            f"{fmt(v.motion_peak_div)} | {fmt(v.ssim3d)} | {fmt(v.identity_drift)} | "
            f"{fmt(v.global_score)} | **{v.verdict}** |"
        )
    lines.append("")

    # Plans nécessitant attention
    problem_verdicts = [v for v in verdicts if v.verdict in ("REVIEW", "REJECT")]
    if problem_verdicts:
        lines.extend([
            "## Shots requiring attention",
            "",
        ])
        for v in problem_verdicts:
            lines.append(f"### Shot {v.shot_id} — {v.verdict}")
            for note in v.notes:
                lines.append(f"- {note}")
            lines.append("")

    # Plans BLOCKED_SAFETY
    blocked = by_verdict.get("BLOCKED_SAFETY", [])
    if blocked:
        lines.extend([
            "## Shots blocked by safety gate",
            "",
            "Ces plans n'ont pas été évalués par les quality gates — le safety gate",
            "les a mis en file d'attente pour revue humaine en amont.",
            "",
        ])
        for v in blocked:
            lines.append(f"- Shot {v.shot_id}")
        lines.append("")

    lines.extend([
        "## Methodology notes",
        "",
        "- **Motion peak \\|div\\|** : valeur maximale de la divergence du champ de",
        "  flux optique Farnebäck entre frames consécutives du plan. Calibré sur",
        "  120 cas synthétiques (AUC 0.997 pour séparer cas propres / cassés).",
        "- **3D-SSIM** : SSIM volumétrique sur blocs 7×7×7, info-weighted pooling.",
        "  Comparaison à un volume de référence ou stabilité volumétrique interne.",
        "- **Identity drift** : distance cosinus moyenne entre l'embedding du premier",
        "  plan d'un personnage et ses occurrences suivantes. Embedder classique",
        "  (HOG+LBP+couleur) par défaut ; swappable pour ArcFace en production.",
        "- **Global score** : moyenne géométrique des trois gates. Pénalise plus",
        "  fortement qu'une moyenne arithmétique la présence d'un gate très bas.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Démo
# ---------------------------------------------------------------------------

def demo():
    # Verdicts simulés pour illustrer le rapport
    th = GateThresholds()
    verdicts = [
        aggregate_shot_verdict(1, motion_peak_div=1.5, ssim3d=0.98, identity_drift=0.05,
                               safety_status="clean", thresholds=th),
        aggregate_shot_verdict(2, motion_peak_div=3.5, ssim3d=0.92, identity_drift=0.12,
                               safety_status="clean", thresholds=th),
        aggregate_shot_verdict(3, motion_peak_div=6.8, ssim3d=0.75, identity_drift=0.45,
                               safety_status="clean", thresholds=th),
        aggregate_shot_verdict(4, motion_peak_div=1.8, ssim3d=0.96, identity_drift=0.08,
                               safety_status="neutralized", thresholds=th),
        aggregate_shot_verdict(5, motion_peak_div=None, ssim3d=None, identity_drift=None,
                               safety_status="blocked_for_review", thresholds=th),
    ]

    backend_info = {
        "backend": "placeholder",
        "needs_gpu": False,
        "needs_network": False,
    }
    md = build_phase4_report(
        verdicts, backend_info, th,
        extra_context={"csv_source": "shots_sample.csv", "n_shots_evaluated": len(verdicts)},
    )
    out = Path(__file__).parent / "_phase4_demo_report.md"
    out.write_text(md, encoding="utf-8")
    print(f"Rapport démo écrit : {out}")
    # Affiche le résumé
    for v in verdicts:
        gs = "—" if v.global_score is None or np.isnan(v.global_score) else f"{v.global_score:.3f}"
        print(f"  Shot {v.shot_id:>2}  safety={v.safety_status:<20}  "
              f"global={gs:<6}  verdict={v.verdict}")


if __name__ == "__main__":
    demo()
