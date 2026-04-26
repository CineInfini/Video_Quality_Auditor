"""
prompt_engineering.py
=====================
Construit les prompts à deux étages (HVC Stage 1 & Stage 2 selon ConsID-Gen,
arXiv:2602.10113) à partir des métadonnées extraites par shot_registry.

Stage 1 : Appearance-aware — fixe l'apparence invariante
Stage 2 : Temporal-aware   — décrit le mouvement / la caméra

Safety layer : neutralise les références à des personnes publiques réelles
et flagge le contenu sensible pour revue humaine (HITL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .shot_registry import ShotMetadata


# ---------------------------------------------------------------------------
# Descripteurs d'apparence fixes pour les personnages récurrents
# ---------------------------------------------------------------------------
# Ces descripteurs sont génériques (pas de lien avec des acteurs réels) et
# serviraient à ancrer l'Identity Cube dans un pipeline de génération.

CHARACTER_APPEARANCE = {
    "forrest": {
        "age_profile": ["child_6-8", "teen_15-17", "young_adult_20-25", "adult_30-45"],
        "visual_traits": "fair-skinned man with short brown hair and blue eyes, average build",
        "default_wardrobe": "neat collared shirt, khakis, leather shoes",
    },
    "forrest_jr": {
        "age_profile": ["child_5-7"],
        "visual_traits": "young boy with short brown hair",
        "default_wardrobe": "casual school clothes",
    },
    "jenny": {
        "age_profile": ["child_6-8", "teen_15-17", "young_adult_20-25", "adult_30-35"],
        "visual_traits": "fair-skinned woman with long blonde hair and green eyes, slim build",
        "default_wardrobe": "period-appropriate dress",
    },
    "mrs_gump": {
        "age_profile": ["adult_35-55"],
        "visual_traits": "fair-skinned woman with dark hair pulled back, warm expression",
        "default_wardrobe": "1950s-60s housedress, apron",
    },
    "bubba": {
        "age_profile": ["young_adult_18-22"],
        "visual_traits": "tall black man, friendly face",
        "default_wardrobe": "army uniform",
    },
    "lt_dan": {
        "age_profile": ["adult_30-45"],
        "visual_traits": "fair-skinned man with dark hair, stern jaw",
        "default_wardrobe": "army officer uniform, later civilian / veteran attire",
    },
    "principal_hancock": {
        "age_profile": ["adult_50-65"],
        "visual_traits": "older man, spectacles, formal demeanor",
        "default_wardrobe": "1950s business suit",
    },
}

DEFAULT_CHARACTER_BLOCK = {
    "visual_traits": "background character, period-appropriate appearance",
    "default_wardrobe": "era-appropriate clothing",
}


# ---------------------------------------------------------------------------
# Style cinématographique constant (pour cohérence visuelle transversale)
# ---------------------------------------------------------------------------

CINEMATIC_STYLE = {
    "grading": "warm nostalgic color grade, slight film grain",
    "lens": "35mm equivalent, shallow depth of field on close-ups",
    "lighting": "natural motivated lighting, soft key",
    "aspect_ratio": "2.39:1",
}


# ---------------------------------------------------------------------------
# Structure de sortie
# ---------------------------------------------------------------------------

@dataclass
class ShotPrompt:
    shot_id: int
    stage1_appearance: str
    stage2_temporal: str
    negative_prompt: str
    safety_status: Literal["clean", "neutralized", "blocked_for_review"]
    safety_notes: list[str] = field(default_factory=list)
    character_ids_in_shot: list[str] = field(default_factory=list)
    scene_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safety policy
# ---------------------------------------------------------------------------

SENSITIVE_BLOCKING = {"ku_klux_klan", "assassination"}
SENSITIVE_WARNING = {"drug_use", "intimate_content", "violence", "war_imagery"}


def _apply_real_person_neutralization(meta: ShotMetadata) -> tuple[list[str], list[str]]:
    """Remplace les noms de personnes publiques par des rôles génériques."""
    if not meta.real_people_mentioned:
        return [], []

    substitutions = []
    notes = []
    for name in meta.real_people_mentioned:
        lname = name.lower()
        if "president" in lname or lname in {"kennedy", "nixon", "johnson", "carter", "ford"}:
            substitutions.append(f"'{name}' → 'a generic U.S. President figure (non-recognizable)'")
        elif lname == "elvis" or lname == "elvis presley":
            substitutions.append(f"'{name}' → 'a 1950s rock-and-roll performer (generic, non-identifiable)'")
        elif lname == "neil armstrong":
            substitutions.append(f"'{name}' → 'an astronaut figure in vintage spacesuit (generic)'")
        elif "wallace" in lname:
            substitutions.append(f"'{name}' → 'a 1960s Southern governor figure (generic)'")
        elif "nathan bedford forrest" in lname:
            substitutions.append(f"'{name}' → 'a 19th-century military figure (generic historical)'")
        else:
            substitutions.append(f"'{name}' → 'a generic public figure (non-recognizable)'")
        notes.append(f"Real person substituted: {substitutions[-1]}")
    return substitutions, notes


def _classify_safety(meta: ShotMetadata) -> tuple[str, list[str]]:
    """Détermine le statut de sécurité et les notes associées."""
    notes = []
    blocking = [f for f in meta.sensitive_flags if f in SENSITIVE_BLOCKING]
    warning = [f for f in meta.sensitive_flags if f in SENSITIVE_WARNING]

    if blocking:
        for flag in blocking:
            if flag == "ku_klux_klan":
                notes.append("KKK imagery present in source. Generation of hateful organization "
                             "regalia is blocked. Shot requires human-in-the-loop decision "
                             "on historical context framing (e.g., archival photo overlay) "
                             "or omission.")
            elif flag == "assassination":
                notes.append("Assassination imagery present. Generation requires HITL review; "
                             "prefer archival/news-style framing without explicit violence.")
        return "blocked_for_review", notes

    if warning:
        for flag in warning:
            notes.append(f"Sensitive category flagged ({flag}): depict suggestively, "
                         f"not graphically; HITL advised.")
    return ("neutralized" if meta.real_people_mentioned else "clean"), notes


# ---------------------------------------------------------------------------
# Construction des prompts
# ---------------------------------------------------------------------------

def _build_appearance_block(meta: ShotMetadata) -> str:
    """HVC Stage 1 — description statique de l'apparence."""
    parts = []

    # Personnages présents
    if meta.characters:
        char_descs = []
        for cid in meta.characters:
            block = CHARACTER_APPEARANCE.get(cid, DEFAULT_CHARACTER_BLOCK)
            visual = block.get("visual_traits", "")
            wardrobe = block.get("default_wardrobe", "")
            char_descs.append(f"[{cid}] {visual}, wearing {wardrobe}")
        parts.append("CHARACTERS: " + "; ".join(char_descs))

    # Décor / lieu
    if meta.locations:
        parts.append(f"LOCATION: {', '.join(meta.locations)}")
    if meta.scene_types:
        parts.append(f"SCENE TYPE: {', '.join(meta.scene_types)}")

    # Époque
    if meta.era:
        parts.append(f"PERIOD: circa {meta.era}, period-accurate props and wardrobe")

    # Style
    style_str = ", ".join(f"{k}: {v}" for k, v in CINEMATIC_STYLE.items())
    parts.append(f"STYLE: {style_str}")

    return " | ".join(parts)


def _build_temporal_block(meta: ShotMetadata) -> str:
    """HVC Stage 2 — description du mouvement / caméra.

    On reste vague sur l'action narrative (pas de reproduction du scénario) ;
    on spécifie juste le type de mouvement de caméra et la dynamique implicite.
    """
    cues = []

    if "vehicle" in meta.scene_types:
        cues.append("tracking shot following movement, moderate speed")
    if "crowd" in meta.scene_types:
        cues.append("wide establishing shot, subtle handheld motion")
    if "military" in meta.scene_types:
        cues.append("documentary-style handheld, slight camera shake")
    if meta.has_voiceover:
        cues.append("slow contemplative pacing to accommodate voice-over")
    if meta.has_tv_within_scene:
        cues.append("includes a diegetic television screen in frame")
    if "exterior_nature" in meta.scene_types:
        cues.append("gentle camera drift, natural light play")
    if not cues:
        cues.append("static medium shot with subtle breathing motion")

    return "CAMERA/MOTION: " + "; ".join(cues) + f" | DURATION: ~4 seconds at 24fps"


def _build_negative_prompt(meta: ShotMetadata) -> str:
    """Prompt négatif standard + interdictions spécifiques."""
    neg = [
        "deformed faces", "extra limbs", "text artifacts", "watermark",
        "low resolution", "jpeg artifacts",
        "identity drift", "temporal flicker",
    ]
    if meta.real_people_mentioned:
        neg.append("recognizable likeness of any specific real person")
    if "ku_klux_klan" in meta.sensitive_flags:
        neg.extend(["hate symbols", "hate group regalia"])
    return ", ".join(neg)


def build_prompt(meta: ShotMetadata) -> ShotPrompt:
    subs, sub_notes = _apply_real_person_neutralization(meta)
    safety_status, safety_notes = _classify_safety(meta)
    all_notes = sub_notes + safety_notes

    appearance = _build_appearance_block(meta)
    if subs:
        appearance += " | REAL-PERSON SUBSTITUTIONS: " + "; ".join(subs)

    temporal = _build_temporal_block(meta)
    negative = _build_negative_prompt(meta)

    return ShotPrompt(
        shot_id=meta.shot_id,
        stage1_appearance=appearance,
        stage2_temporal=temporal,
        negative_prompt=negative,
        safety_status=safety_status,
        safety_notes=all_notes,
        character_ids_in_shot=meta.characters,
        scene_tags=meta.scene_types,
    )


def build_all_prompts(registry: list[ShotMetadata]) -> list[ShotPrompt]:
    return [build_prompt(m) for m in registry]


# ---------------------------------------------------------------------------
# Démo
# ---------------------------------------------------------------------------

def demo():
    from .shot_registry import load_registry, dedupe_registry
    from pathlib import Path

    csv_path = Path(__file__).parent / "shots_sample.csv"
    registry = dedupe_registry(load_registry(csv_path))
    prompts = build_all_prompts(registry)

    status_count = {"clean": 0, "neutralized": 0, "blocked_for_review": 0}
    for p in prompts:
        status_count[p.safety_status] += 1

    print(f"Prompts générés : {len(prompts)}")
    print(f"  Clean             : {status_count['clean']}")
    print(f"  Neutralized       : {status_count['neutralized']}  (personnes publiques remplacées)")
    print(f"  Blocked for review: {status_count['blocked_for_review']}  (HITL requis)")
    print()

    # Affiche 3 exemples variés
    samples = [
        p for p in prompts if p.safety_status == "clean"
    ][:1] + [
        p for p in prompts if p.safety_status == "neutralized"
    ][:1] + [
        p for p in prompts if p.safety_status == "blocked_for_review"
    ][:1]

    for p in samples:
        print(f"--- Shot {p.shot_id} [{p.safety_status}] ---")
        print(f"Stage 1 (apparence)  : {p.stage1_appearance[:180]}...")
        print(f"Stage 2 (temporel)   : {p.stage2_temporal}")
        print(f"Prompt négatif       : {p.negative_prompt[:100]}...")
        if p.safety_notes:
            print(f"Notes de sécurité    :")
            for note in p.safety_notes:
                print(f"  - {note[:140]}")
        print()


if __name__ == "__main__":
    demo()
