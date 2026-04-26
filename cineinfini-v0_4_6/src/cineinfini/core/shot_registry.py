"""
shot_registry.py
================
Parse un CSV (shot_id, shot_description) et en extrait des métadonnées
structurées : personnages nommés, lieux, époque, type d'action, indicateurs
sensibles (personnes publiques réelles, contenu historiquement chargé).

Ne republie jamais les descriptions brutes dans les outputs — seules les
métadonnées dérivées sortent. La description complète reste uniquement
en mémoire pendant le traitement.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Lexiques pour extraction par règles (pas de NLP lourd)
# ---------------------------------------------------------------------------

# Personnages récurrents dans le matériau source — identité à suivre dans le temps
TRACKED_CHARACTERS = {
    "forrest": "forrest",
    "forrest gump": "forrest",
    "forrest jr": "forrest_jr",
    "jenny": "jenny",
    "jenny curran": "jenny",
    "mrs. gump": "mrs_gump",
    "momma": "mrs_gump",
    "bubba": "bubba",
    "lt. dan": "lt_dan",
    "lieutenant dan": "lt_dan",
    "principal": "principal_hancock",
    "mr. hancock": "principal_hancock",
    "louise": "louise",
    "drill sergeant": "drill_sergeant",
    "sgt. sims": "sgt_sims",
    "carla": "carla",
    "lenore": "lenore",
    "ruben": "ruben",
    "wesley": "wesley",
}

# Personnes publiques réelles : drapeau rouge pour la génération
REAL_PEOPLE_FLAGGED = {
    "elvis presley", "elvis",
    "nathan bedford forrest",
    "chet huntley",
    "george wallace",
    "governor wallace",
    "katzenbach",
    "president kennedy", "kennedy", "john", "bobby",
    "marilyn monroe",
    "president johnson", "johnson", "lbj",
    "president nixon", "nixon",
    "president ford", "gerald ford",
    "president carter", "carter",
    "bob hope",
    "dick clark",
    "neil armstrong",
    "sarah jane moore",
    "mao tse-tung", "mao",
    "frank wills",
}

# Lieux récurrents
LOCATIONS = {
    "greenbow": "greenbow_alabama",
    "alabama": "alabama_generic",
    "savannah": "savannah_ga",
    "vietnam": "vietnam",
    "bayou la batre": "bayou_la_batre",
    "tuscaloosa": "tuscaloosa",
    "washington": "washington_dc",
    "white house": "washington_dc",
    "california": "california",
    "berkeley": "berkeley_ca",
    "memphis": "memphis_tn",
    "china": "china",
    "new york": "new_york",
    "times square": "new_york",
    "watergate": "washington_dc",
    "mississippi": "mississippi",
    "santa monica": "california",
}

SCENE_TYPES = {
    "interior": ["inside", "bedroom", "room", "office", "kitchen", "hall", "house"],
    "exterior_nature": ["oak tree", "river", "field", "lake", "desert", "mountain", "bayou", "ocean"],
    "exterior_urban": ["street", "sidewalk", "city", "highway", "road", "park"],
    "vehicle": ["bus", "car", "cab", "truck", "helicopter", "boat"],
    "institutional": ["school", "hospital", "church", "army", "principal", "football field"],
    "crowd": ["crowd", "people", "followers", "protesters", "boarders"],
    "military": ["army", "vietnam", "platoon", "helicopter", "soldiers", "war", "firebase"],
}

ERA_KEYWORDS = {
    "1951": 1951, "1963": 1963, "1972": 1972, "1982": 1982,
    "civil war": 1864, "revolutionary war": 1776, "slavery": 1860,
}

SENSITIVE_CONTENT = {
    "ku_klux_klan": ["kkl", "ku klux klan", "klan", "hooded sheet"],
    "assassination": ["assassination", "shot that nice young", "gunshots", "would-be assassin"],
    "drug_use": ["cocaine", "acid", "sugar cube", "syringe", "snorting"],
    "intimate_content": ["make love", "removes her top", "lenore leaps on him"],
    "violence": ["chain gang", "throw rocks", "pound on the roof"],
    "war_imagery": ["chain gang", "bullet wound", "dead in the snow", "wheelchair"],
}


# ---------------------------------------------------------------------------
# Structure de données
# ---------------------------------------------------------------------------

@dataclass
class ShotMetadata:
    shot_id: int
    n_words: int
    characters: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    scene_types: list[str] = field(default_factory=list)
    era: int | None = None
    has_voiceover: bool = False
    has_dialogue: bool = False
    has_tv_within_scene: bool = False
    real_people_mentioned: list[str] = field(default_factory=list)
    sensitive_flags: list[str] = field(default_factory=list)
    estimated_complexity: float = 1.0  # facteur pour budget GPU hypothétique


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _clean_description(raw: str) -> str:
    """Retire le boilerplate commercial du scrape et normalise."""
    noise_patterns = [
        r"Script provided for educational purposes.*?library",
        r"More scripts can be found here:.*?\s",
        r"http\S+",
    ]
    text = raw
    for p in noise_patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _find_tokens(text_lower: str, lexicon: dict) -> list[str]:
    """Retourne les valeurs canoniques du lexique trouvées dans le texte."""
    hits = set()
    for surface, canonical in lexicon.items():
        if re.search(rf"\b{re.escape(surface)}\b", text_lower):
            hits.add(canonical)
    return sorted(hits)


def _find_real_people(text_lower: str) -> list[str]:
    hits = set()
    for name in REAL_PEOPLE_FLAGGED:
        if re.search(rf"\b{re.escape(name)}\b", text_lower):
            hits.add(name)
    return sorted(hits)


def _find_scene_types(text_lower: str) -> list[str]:
    hits = set()
    for scene_type, keywords in SCENE_TYPES.items():
        for kw in keywords:
            if kw in text_lower:
                hits.add(scene_type)
                break
    return sorted(hits)


def _find_era(text_lower: str) -> int | None:
    for kw, year in ERA_KEYWORDS.items():
        if kw in text_lower:
            return year
    # Détection implicite par mention de présidents
    if "kennedy" in text_lower or "1963" in text_lower:
        return 1963
    if "nixon" in text_lower:
        return 1972
    if "carter" in text_lower:
        return 1978
    if "ford" in text_lower and "president" in text_lower:
        return 1975
    return None


def _find_sensitive(text_lower: str) -> list[str]:
    hits = set()
    for cat, patterns in SENSITIVE_CONTENT.items():
        for p in patterns:
            if p in text_lower:
                hits.add(cat)
                break
    return sorted(hits)


def _complexity_score(meta: ShotMetadata) -> float:
    """Heuristique de complexité de génération.

    Plus il y a de personnages, de transitions spatiales, d'époque exotique,
    plus le shot coûterait cher à générer.
    """
    score = 1.0
    score += 0.3 * len(meta.characters)
    score += 0.2 * len(meta.locations)
    score += 0.4 * len(meta.scene_types)
    if meta.has_tv_within_scene:
        score += 0.5  # rendu de TV dans scène = récursif
    if "crowd" in meta.scene_types:
        score += 1.0
    if "military" in meta.scene_types:
        score += 1.5
    if meta.era and meta.era < 1950:
        score += 1.0  # reconstitution d'époque
    return round(score, 2)


def extract_metadata(shot_id: int, raw_description: str) -> ShotMetadata:
    text = _clean_description(raw_description)
    text_lower = text.lower()

    meta = ShotMetadata(
        shot_id=shot_id,
        n_words=len(text.split()),
        characters=_find_tokens(text_lower, TRACKED_CHARACTERS),
        locations=_find_tokens(text_lower, LOCATIONS),
        scene_types=_find_scene_types(text_lower),
        era=_find_era(text_lower),
        has_voiceover="(v.o.)" in text_lower or "(vo)" in text_lower,
        has_dialogue=bool(re.search(r"\b[A-Z]{3,}\b", text)),
        has_tv_within_scene="television" in text_lower or "over tv" in text_lower or "on tv" in text_lower,
        real_people_mentioned=_find_real_people(text_lower),
        sensitive_flags=_find_sensitive(text_lower),
    )
    meta.estimated_complexity = _complexity_score(meta)
    return meta


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_registry(csv_path: str | Path) -> list[ShotMetadata]:
    """Charge un CSV (shot_id, shot_description) et retourne la liste de métadonnées."""
    csv_path = Path(csv_path)
    registry: list[ShotMetadata] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sid = int(row["shot_id"])
            except (ValueError, KeyError):
                continue
            desc = row.get("shot_description", "")
            registry.append(extract_metadata(sid, desc))
    return registry


def dedupe_registry(registry: list[ShotMetadata]) -> list[ShotMetadata]:
    """Enlève les doublons par shot_id (le CSV fourni avait des lignes répétées)."""
    seen = set()
    deduped = []
    for meta in registry:
        if meta.shot_id in seen:
            continue
        seen.add(meta.shot_id)
        deduped.append(meta)
    return deduped


# ---------------------------------------------------------------------------
# Démo
# ---------------------------------------------------------------------------

def demo():
    import json
    csv_path = Path(__file__).parent / "shots_sample.csv"
    registry = dedupe_registry(load_registry(csv_path))
    print(f"Registre chargé : {len(registry)} plans uniques\n")

    print(f"{'ID':>3} {'chars':<25} {'locations':<20} {'types':<25} {'C':>5} {'flags':<15}")
    print("-" * 100)
    for m in registry:
        chars = ",".join(m.characters)[:24]
        locs = ",".join(m.locations)[:19]
        types = ",".join(m.scene_types)[:24]
        flags = ",".join(m.sensitive_flags)[:14] or "-"
        print(f"{m.shot_id:>3} {chars:<25} {locs:<20} {types:<25} {m.estimated_complexity:>5.2f} {flags:<15}")

    # Stats agrégées
    total_complexity = sum(m.estimated_complexity for m in registry)
    sensitive = [m for m in registry if m.sensitive_flags]
    real_people_shots = [m for m in registry if m.real_people_mentioned]

    print(f"\nStatistiques :")
    print(f"  Complexité cumulée (unités GPU-hypothétiques) : {total_complexity:.1f}")
    print(f"  Plans avec contenu sensible : {len(sensitive)}/{len(registry)}")
    print(f"  Plans mentionnant des personnes publiques réelles : {len(real_people_shots)}/{len(registry)}")
    if real_people_shots:
        print(f"    → Plans {[m.shot_id for m in real_people_shots]} nécessitent un traitement spécial "
              f"(ne peuvent pas être générés directement)")


if __name__ == "__main__":
    demo()
