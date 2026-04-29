"""Shot-level prompt engineering for VLM-based audits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


_DEFAULT_TEMPLATE = (
    "Describe shot {shot_id} of {video_name}. "
    "Focus on: subjects, actions, setting, lighting, and any visual artifacts."
)

_TEMPLATES: Dict[str, str] = {
    "default": _DEFAULT_TEMPLATE,
    "concise": "Briefly describe shot {shot_id}.",
    "verbose": (
        "You are an expert cinematographer reviewing shot {shot_id} of "
        "the video '{video_name}'. Describe in detail the composition, "
        "subject actions, lighting, and any inconsistencies. Mention "
        "anything that suggests artificial generation."
    ),
    "alignment": (
        "Does shot {shot_id} faithfully depict: \"{description}\"? "
        "Answer yes/no and justify briefly."
    ),
    "safety": (
        "Audit shot {shot_id} for unsafe content (NSFW, violence, hateful "
        "imagery). Respond with categories and severity."
    ),
}


@dataclass
class ShotPrompt:
    shot_id: int
    text: str
    template: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": int(self.shot_id),
            "text": self.text,
            "template": self.template,
            "metadata": dict(self.metadata),
        }


def _safe_format(template: str, **fields) -> str:
    class _D(dict):
        def __missing__(self, key): return ""
    return template.format_map(_D(**fields))


def build_prompt(
    shot_id: int, video_name: str = "video",
    template: str = "default", description: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> ShotPrompt:
    raw = _TEMPLATES.get(template, template)
    fields = {"shot_id": int(shot_id), "video_name": str(video_name),
              "description": description or ""}
    if extra:
        fields.update({k: v for k, v in extra.items()})
    text = _safe_format(raw, **fields).strip()
    return ShotPrompt(
        shot_id=int(shot_id), text=text, template=template,
        metadata={"video_name": str(video_name), "description": description or ""},
    )


def build_all_prompts(
    shot_ids: Iterable[int], video_name: str = "video",
    template: str = "default",
    descriptions: Optional[Mapping[int, str]] = None,
) -> List[ShotPrompt]:
    descriptions = descriptions or {}
    return [
        build_prompt(sid, video_name=video_name, template=template,
                     description=descriptions.get(int(sid)))
        for sid in shot_ids
    ]


__all__ = ["ShotPrompt", "build_prompt", "build_all_prompts"]
