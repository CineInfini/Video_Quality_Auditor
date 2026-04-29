"""Shot metadata: lightweight record of one shot's identity & location."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class ShotMetadata:
    shot_id: int
    start_frame: int
    end_frame: int
    duration_s: float = 0.0
    fps: float = 24.0
    label: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return max(0, int(self.end_frame) - int(self.start_frame) + 1)

    @property
    def start_time_s(self) -> float:
        return float(self.start_frame) / float(self.fps) if self.fps > 0 else 0.0

    @property
    def end_time_s(self) -> float:
        return float(self.end_frame) / float(self.fps) if self.fps > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": int(self.shot_id),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "duration_s": float(self.duration_s),
            "fps": float(self.fps),
            "label": self.label,
            "description": self.description,
            "tags": list(self.tags),
            "extra": dict(self.extra),
            "n_frames": self.n_frames,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
        }

    @classmethod
    def from_tuple(cls, shot_id: int, data, fps: float = 24.0) -> "ShotMetadata":
        if len(data) >= 3:
            start, end, duration = int(data[0]), int(data[1]), float(data[2])
        else:
            start, end = int(data[0]), int(data[1])
            duration = (end - start + 1) / fps if fps > 0 else 0.0
        return cls(shot_id=int(shot_id), start_frame=start, end_frame=end,
                   duration_s=duration, fps=fps)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShotMetadata":
        return cls(
            shot_id=int(data["shot_id"]),
            start_frame=int(data["start_frame"]),
            end_frame=int(data["end_frame"]),
            duration_s=float(data.get("duration_s", 0.0)),
            fps=float(data.get("fps", 24.0)),
            label=data.get("label"),
            description=data.get("description"),
            tags=list(data.get("tags", [])),
            extra=dict(data.get("extra", {})),
        )


__all__ = ["ShotMetadata"]
