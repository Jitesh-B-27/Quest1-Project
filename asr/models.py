"""Typed result models for the ASR module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Word:
    """A single word with timing and raw model probability."""

    word: str
    start: float
    end: float
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptResult:
    """Structured transcription result: metadata + flat word list."""

    metadata: dict[str, Any]
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Full transcript as a single string."""
        return " ".join(w.word for w in self.words).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "words": [w.to_dict() for w in self.words],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_json_file(self, output_path: str | Path) -> Path:
        """Persist the transcript to a JSON file (parent dirs auto-created)."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptResult:
        words = [Word(**w) for w in data.get("words", [])]
        return cls(metadata=data.get("metadata", {}), words=words)
