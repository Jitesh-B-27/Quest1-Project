"""Core matching logic for the V1 dialogue matcher.

The four independently-modifiable operations are:
    normalize_text()      -- text normalization
    generate_windows()    -- candidate generation
    text_similarity()     -- similarity calculation
    rank_candidates()     -- candidate ranking
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


class MatcherError(Exception):
    """Base exception for the matcher module."""


class TranscriptError(MatcherError):
    """Raised when the transcript file is missing, invalid, or malformed."""


class MatchNotFoundError(MatcherError):
    """Raised when no candidate meets the minimum similarity threshold."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Candidate:
    matched_text: str
    start_time: float
    end_time: float
    text_similarity: float
    average_word_probability: float
    minimum_word_probability: float
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_text": self.matched_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text_similarity": round(self.text_similarity, 4),
            "average_word_probability": round(self.average_word_probability, 4),
            "minimum_word_probability": round(self.minimum_word_probability, 4),
        }


@dataclass
class MatchResult:
    target_text: str
    matched_text: str
    start_time: float
    end_time: float
    text_similarity: float
    average_word_probability: float
    minimum_word_probability: float
    matched_words: list[Word]
    candidates: list[Candidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_text": self.target_text,
            "matched_text": self.matched_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text_similarity": round(self.text_similarity, 4),
            "average_word_probability": round(self.average_word_probability, 4),
            "minimum_word_probability": round(self.minimum_word_probability, 4),
            "candidates": [c.to_dict() for c in self.candidates],
        }


# ---------------------------------------------------------------------------
# Normalization (V1: lowercase, punctuation, whitespace, apostrophes)
# ---------------------------------------------------------------------------

_APOSTROPHES = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u02bc": "'",
})
_PUNCT_TO_SPACE = str.maketrans({c: " " for c in string.punctuation if c != "'"})


def normalize_text(text: str) -> str:
    """Normalize text for comparison (same rules for target and transcript)."""
    text = text.lower()
    text = text.translate(_APOSTROPHES)
    # Keep apostrophes inside words (don't -> don't); drop boundary ones.
    text = re.sub(r"(?:^|(?<=\s))'|'(?=$|\s)", " ", text)
    text = text.translate(_PUNCT_TO_SPACE)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Candidate generation (sliding windows of size N-2 .. N+2)
# ---------------------------------------------------------------------------

def generate_windows(words: list[Word], target_len: int) -> list[list[Word]]:
    """Return all contiguous word windows sized N-2 through N+2."""
    sizes = range(max(1, target_len - 2), target_len + 3)
    windows: list[list[Word]] = []
    for size in sizes:
        if size > len(words):
            continue
        for i in range(len(words) - size + 1):
            windows.append(words[i:i + size])
    return windows


# ---------------------------------------------------------------------------
# Similarity calculation (V1: lexical ratio on normalized text)
# ---------------------------------------------------------------------------

def text_similarity(target_norm: str, window_norm: str) -> float:
    """Normalized lexical similarity in [0.0, 1.0] (1.0 = exact match)."""
    if not target_norm or not window_norm:
        return 0.0
    return SequenceMatcher(None, target_norm, window_norm).ratio()


# ---------------------------------------------------------------------------
# Candidate ranking (V1: similarity is primary; avg probability tiebreak)
# ---------------------------------------------------------------------------

def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (c.text_similarity, c.average_word_probability),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Transcript loading and validation
# ---------------------------------------------------------------------------

def load_transcript(path: str | Path) -> list[Word]:
    """Load and validate transcript.json, returning a flat list of Words."""
    path = Path(path)
    if not path.exists():
        raise TranscriptError(f"Transcript file does not exist: {path.resolve()}")
    if not path.is_file():
        raise TranscriptError(f"Transcript path is not a file: {path.resolve()}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise TranscriptError(f"Transcript file contains invalid JSON: {e}") from e
    except OSError as e:
        raise TranscriptError(f"Cannot read transcript file '{path}': {e}") from e

    if not isinstance(data, dict):
        raise TranscriptError("Transcript root must be a JSON object.")

    raw_words = data.get("words")
    if raw_words is None:
        raise TranscriptError("Transcript is missing the 'words' field.")
    if not isinstance(raw_words, list):
        raise TranscriptError("Transcript 'words' field must be a list.")
    if not raw_words:
        raise TranscriptError("Transcript 'words' list is empty.")

    words: list[Word] = []
    for i, entry in enumerate(raw_words):
        try:
            if not isinstance(entry, dict):
                raise TypeError("word entry must be an object")
            w = Word(
                word=str(entry["word"]),
                start=float(entry["start"]),
                end=float(entry["end"]),
                probability=float(entry["probability"]),
            )
            if w.end < w.start:
                raise ValueError("end timestamp is before start timestamp")
        except (KeyError, TypeError, ValueError) as e:
            raise TranscriptError(
                f"Malformed word entry at index {i}: {e}"
            ) from e
        words.append(w)
    return words


# ---------------------------------------------------------------------------
# DialogueMatcher
# ---------------------------------------------------------------------------

def _confidence(words: list[Word]) -> tuple[float, float]:
    probs = [w.probability for w in words]
    return sum(probs) / len(probs), min(probs)


class DialogueMatcher:
    """Find the best occurrence of a target phrase within a transcript."""

    def __init__(
        self,
        top_n: int = 5,
        min_similarity: float = 0.7,
    ) -> None:
        self.top_n = top_n
        self.min_similarity = min_similarity

    def find_best_match(
        self,
        target: str,
        transcript_path: str | Path,
    ) -> MatchResult:
        """Match target text against a transcript.json file.

        Raises:
            MatcherError / subclasses: On bad input, bad transcript, or when
                no candidate reaches ``min_similarity``.
        """
        if not isinstance(target, str) or not target.strip():
            raise MatcherError("Target text must be a non-empty string.")
        if not isinstance(transcript_path, (str, Path)) or not Path(transcript_path).name:
            raise MatcherError("Transcript path must be a valid path string.")

        words = load_transcript(transcript_path)
        return self.find_best_match_in_words(target, words)

    def find_best_match_in_words(self, target: str, words: list[Word]) -> MatchResult:
        """Same as find_best_match but takes an already-loaded word list."""
        if not isinstance(target, str) or not target.strip():
            raise MatcherError("Target text must be a non-empty string.")

        target_norm = normalize_text(target)
        if not target_norm:
            raise MatcherError(
                f"Target normalizes to empty text (only punctuation/whitespace): {target!r}"
            )

        target_tokens = target_norm.split()
        windows = generate_windows(words, len(target_tokens))
        if not windows:
            raise MatcherError(
                f"No candidate windows could be generated "
                f"(transcript has {len(words)} words)."
            )

        candidates: list[Candidate] = []
        for window in windows:
            window_norm = " ".join(normalize_text(w.word) for w in window).strip()
            similarity = text_similarity(target_norm, window_norm)
            avg_prob, min_prob = _confidence(window)
            candidates.append(Candidate(
                matched_text=" ".join(w.word for w in window),
                start_time=window[0].start,
                end_time=window[-1].end,
                text_similarity=similarity,
                average_word_probability=avg_prob,
                minimum_word_probability=min_prob,
                words=window,
            ))

        ranked = rank_candidates(candidates)
        top = ranked[: self.top_n]
        best = top[0]

        if best.text_similarity < self.min_similarity:
            raise MatchNotFoundError(
                f"No reasonable match found. Best similarity "
                f"{best.text_similarity:.2f} is below minimum "
                f"{self.min_similarity:.2f}. Best candidate: "
                f"'{best.matched_text}'"
            )

        return MatchResult(
            target_text=target,
            matched_text=best.matched_text,
            start_time=best.start_time,
            end_time=best.end_time,
            text_similarity=best.text_similarity,
            average_word_probability=best.average_word_probability,
            minimum_word_probability=best.minimum_word_probability,
            matched_words=list(best.words),
            candidates=top,
        )
