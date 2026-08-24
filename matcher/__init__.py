"""V1 Target Dialogue Matching module.

Finds the occurrence in an ASR transcript that most closely matches a
target dialogue string, using sliding-window lexical similarity.

Python API:
    from matcher import DialogueMatcher

    matcher = DialogueMatcher()
    result = matcher.find_best_match(
        "My mind rebels at stagnation",
        "transcript/transcript.json",
    )

CLI:
    python -m matcher --transcript transcript/transcript.json \
        --target "My mind rebels at stagnation"
"""

from matcher.core import (
    Candidate,
    DialogueMatcher,
    MatchResult,
    MatcherError,
    TranscriptError,
)

__all__ = [
    "Candidate",
    "DialogueMatcher",
    "MatchResult",
    "MatcherError",
    "TranscriptError",
]
