"""V2 coarse-to-fine dialogue localization.

Python API:
    from localizer import run_cascade
    from localizer.core import format_timestamp

    result = run_cascade("audio/audio.wav", "My mind rebels at stagnation")
    print(result.global_timestamp, format_timestamp(result.global_timestamp))

CLI:
    python -m localizer --wav audio/audio.wav --target "..."
"""

from typing import Any

__all__ = [
    "CascadeResult",
    "CandidateRegion",
    "LocalizationError",
    "format_timestamp",
    "run_cascade",
]

_LAZY_ATTRS = {
    "CascadeResult": ("localizer.cascade", "CascadeResult"),
    "CandidateRegion": ("localizer.core", "CandidateRegion"),
    "LocalizationError": ("localizer.core", "LocalizationError"),
    "format_timestamp": ("localizer.core", "format_timestamp"),
    "run_cascade": ("localizer.cascade", "run_cascade"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
