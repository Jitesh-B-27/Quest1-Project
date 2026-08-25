"""NVIDIA Parakeet provider (CPU-compatible, via NVIDIA NeMo).

Uses an open Parakeet model (default: nvidia/parakeet-tdt-0.6b-v3) and maps
NeMo's native word timestamps + confidence onto the common Word schema.

Confidence policy: per-word scores are used when NeMo exposes them;
otherwise the hypothesis-level confidence is applied to its words. If
neither is available the probability is 0.0 and metadata records
``probability_source: none`` — no values are fabricated.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

from asr.exceptions import (
    DependencyError,
    ModelLoadError,
    TranscriptionError,
    ValidationError,
)
from asr.models import Word
from asr.providers.base import ASRProvider

DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

# Well-known open Parakeet checkpoints (any HF/NGC ASR name is also accepted).
KNOWN_PARAKEET_MODELS = (
    "nvidia/parakeet-tdt-0.6b-v3",
    "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-ctc-0.6b",
    "nvidia/stt_en_conformer_ctc_large",
)

_WHISPER_STYLE_SIZES = ("tiny", "base", "small", "medium", "large", "large-v3")


class ParakeetProvider(ASRProvider):
    provider_name = "parakeet"
    default_model = DEFAULT_PARAKEET_MODEL
    valid_models = ()  # any HF/NGC model name accepted

    def __init__(self, model=None, device="cpu", language=None,
                 chunk_seconds: int = 120, **kwargs):
        # compute_type is not applicable to NeMo; ignored by design.
        super().__init__(model=model, device=device, language=language)
        self._validate_model_name(self.model)
        self.chunk_seconds = chunk_seconds
        self._model = None

    # ------------------------------------------------------------------
    # Long-audio support: split into fixed-length chunks so NeMo never has
    # to decode an entire multi-minute file in one pass (a 54-min input
    # otherwise triggers a ~99 GB allocation inside the TDT decoder).
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ffmpeg() -> str | None:
        local = Path(__file__).resolve().parents[2] / "tools" / "ffmpeg" / "bin"
        found = shutil.which("ffmpeg", path=str(local)) or shutil.which("ffmpeg")
        return str(Path(found).resolve()) if found else None

    def _prepare_chunks(self, audio_path: str):
        """Convert to mono 16 kHz WAV and split into chunks via FFmpeg.

        Returns ``(chunks, cleanup)`` where ``chunks`` is a list of
        ``(path, offset_seconds)`` tuples. Falls back to the original
        single-file input when FFmpeg is unavailable or fails.
        """
        import tempfile

        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            return [(Path(audio_path), 0.0)], None

        tmp_dir = tempfile.TemporaryDirectory(prefix="parakeet_chunks_")
        try:
            pattern = str(Path(tmp_dir.name) / "chunk_%05d.wav")
            cmd = [
                ffmpeg, "-y", "-i", audio_path,
                "-ac", "1", "-ar", "16000", "-vn",
                "-f", "segment",
                "-segment_time", str(self.chunk_seconds),
                pattern,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=1800,
            )
            chunk_paths = sorted(Path(tmp_dir.name).glob("chunk_*.wav"))
            if result.returncode != 0 or not chunk_paths:
                shutil.rmtree(tmp_dir.name, ignore_errors=True)
                return [(Path(audio_path), 0.0)], None

            chunks = [
                (cp, round(i * self.chunk_seconds, 3))
                for i, cp in enumerate(chunk_paths)
            ]
            return chunks, tmp_dir
        except Exception:
            shutil.rmtree(tmp_dir.name, ignore_errors=True)
            return [(Path(audio_path), 0.0)], None

    @staticmethod
    def _validate_model_name(model: str) -> None:
        """Parakeet uses HF/NGC repo names, not Whisper-style size keywords."""
        lowered = (model or "").strip().lower()
        if "/" in model and len(model) > 1:
            return
        hint = ""
        if lowered in _WHISPER_STYLE_SIZES:
            hint = (" Size keywords like 'small' are a Whisper concept; "
                    "Parakeet models are identified by repository name, e.g. "
                    "'nvidia/parakeet-tdt-0.6b-v3'.")
        raise ValidationError(
            f"Invalid Parakeet model '{model}'. Expected a HuggingFace/NGC "
            f"repository name such as 'nvidia/parakeet-tdt-0.6b-v3'. "
            f"Known models: {', '.join(KNOWN_PARAKEET_MODELS)}.{hint}"
        )

    def _load_model(self):
        if self._model is None:
            try:
                asr_models = importlib.import_module(
                    "nemo.collections.asr.models"
                )
            except ImportError as e:
                raise DependencyError(
                    "NVIDIA NeMo is not installed. Install with: "
                    "pip install nemo_toolkit[asr]"
                ) from e
            try:
                self._model = asr_models.ASRModel.from_pretrained(
                    model_name=self.model
                )
                self._model.eval()
                self._model.to(self.device)
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load Parakeet model '{self.model}': {e}"
                ) from e
        return self._model

    @staticmethod
    def _extract_words(hyp) -> list[Word]:
        """Map a NeMo hypothesis to words across known NeMo output shapes."""
        raw_words = getattr(hyp, "words", None) or []
        timestamps = getattr(hyp, "word_time_offset", None)
        word_scores = getattr(hyp, "word_scores", None)

        hyp_confidence = None
        conf = getattr(hyp, "confidence", None)
        if isinstance(conf, (int, float)):
            hyp_confidence = float(conf)

        out: list[Word] = []
        for i, entry in enumerate(raw_words):
            probability = 0.0
            text, start, end = "", None, None

            if isinstance(entry, dict):
                text = str(entry.get("word", ""))
                start = entry["start"] if entry.get("start") is not None else entry.get("start_offset")
                end = entry["end"] if entry.get("end") is not None else entry.get("end_offset")
                prob = entry.get("confidence", entry.get("score"))
                if isinstance(prob, (int, float)):
                    probability = float(prob)
            else:
                obj = entry
                text = str(getattr(obj, "word", obj))
                start = getattr(obj, "start", None) or getattr(obj, "start_offset", None)
                end = getattr(obj, "end", None) or getattr(obj, "end_offset", None)
                prob = getattr(obj, "confidence", getattr(obj, "score", None))
                if isinstance(prob, (int, float)):
                    probability = float(prob)

            if isinstance(word_scores, (list, tuple)) and i < len(word_scores):
                score = word_scores[i]
                if isinstance(score, (int, float)):
                    probability = float(score)

            if probability == 0.0 and hyp_confidence is not None:
                probability = hyp_confidence

            if start is None and timestamps and i < len(timestamps):
                ts = timestamps[i]
                if isinstance(ts, dict):
                    start, end = ts.get("start"), ts.get("end")

            if not text or start is None or end is None:
                continue  # skip entries without usable timing data

            out.append(Word(
                word=text.strip(),
                start=round(float(start), 3),
                end=round(float(end), 3),
                probability=round(float(probability), 4),
            ))
        return out

    @staticmethod
    def _patch_tempdir_cleanup():
        """Monkey-patch ``tempfile.TemporaryDirectory.cleanup`` on Windows.

        NeMo's ``transcribe_generator`` wraps its work inside::

            with tempfile.TemporaryDirectory() as tmpdir:
                ...  # writes manifest.json, creates Lhotse dataloader

        When the ``with`` block exits, ``cleanup()`` calls ``shutil.rmtree``
        which raises ``[WinError 32]`` because the Lhotse / PyTorch
        DataLoader still holds an open handle on ``manifest.json``.

        The patch makes ``cleanup`` retry up to 10 times with short sleeps,
        then fall back to ``ignore_errors=True`` so that NeMo never crashes
        on temp-dir removal.  The patch is idempotent (safe to call twice).
        """
        import sys
        if sys.platform != "win32":
            return  # only needed on Windows

        import shutil
        import tempfile
        import time as _time

        if getattr(tempfile.TemporaryDirectory, "_win32_patched", False):
            return  # already patched

        _original_cleanup = tempfile.TemporaryDirectory.cleanup

        def _robust_cleanup(self):
            for attempt in range(10):
                try:
                    _original_cleanup(self)
                    return
                except (PermissionError, OSError, NotADirectoryError):
                    if attempt < 9:
                        _time.sleep(0.5)
            # Last resort — swallow errors so NeMo doesn't crash.
            try:
                shutil.rmtree(self.name, ignore_errors=True)
            except Exception:
                pass

        tempfile.TemporaryDirectory.cleanup = _robust_cleanup
        tempfile.TemporaryDirectory._win32_patched = True  # type: ignore[attr-defined]

    def _transcribe_one(self, model, torch, path: str):
        """Transcribe a single (short) audio file; return the hypothesis."""
        transcribe_kwargs = {
            "return_hypotheses": True,
            "num_workers": 0,
            "batch_size": 1,
        }
        attempts = [transcribe_kwargs]
        # Older NeMo versions may not accept newer kwargs; degrade gracefully.
        attempts.append({
            k: v for k, v in transcribe_kwargs.items()
            if k in ("return_hypotheses",)
        })
        attempts.append({})

        hypotheses = None
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                with torch.no_grad():
                    hypotheses = model.transcribe([path], **kwargs)
                break
            except TypeError as e:
                last_error = e  # unsupported kwargs on this NeMo version
                continue
            except Exception as e:
                raise TranscriptionError(
                    f"Parakeet transcription failed for '{path}': {e}"
                ) from e

        if hypotheses is None:
            raise TranscriptionError(
                f"Parakeet transcription failed for '{path}' "
                f"(tried {len(attempts)} call signatures): {last_error}"
            ) from last_error

        try:
            hyp = hypotheses[0][0] if isinstance(hypotheses[0], (list, tuple)) \
                else hypotheses[0]
        except (IndexError, TypeError, KeyError) as e:
            raise TranscriptionError(
                f"Parakeet returned no usable hypothesis for '{path}'."
            ) from e
        return hyp

    def _transcribe_words(self, audio_path: str):
        try:
            torch = importlib.import_module("torch")
        except ImportError as e:
            raise DependencyError(
                "PyTorch is required for the parakeet provider: pip install torch"
            ) from e

        model = self._load_model()

        # ---- Windows WinError 32 workaround ----
        # NeMo's transcribe_generator() uses a
        # ``with tempfile.TemporaryDirectory()`` block.  On Windows the
        # Lhotse DataLoader keeps manifest.json open, so cleanup() crashes
        # with WinError 32.  Patching cleanup to retry fixes this at the
        # source; the remaining config tweaks are defence-in-depth.
        self._patch_tempdir_cleanup()

        # Force num_workers=0 in the model's dataloader config so no
        # child processes hold file locks.
        if hasattr(model, "cfg"):
            try:
                from omegaconf import open_dict

                with open_dict(model.cfg):
                    if hasattr(model.cfg, "test_ds"):
                        model.cfg.test_ds.num_workers = 0
                    if hasattr(model.cfg, "validation_ds"):
                        model.cfg.validation_ds.num_workers = 0
            except Exception:
                pass  # non-critical: transcribe kwargs also set num_workers

        chunks, tmp_ctx = self._prepare_chunks(audio_path)
        all_words: list[Word] = []
        chunk_count = len(chunks)
        confidence_source = 'none'
        try:
            for path, offset in chunks:
                hyp = self._transcribe_one(model, torch, str(path))
                if confidence_source == 'none':
                    if getattr(hyp, 'word_scores', None):
                        confidence_source = 'nemo-word-scores'
                    elif getattr(hyp, 'confidence', None) is not None:
                        confidence_source = 'nemo-hypothesis-confidence'
                words = self._extract_words(hyp)
                if offset:
                    words = [
                        Word(word=w.word, start=round(w.start + offset, 3),
                             end=round(w.end + offset, 3),
                             probability=w.probability)
                        for w in words
                    ]
                all_words.extend(words)
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

        duration = max((w.end for w in all_words), default=None)
        extra = {
            "audio_duration_seconds": duration,
            "probability_source": confidence_source,
            "chunk_count": chunk_count,
            "chunk_seconds": self.chunk_seconds,
        }
        return all_words, extra
