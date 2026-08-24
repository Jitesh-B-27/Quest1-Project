"""V1 Frame Extraction module.

Extracts the video frame corresponding to a dialogue start timestamp
and saves it as an image together with structured metadata.

Python API:
    from frame_extractor import FrameExtractor

    extractor = FrameExtractor()
    result = extractor.extract_at_timestamp("video/video.mp4", 324.990)

CLI:
    python -m frame_extractor --video video/video.mp4 --timestamp 324.990
"""

from frame_extractor.core import (
    FrameExtractor,
    FrameExtractorError,
    FrameResult,
    MetadataError,
    TimestampToFrameNumberError,
    ValidationError,
    timestamp_to_frame_number,
)

__all__ = [
    "FrameExtractor",
    "FrameExtractorError",
    "FrameResult",
    "MetadataError",
    "TimestampToFrameNumberError",
    "ValidationError",
    "timestamp_to_frame_number",
]
