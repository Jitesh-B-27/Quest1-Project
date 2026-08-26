# Project Scope: Video Dialogue Localization

## Purpose

This project locates a requested spoken dialogue in a video without requiring a person to inspect the video manually. Given a supported video source and a target phrase, the system identifies the most likely point at which the phrase is spoken and extracts the corresponding video frame.

The supplied problem phrase, "My mind rebels at stagnation," and its public video are the primary evaluation example. The solution is designed to work with other videos and target phrases subject to the assumptions and limitations described below.

## In Scope

The current project includes:

- Accepting one non-empty target phrase and one video source per run.
- Using either a publicly downloadable video URL or compatible video and WAV files already available locally.
- Automatically downloading the video when a URL is supplied.
- Extracting audio from supported video files.
- Transcribing speech with word-level timing and confidence information.
- Finding the single best lexical or near-lexical match for the target phrase in the transcript.
- Estimating the start time of the matched spoken phrase and mapping it to a video frame.
- Extracting one JPEG image at that point in the video.
- Returning the target and matched text, timestamp, frame number, frame image path, similarity and confidence indicators, and relevant processing metadata.
- Providing the workflow through both a command-line interface and a local web interface.
- Supporting multiple speech-recognition backends and model sizes for experimentation and benchmarking.
- Reusing previously downloaded video or extracted audio to avoid unnecessary repeated processing.
- Supporting both a full-audio baseline pipeline and an optimized coarse-to-fine localization pipeline.
- Refining the final dialogue onset using forced alignment in the optimized pipeline.
- Reporting a clear failure when media preparation, transcription, matching, or frame extraction cannot be completed reliably.

## Project Assumptions

- "Dialogue" is interpreted as speech audible in the video's audio track. The provided evaluation video contains the requested dialogue as speech rather than visible on-screen text, so the solution uses speech localization rather than OCR.

- The extracted frame represents the estimated onset of the spoken phrase. It does not attempt to determine when equivalent subtitle or caption text becomes visible.

- The target phrase is known in advance and is expected to be in the same language as the speech being transcribed.

- When a phrase occurs multiple times, returning the highest-ranked occurrence is sufficient for the current scope.

- The spoken wording is identical or reasonably close to the supplied target after normalization. Minor ASR errors, punctuation differences, and small lexical differences are expected and handled through fuzzy matching.

- The video's audio must be sufficiently recognizable by the selected ASR model.

- Remote video sources must be accessible to the configured downloader.

- The input media must contain usable audio and video streams so that both dialogue localization and frame extraction can be performed.

- Similarity and ASR confidence values are supporting signals rather than guarantees of correctness. Results below the configured acceptance threshold are rejected.

- The web interface is intended as a local interface for running and demonstrating the localization pipeline.

## Out of Scope

The following capabilities are not part of the current implementation:

- **OCR-based text localization.** The system localizes spoken dialogue and does not search video frames for subtitles, captions, signs, or other visible text.

- **Semantic or translated matching.** Matching is based on the recognized wording of the dialogue. Substantially paraphrased or translated versions of the target are not currently searched semantically.

- **Multiple-occurrence retrieval.** The pipeline returns the single best matching occurrence rather than every occurrence of the target dialogue.

- **Speaker identification.** The system determines when the target dialogue is spoken, but does not identify which speaker said it.

- **Live or real-time video processing.** The current pipeline operates on downloadable or locally available media rather than live streams.

- **Batch processing.** A pipeline execution processes one video and one target dialogue at a time.

- **Production-scale web serving.** The provided FastAPI interface is intended for local execution and demonstration rather than concurrent multi-user deployment.

- **Formal dataset-level accuracy evaluation.** The implementation has been tested and benchmarked on multiple videos, but no labelled evaluation dataset is used to report formal localization accuracy or WER.

## Completion Criteria

For a supported input in which the target dialogue can be recognized with sufficient confidence, the project is considered successful when it automatically produces, without manual inspection:

- the localized timestamp in seconds and `HH:MM:SS.sss` format;
- the corresponding frame number;
- the matched dialogue text;
- the corresponding extracted frame image; and
- similarity/confidence information indicating the quality of the match.

If the target cannot be localized with sufficient confidence, the system should return an identifiable failure rather than presenting an unreliable result as a valid match.