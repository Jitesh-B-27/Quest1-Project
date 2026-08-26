# Engineering Approach

## 1. Establishing a Complete Baseline

The first goal was to solve the full problem before optimizing individual stages. V1 therefore followed the shortest complete path:

```text
video download -> audio extraction -> full-audio ASR
-> lexical dialogue match -> timestamp -> frame extraction
```

`pipeline.run_pipeline()` connected the existing importable stages and made failures attributable to a specific step. The initial practical configuration used faster-whisper `small` on CPU. This produced a measurable MVP and, equally importantly, a reference against which later performance changes could be checked. Optimizing before this point would have made it difficult to tell whether a faster component still produced the correct dialogue and frame.

## 2. Measurement and ASR Experiments

The long Sherlock input is 3,261.781 seconds (54.4 minutes). The recorded V1 runs provide the following comparison. Whisper and WhisperX were tested with `tiny`; faster-whisper was also tested across model sizes and before/after CPU optimization.

| Backend | Model / configuration | Media preparation | ASR time | Total time | Matched text | Similarity | Timestamp | Frame |
|---|---|---|---:|---:|---|---:|---:|---:|
| faster-whisper | `small`, initial V1 | Download + audio extraction included | 856.562 s | 1025.7 s | Exact | 1.0000 | 324.990 s | 7791 |
| faster-whisper | `base` | Cached video + WAV | 402.203 s | 414.7 s | “My mind rebels its stagnation.” | 0.9474 | 325.150 s | 7795 |
| faster-whisper | `tiny`, before CPU optimization | Cached video + WAV | 237.406 s | 249.0 s | “My mind represents stagnation.” | 0.8421 | 324.810 s | 7787 |
| faster-whisper | `tiny`, after CPU optimization | Cached video + WAV | 120.531 s | 132.3 s | Exact | 1.0000 | 325.310 s | 7799 |
| Whisper | `tiny` | Cached video + WAV | 547.813 s | 572.8 s | “My mind reveals its stagnation.” | 0.8966 | 325.310 s | 7799 |
| WhisperX | `tiny` | Cached video + WAV | 637.703 s | 697.0 s | Exact | 1.0000 | 325.283 s | 7798 |

The `small` total includes media acquisition and extraction, so it is not directly comparable with the cached-media totals. In the cached runs, ASR contributed approximately 91%-97% of total runtime across the listed backends and configurations. This made download, matching, and frame extraction poor primary optimization targets: even eliminating them entirely would leave most of the runtime intact.

The original ASR implementation was faster-whisper-specific. It was refactored behind a common provider contract so experiments would not spread backend-specific logic through the pipeline. `asr.Transcriber` is the facade used by callers; `BaseASRProvider` defines the shared `load()` and `transcribe()` contract; and `asr/factory.py` maps the requested model type to `FasterWhisperProvider`, `OpenAIWhisperProvider`, or `WhisperXProvider`. Every provider returns the same `TranscriptResult` containing timestamped `Word` values. The matcher, pipeline, benchmark path, and result format could therefore be reused unchanged while only backend/model configuration changed. Model benchmarking became a configuration change rather than a pipeline rewrite.

The results made faster-whisper the practical CPU-oriented path: its optimized `tiny` run was the fastest recorded V1 run while still producing the exact target text.

## 3. CPU and Matching Improvements

Repository history confirms that the earlier `FasterWhisperProvider` relied on CTranslate2 defaults: `cpu_threads` was not explicitly set and transcription did not request greedy decoding. The current provider in `asr/providers/faster_whisper_provider.py`:

- defaults to `int8` compute on CPU;
- sets `cpu_threads` to the available logical-core count;
- uses one worker to avoid redundant thread competition;
- sets `beam_size=1` and `temperature=0.0` to avoid unnecessary beam search and retry work;
- enables VAD to skip silence; and
- decodes the source once to an in-memory 16 kHz float array.

These changes directly target the dominant stage by reducing decoding work and allowing CTranslate2 to use the CPU explicitly. On the same cached Sherlock media with faster-whisper `tiny`, ASR time fell from 237.406 seconds to 120.531 seconds: a 49.2% reduction, or about 1.97x faster. Total pipeline time fell from 249.0 seconds to 132.3 seconds (about 1.88x faster). The matched output also changed from a fuzzy 0.8421 result to the exact phrase, although a single paired run is not enough to attribute accuracy improvement generally to the optimization.

Matching was also made cheaper without changing its role. Git history and `matcher/core.py` show the move from `difflib.SequenceMatcher` to RapidFuzz's compiled `fuzz.ratio`. The matcher applies the same normalization to the target and transcript, evaluates contiguous windows from N-2 through N+2 words, and ranks primarily by lexical similarity with average word probability as a tie-breaker. Probabilities remain supporting evidence rather than a complex combined score. This fits a known-quote search, while accepting that paraphrases and semantic equivalents are not handled.

### Development workflow

Repository refs preserve the working V1 baseline on `version-1` and show separate `model-abstraction` and `pipeline-version-2` branches/checkpoints for the larger experiments. The commit history then places the validated ASR abstraction and V2 end-to-end work on the lineage leading to `main`. Keeping known-good code recoverable while testing architectural changes reduced migration risk during rapid development; the history is linear and does not record explicit merge commits.

## 4. From V1 to Coarse-to-Fine V2

V1 exposed the central tradeoff: a small model was useful for accuracy, but applying it to an entire long recording was expensive; a tiny model was cheaper, but could mistranscribe the phrase. V2 avoids choosing only one side of that tradeoff:

- a cheap model searches the full audio for recall and reduces the search space;
- the stronger `small` model is applied only to short candidate regions for precision;
- expensive alignment is applied only after one candidate wins.

In `localizer/core.py`, `generate_top_k()` scores every N±2 coarse window and retains up to five diverse candidates by default. It deliberately has no coarse similarity rejection threshold: a noisy transcription should still nominate regions. `pad_and_merge()` adds context around candidates, clamps them to the recording, and merges overlaps before FFmpeg extracts short WAV files. `localizer/cascade.py` then transcribes each region with faster-whisper `small`; `DialogueMatcher` performs the actual thresholded RapidFuzz validation and selects the best result using similarity and word probabilities.

Top-K is important because trusting Top-1 would allow one coarse recognition error to discard the true region. Padding gives fine ASR enough surrounding speech to decode the phrase, while merging prevents repeated work on overlapping candidates.

## 5. Graceful Fallback and Timestamp Precision

The default cascade preserves V1 behavior when cheap retrieval fails:

```text
tiny full audio -> Top-K + small validation
base full audio -> Top-K + small validation
small full audio + V1-style matching
MatchNotFoundError
```

An empty coarse transcript, no candidate, or no fine match above the threshold advances to the next tier. All tiers reuse the same WAV. This makes V2 an optimization in the normal case rather than a replacement that sacrifices the known full-audio path. A region-extraction failure remains an operational error instead of being misreported as “not found.”

Timestamp correctness is kept separate from text retrieval. `CandidateRegion` boundaries are global full-audio times, but fine-ASR and alignment timestamps are local to an extracted region. `run_cascade()` converts the selected onset with:

```text
global timestamp = winner region global_start + local dialogue onset
```

Raw Whisper word timing can drift enough to select a neighboring frame. Instead of force-aligning the full 54-minute transcript, `asr/aligner.refine_word_timestamps()` runs WhisperX alignment only on the winning short region. `match_refined_onset()` maps the validated words to the refined local onset. If alignment fails or cannot map the phrase, the pipeline keeps the fine-ASR onset; alignment never invalidates an already validated result. The resulting global timestamp is passed to `FrameExtractor`, which maps it to a frame and extracts the JPEG. Human-readable `HH:MM:SS.sss` formatting happens at final output.

## 6. Observed V2 Results

The recorded runs below are engineering observations on two videos, not a formal accuracy benchmark or dataset-level claim.

| Input / target | Duration | Total | Coarse `tiny` | Region extraction | Fine validation | Alignment | Result | Timestamp | Frame |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Sherlock / “My mind rebels at stagnation” | 3261.781 s | 199.5 s | 135.984 s | 0.437 s | 27.313 s | 27.531 s | Exact, 1.0000 | 325.290 s | 7799 |
| Got any hobbies / “knew me earlier” | Not recorded | 88.0 s | 31.266 s | 0.469 s | 26.610 s | 27.734 s | Exact, 1.0000 | 422.178 s | 10132 |

The Sherlock run demonstrates the intended tradeoff: it reached an exact result while applying `small` ASR and alignment only to selected regions, rather than the full recording. Compared with the optimized faster-whisper `tiny` V1 run, V2 spent additional time on fine validation and alignment to obtain a refined timestamp; its goal is balancing retrieval cost, text accuracy, and frame timing rather than beating the cheapest coarse model alone. The second run shows the same cascade working on another video and phrase, but these observations should not be interpreted as formal accuracy evidence.

## 7. Adding the Web Interface

The web layer was added after the localization path was working, keeping presentation concerns out of the core experiments. A pipeline run can take minutes and emits meaningful intermediate stages, so `webapp/main.py` does not hold one HTTP request open until completion. `POST /api/localize` creates a background thread and returns a job ID; the static JavaScript frontend polls `GET /api/jobs/{job_id}` for progress, logs, failure, and the final `PipelineResult`.

`webapp.jobs.JobStore` intentionally keeps this interface lightweight: state is in memory and a worker lock permits one pipeline job at a time. This is sufficient for a local demonstration and avoids introducing persistence or distributed-job infrastructure into the localization work.

## 8. Bottlenecks and Trade-offs

| Bottleneck / constraint | Observation | Decision / trade-off |
|---|---|---|
| Full-audio ASR cost | ASR consumed about 91%-97% of cached V1 runtime | Optimize ASR before matcher or frame extraction |
| Stronger ASR on long audio | Better text quality carried a high CPU cost | Use `tiny` for coarse search and `small` only on candidate regions |
| Coarse-model errors | `tiny` can mistranscribe or miss the target | Keep Top-K regions, then retry `base` and full-audio `small` |
| Exact string matching | Small ASR wording errors break equality | Use normalized N±2 windows with RapidFuzz similarity |
| Raw word timestamps | ASR onset can select a neighboring frame | Force-align the validated winner for more precise timing |
| Full-audio alignment cost | Aligning the whole recording would add work outside the winning area | Apply WhisperX alignment only to the final short region |
| CPU decoding defaults | Default beam/thread behavior left avoidable work in the dominant stage | Configure logical CPU threads and greedy decoding explicitly |
| Long-running HTTP work | Localization can take minutes and has useful intermediate progress | Start a background job and poll its state |
| CPU contention in the web demo | Concurrent inference jobs would compete for the same host resources | Allow one active in-memory job |
| V2 recall risk | Search-space reduction can fail before fine validation | Preserve full-audio V1-style matching as the final fallback |

## 9. Engineering Takeaways

The implementation followed a simple sequence: build a correct end-to-end baseline, measure it, optimize the dominant ASR cost, and preserve the baseline as a fallback. Measurements, rather than assumptions, determined which layer was optimized next. V2 spends cheap computation on broad retrieval and expensive computation only where it improves validation or timing. Stable interfaces made model and architecture experiments inexpensive, while isolated Git branches/checkpoints kept working code recoverable and reduced migration risk.
