# Video Dialogue Localization

Given a video URL or local media and a target spoken dialogue, this project automatically locates the dialogue, refines its timestamp, determines the corresponding frame number, and extracts the frame as a JPEG image. The recommended path is a CPU-first V2 pipeline that combines coarse full-audio search with fine transcription and alignment on short candidate regions.

## Example Result

Target: `"My mind rebels at stagnation"`

```text
Timestamp   : 00:05:25.290
Frame       : 7799
Matched text: "My mind rebels at stagnation."
Similarity  : 1.0000
```

## How It Works

```mermaid
flowchart LR
    A[Video URL] --> B[Download]
    B --> C[Audio extraction]
    C --> D[Coarse faster-whisper tiny]
    D --> E[Top-K candidate regions]
    E --> F[Fine faster-whisper small]
    F --> G[RapidFuzz validation]
    G --> H[WhisperX forced alignment]
    H --> I[Global timestamp]
    I --> J[Frame extraction]
```

V1 transcribes the complete audio and remains available as the baseline and reference path. V2 reduces the expensive search space by re-transcribing only likely regions, then refines the winning onset. If coarse retrieval fails, V2 retries with `base` and finally falls back to full-audio `small` matching.

## Key Features

- End-to-end URL-to-frame localization
- CPU-first execution with an optimized faster-whisper provider
- V1 full-audio baseline and V2 coarse-to-fine pipeline
- RapidFuzz lexical matching with word-level timestamps and probabilities
- WhisperX forced alignment on only the winning V2 region
- Top-K candidate retrieval with `tiny` → `base` → full-`small` fallback
- Frame extraction using actual video FPS and ffprobe metadata
- Cached video and WAV reuse for repeated runs
- Command-line interface
- FastAPI web app with a vanilla HTML/CSS/JavaScript frontend
- Pluggable faster-whisper, OpenAI Whisper, and WhisperX backends in V1

## Performance

Recorded on the 3,261.781-second Sherlock input in the CPU environment used during development:

| Pipeline | ASR stages | Total | Result | Timestamp |
|---|---:|---:|---|---:|
| V1 faster-whisper `small` | Full ASR: 856.562 s | 1025.7 s | Exact match | 324.990 s |
| V2 `tiny` → `small` | Coarse: 135.984 s; fine: 27.313 s; alignment: 27.531 s | 199.5 s | Exact match | 325.290 s |

The recorded totals show an observed improvement of about 5.1×. Media-preparation conditions differed between these runs, so this is an engineering measurement rather than a controlled performance or formal accuracy study. See [Engineering Approach](approach.md) for the complete experiments and trade-offs.

## Requirements

- 64-bit CPython 3.12.x (tested configuration)
- Dependencies from `requirements.txt`
- `ffmpeg` and `ffprobe` available on `PATH` or under `tools/ffmpeg/bin`
- Network access on first use to download the selected model checkpoints
- CPU is the default and primary tested execution environment

WhisperX/TorchCodec uses FFmpeg shared libraries from the supported FFmpeg 4–7 range. See the notes in `requirements.txt` if TorchCodec reports a shared-library loading error.

## Installation

Windows PowerShell:

```powershell
git clone https://github.com/Jitesh-B-27/Quest1-Project.git
cd Quest1-Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Install FFmpeg separately and ensure both `ffmpeg` and `ffprobe` are on `PATH`, or place their binaries in `tools/ffmpeg/bin`.

## Running V2 (Recommended)

```powershell
python pipeline.py --arch v2 --url "<VIDEO_URL>" --target "<TARGET_DIALOGUE>"
```

Sherlock example:

```powershell
python pipeline.py --arch v2 --url "https://ok.ru/video/248244667877" --target "My mind rebels at stagnation"
```

Run `python pipeline.py --help` to view all supported options.

## Running V1

```powershell
python pipeline.py --arch v1 --url "<VIDEO_URL>" --target "<TARGET_DIALOGUE>"
```

V1 transcribes the entire audio and is mainly useful as a reference, fallback behavior, and direct ASR benchmarking path.

## Running Different ASR Models

V1 accepts the three providers registered by `asr/factory.py`:

```powershell
python pipeline.py --arch v1 --video-path "<VIDEO_PATH>" --audio-path "<AUDIO_PATH>" --target "My mind rebels at stagnation" --model-type faster-whisper --model tiny

python pipeline.py --arch v1 --video-path "<VIDEO_PATH>" --audio-path "<AUDIO_PATH>" --target "My mind rebels at stagnation" --model-type whisper --model tiny

python pipeline.py --arch v1 --video-path "<VIDEO_PATH>" --audio-path "<AUDIO_PATH>" --target "My mind rebels at stagnation" --model-type whisperx --model tiny
```

Model size is configurable; see `python pipeline.py --help`. V2 uses its configured faster-whisper coarse tiers and fine model, while V1 is the direct backend/model comparison path.

## Cached Media

Reuse both files to skip download and audio extraction:

```powershell
python pipeline.py --arch v2 --video-path "<VIDEO_PATH>" --audio-path "<WAV_PATH>" --target "<TARGET_DIALOGUE>"
```

Providing only `--video-path` skips downloading but still extracts audio. Each option also accepts a directory, from which the pipeline selects the first supported file in sorted order.

## Running the Web Interface

On Windows, run:

```powershell
.\run_server.bat
```

Equivalent direct command:

```powershell
uvicorn webapp.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000), enter a video URL and target dialogue, submit the job, and follow stage progress through the final frame and localization result.

## Output

```text
Timestamp : 00:05:25.290
Frame     : 7799
Text      : "My mind rebels at stagnation."
Similarity: 1.0000
Image     : frames/frame_7799.jpg
```

The combined result is written to `output/result.json`, and extracted images are stored in `frames/`. V2 results also report whether alignment was applied, the successful fallback tier, model metadata, and stage timings.

## Project Structure

```text
pipeline.py          V1/V2 orchestration and CLI
video_downloader.py  yt-dlp video acquisition
audio_extractor.py   FFmpeg WAV extraction
asr/                 ASR facade, providers, models, and aligner
matcher/             RapidFuzz dialogue matching
localizer/           V2 candidate localization and cascade
frame_extractor/     Metadata, frame mapping, and JPEG extraction
webapp/              FastAPI app, job store, and static frontend
tests/               Unit and orchestration tests
scope.md             Project boundaries and assumptions
architecture.md      Technical system design
approach.md          Engineering decisions and benchmarks
prompts.txt          Significant AI prompts
requirements.txt     Python dependencies and system notes
```

## Testing

The tests use Python's standard `unittest` framework:

```powershell
python -m unittest discover -s tests -v
```

## Documentation

- [Project Scope](scope.md)
- [Architecture](architecture.md)
- [Engineering Approach](approach.md)
- [AI Prompts](prompts.txt)

## AI-Assisted Development

AI tools were used for research, architecture exploration, implementation assistance, debugging, optimization analysis, testing, and documentation. Significant prompts are recorded in [`prompts.txt`](prompts.txt).

## Limitations

- Localizes audible speech, not text rendered in video frames or subtitles.
- Uses lexical/near-lexical matching rather than semantic paraphrase search.
- Requires downloadable or local, seekable media with usable audio and video.
- The local web interface stores jobs in memory and runs one job at a time.
- CPU inference remains the dominant cost for long media.

See [Project Scope](scope.md) for the complete boundaries and assumptions.
