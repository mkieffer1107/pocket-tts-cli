# Pocket TTS Voice-Cloning Pipeline

This repo contains a generic pipeline script that:
1. Ingests source audio from either a source URL (YouTube/Reddit/etc.) or a local media file.
2. Extracts requested segments and converts the source to a mono WAV voice prompt (`ffmpeg`)
3. Saves reusable voice profiles in versioned folders (`voices/<voice>/<version>/`)
4. Runs Kyutai Pocket TTS voice cloning (`pocket-tts`)
5. Stores generation runs under each voice version (`voices/<voice>/<version>/runs/<timestamp>/`)

Downloaded/cached source media is stored in `media/downloads/`.

---

## Start Here

- **Web Viewer (recommended):** [Run the web viewer (clone + synthesize)](#web-viewer)
- **Python setup + pipeline:** [Step-by-step setup](#step-by-step-setup)
- **Interactive Python CLI:** [Option A: interactive CLI](#option-a-interactive-cli)
- **Direct pipeline command:** [Option B: direct clone job command](#option-b-direct-clone-job-command)

---

## Web Viewer

Recommended path for most users. Clone voices (from URL or local media), choose versions, and synthesize speech from the same UI.

```bash
cd site
npm install
npm run dev
```

Based on code for `jax-js` TTS: `https://jax-js.com/tts`.

---

## Prerequisites

- `uv`
- `ffmpeg`

This project is configured for Python 3.12 via UV (`.python-version`).

## Step-by-Step Setup

1. Create a local virtual environment with Python 3.12:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

2. Sync project dependencies from `pyproject.toml` / `uv.lock`:

```bash
uv sync
```

3. Verify tool versions from the UV environment:

```bash
uv run yt-dlp --version
uv run pocket-tts --help
```

4. Enable gated Pocket TTS voice-cloning access:
- Accept terms at `https://huggingface.co/kyutai/pocket-tts`
- Authenticate locally:

```bash
uvx hf auth login
```

5. Choose one way to run your first clone:

### Option A: interactive CLI

```bash
uv run cli
```

From the menu, choose `Clone a new voice from URL` and use:
- source URL: `https://www.youtube.com/watch?v=UF8uR6Z6KLc`
- voice: `stefan`
- start: `2:31`

### Option B: direct clone job command

```bash
uv run src/pocket_tts_youtube_pipeline.py \
  --source-url "https://www.youtube.com/watch?v=UF8uR6Z6KLc" \
  --voice "stefan" \
  --start "2:31" \
  --text "This line is synthesized in the cloned voice." \
  --device cpu
```

`--source-url` supports YouTube and Reddit media URLs (including `v.redd.it` HLS links).
Use `--source-file` to clone from a local MP3/WAV instead of downloading.

If you pass only `--start`, the script uses a 30-second window by default:

```bash
uv run src/pocket_tts_youtube_pipeline.py \
  --source-url "https://www.youtube.com/watch?v=UF8uR6Z6KLc" \
  --voice "stefan" \
  --start "2:31" \
  --text "This line is synthesized in the cloned voice" \
  --device cpu
```


Local file example:

`--source-file` supports local `.wav`, `.mp3`, `.m4a`, `.mp4`, `.webm`, `.opus`, `.aac`, and `.flac` files.
For video inputs like `.mp4`, the pipeline extracts the audio track with `ffmpeg`, applies `--start/--end` if provided, and uses that clipped audio as the voice prompt input.


```bash
uv run src/pocket_tts_youtube_pipeline.py \
  --source-file "<path_to_file>" \
  --voice "stefan" \
  --start "<start>" \
  --end "<end>" \
  --text "This line is synthesized in the cloned voice" \
  --device cpu

uv run src/pocket_tts_youtube_pipeline.py \
  --source-file "media/steve-vid.mp4" \
  --voice "stanford" \
  --start "00:08" \
  --end "00:14" \
  --text "This line is synthesized in the cloned voice" \
  --device cpu
```

This creates reusable voice files:
- `voices/stefan/1/voice.wav`
- `voices/stefan/1/voice.safetensors`
- `voices/stefan/1/runs/<timestamp>/cloned_output.wav` (when generation is enabled)

6. Reuse the saved voice in future runs:

`--voice "stefan"` defaults to version 1 (`stefan/1`):

```bash
uv run src/pocket_tts_youtube_pipeline.py \
  --voice "stefan" \
  --text "This is another line in the same cloned voice." \
  --device cpu
```

Use `--voice "<name>-<version>"` to pick a specific version, for example version 2:

```bash
uv run src/pocket_tts_youtube_pipeline.py \
  --voice "stefan-2" \
  --text "This line uses voice version 2." \
  --device cpu
```
