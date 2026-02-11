# Pocket TTS URL Voice-Cloning Pipeline

This repo contains a generic pipeline script that:
1. Downloads audio from a source URL (YouTube/Reddit/etc.) as MP3 (`yt-dlp`)
2. Converts it to a mono WAV voice prompt (`ffmpeg`)
3. Saves reusable voice profiles in versioned folders (`runs/voice-clones/<voice>/<version>/`)
4. Runs Kyutai Pocket TTS voice cloning (`pocket-tts`)

Script: `src/pocket_tts_youtube_pipeline.py`

Interactive helper: `src/voice_workbench.py`

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

Option A: interactive CLI

```bash
uv run cli
```

From the menu, choose `Clone a new voice from URL` and use:
- source URL: `https://www.youtube.com/watch?v=UF8uR6Z6KLc`
- voice: `stefan`
- start: `2:31`

Option B: direct clone job command

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
- `runs/voice-clones/stefan/1/voice.wav`
- `runs/voice-clones/stefan/1/voice.safetensors`

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
