# AGENTS Notes

This file keeps advanced operational details that were removed from `README.md` to keep onboarding concise.

## Entry Points

- Interactive CLI:
  - `uv run cli`
  - Equivalent explicit path: `uv run src/voice_workbench.py`
- Direct pipeline:
  - `uv run src/pocket_tts_youtube_pipeline.py ...`

Use `uv run` for this repo's tools. Use `uvx` only for one-off utilities such as Hugging Face login.

## CLI Behavior Notes

- Main capabilities:
  - Clone voices from URLs (YouTube/Reddit)
  - Browse existing voices/snippets and generate new snippets
  - Upload `voice.safetensors` to Hugging Face
- Useful CLI options:
  - `--output-root runs`
  - `--dry-run`
  - `--verbose-pipeline`
  - `--device <name>`
- Device behavior:
  - One active device is used for clone/generate actions.
  - Known MPS mismatch failures (`Passed CPU tensor to MPS op`) are retried once on CPU.
- Hugging Face upload behavior:
  - Uploads only `voice.safetensors`.
  - Stores at `embeddings/<voice>-<version>.safetensors` in the target repo.
  - Embedding URL format: `hf://<owner>/<repo>/embeddings/<voice>-<version>.safetensors`

## Voice And Timestamp Rules

- Voice selector in generation mode:
  - `--voice stefan` => `stefan/1`
  - `--voice stefan-2` => `stefan/2`
- Voice selector in clone mode (`--source-url`):
  - Use a base name without `-` (for example `stefan`).
- Timestamp formats: `MM:SS` or `HH:MM:SS`.
- If only `--start` is provided, the pipeline uses a 30-second window (`start + 30s`).
- `--youtube-url` is accepted as a backward-compatible alias for `--source-url`.
- Prompt preprocessing defaults to 30-second truncation unless overridden.

## Artifacts And Caching

- Clone artifacts:
  - `runs/voice-clones/<voice>/<version>/voice_prompt.wav`
  - `runs/voice-clones/<voice>/<version>/voice.wav`
  - `runs/voice-clones/<voice>/<version>/voice.safetensors`
  - `runs/voice-clones/<voice>/<version>/run_manifest.json`
- Generated outputs:
  - `runs/voices/<voice>/<version>/<timestamp>/cloned_output.wav`
  - `runs/voices/<voice>/<version>/<timestamp>/run_manifest.json`
- Download cache:
  - `runs/downloads/source_<hash>.<ext>`
  - Reused across runs for the same source URL.

## Additional Examples

- Reuse a saved voice:
  - `uv run src/pocket_tts_youtube_pipeline.py --voice "stefan-1" --text "Hello again from my saved cloned voice." --device cpu`
- Batch mode TSV format:
  - `<source_url><TAB><text><TAB>[optional_run_name]<TAB>[optional_voice_name]`
  - Run with: `uv run src/pocket_tts_youtube_pipeline.py --jobs-tsv jobs/example_jobs.tsv`

## Troubleshooting

- If `yt-dlp` fails due to site changes:
  - `uv lock --upgrade-package yt-dlp`
  - `uv sync`
- If Pocket TTS cannot download clone weights:
  - Confirm model terms are accepted at `https://huggingface.co/kyutai/pocket-tts`
  - Run `uvx hf auth login`
