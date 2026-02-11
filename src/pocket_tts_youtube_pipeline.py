#!/usr/bin/env python3
"""Source URL -> MP3 -> Pocket TTS voice-cloning pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    Console = None
    Panel = None
    Table = None
    Text = None

OUT_CONSOLE = Console() if Console is not None else None
ERR_CONSOLE = Console(stderr=True) if Console is not None else None


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".wav",
    ".mp4",
    ".webm",
    ".opus",
    ".aac",
    ".flac",
}

PREDEFINED_VOICES = {
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
}

VOICE_BASE_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
VOICE_SELECTOR_PATTERN = re.compile(r"^([A-Za-z0-9_]+)(?:-([1-9][0-9]*))?$")
START_ONLY_DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_DOWNLOAD_CACHE_DIR = Path("media") / "downloads"


@dataclass
class Job:
    source_url: str | None
    source_path: Path | None
    text: str | None
    run_name: str | None = None
    voice_name: str | None = None


@dataclass
class CommandFailure(Exception):
    cmd: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    def __str__(self) -> str:
        return (
            f"Command failed (exit {self.returncode}): {shlex.join(self.cmd)}\n"
            f"--- stdout ---\n{self.stdout}\n"
            f"--- stderr ---\n{self.stderr}"
        )


def console_print(
    message: str = "",
    *,
    stderr: bool = False,
    markup: bool = False,
    highlight: bool = False,
    end: str = "\n",
) -> None:
    target = ERR_CONSOLE if stderr else OUT_CONSOLE
    if target is not None:
        target.print(message, markup=markup, highlight=highlight, end=end)
        return
    stream = sys.stderr if stderr else sys.stdout
    print(message, file=stream, end=end, flush=True)


def log_success(message: str) -> None:
    if OUT_CONSOLE is not None:
        OUT_CONSOLE.print(message, style="bold green")
        return
    print(message, flush=True)


def log_warning(message: str) -> None:
    if ERR_CONSOLE is not None:
        ERR_CONSOLE.print(message, style="bold yellow")
        return
    print(message, file=sys.stderr, flush=True)


def log_error(message: str) -> None:
    if ERR_CONSOLE is not None:
        ERR_CONSOLE.print(message, style="bold red")
        return
    print(message, file=sys.stderr, flush=True)


def print_command(cmd: Sequence[str]) -> None:
    cmd_text = shlex.join(cmd)
    if OUT_CONSOLE is not None and Text is not None:
        command_text = Text("$ ", style="bold bright_black")
        command_text.append(cmd_text, style="bold cyan")
        OUT_CONSOLE.print(command_text)
        return
    print(f"$ {cmd_text}", flush=True)


def step_status(args: argparse.Namespace, message: str):
    if args.verbose_command_output or args.dry_run:
        return nullcontext()
    if OUT_CONSOLE is not None:
        return OUT_CONSOLE.status(f"[bold cyan]{message}[/]")
    print(message, flush=True)
    return nullcontext()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download source audio (YouTube/Reddit/etc.) and run voice cloning.",
    )
    parser.add_argument(
        "--source-url",
        help="Source URL to create/update a cloned voice profile from (YouTube, Reddit, etc.).",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        help=(
            "Local source media file (mp3/wav/mp4/etc.) to create/update "
            "a cloned voice profile from."
        ),
    )
    parser.add_argument(
        "--youtube-url",
        help="Deprecated alias for --source-url.",
    )
    parser.add_argument(
        "--text",
        help="Text to synthesize. Required unless --skip-generate is set.",
    )
    parser.add_argument(
        "--voice",
        help=(
            "Voice selector. With --source-url, use a base name like `goofy` to create "
            "the next version under voices/goofy/<n>. Without --source-url, "
            "use `goofy` (defaults to version 1), `goofy-1`, `goofy-2`, etc."
        ),
    )
    parser.add_argument(
        "--run-name",
        help="Optional label for generation run folder naming.",
    )
    parser.add_argument(
        "--jobs-tsv",
        type=Path,
        help=(
            "Batch mode: tab-separated file with columns "
            "<source_url> <text> [optional_run_name] [optional_voice_name]."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("voices"),
        help="Directory where voice/version folders are created. Default: voices",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Pocket TTS device argument. Examples: cpu, mps, cuda",
    )
    parser.add_argument(
        "--variant",
        help="Pocket TTS model signature override (advanced).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Pocket TTS generation temperature.",
    )
    parser.add_argument(
        "--lsd-decode-steps",
        type=int,
        help="Pocket TTS LSD decode steps.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Pocket TTS max tokens per chunk.",
    )
    parser.add_argument(
        "--noise-clamp",
        type=float,
        help="Pocket TTS noise clamp value.",
    )
    parser.add_argument(
        "--eos-threshold",
        type=float,
        help="Pocket TTS EOS threshold.",
    )
    parser.add_argument(
        "--frames-after-eos",
        type=int,
        help="Pocket TTS frames generated after EOS.",
    )
    parser.add_argument(
        "--trim-start-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional ffmpeg trim start offset for the voice prompt. "
            "Legacy option; --start/--end is preferred."
        ),
    )
    parser.add_argument(
        "--trim-duration-seconds",
        type=float,
        help=(
            "Optional ffmpeg trim duration for the voice prompt. "
            "If unset, --auto-truncate-seconds is used. "
            "Legacy option; --start/--end is preferred."
        ),
    )
    parser.add_argument(
        "--start",
        help=(
            "Clip start time for source cloning, e.g. 21:34 or 1:21:34. "
            "If set without --end, --end defaults to start+30s."
        ),
    )
    parser.add_argument(
        "--end",
        help="Clip end time for source cloning, e.g. 23:01 or 1:23:01.",
    )
    parser.add_argument(
        "--auto-truncate-seconds",
        type=float,
        default=30.0,
        help=(
            "Default max prompt length in seconds when --trim-duration-seconds is not set. "
            "Set to 0 to disable. Mirrors pocket-tts export-voice --truncate behavior."
        ),
    )
    parser.add_argument(
        "--prompt-sample-rate",
        type=int,
        default=24000,
        help="Sample rate for the voice prompt WAV. Default: 24000",
    )
    parser.add_argument(
        "--cookies-from-browser",
        help=(
            "Optional yt-dlp cookies-from-browser value (for restricted source access), "
            "e.g. chrome, firefox."
        ),
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        help="Optional yt-dlp cookies file path.",
    )
    parser.add_argument(
        "--force-ipv4",
        action="store_true",
        help="Pass --force-ipv4 to yt-dlp.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Stop after saving voice profile files (skip pocket-tts generation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned outputs without executing commands.",
    )
    parser.add_argument(
        "--verbose-command-output",
        action="store_true",
        help="Show raw command lines and full subprocess stdout/stderr.",
    )
    parser.add_argument(
        "--use-system-tools",
        action="store_true",
        help=(
            "Use system `yt-dlp` and `pocket-tts` binaries instead of "
            "running tools through `uv run`."
        ),
    )
    parser.add_argument(
        "--uv-no-sync",
        action="store_true",
        help=(
            "Pass --no-sync to `uv run` for faster execution when your `.venv` "
            "is already synced."
        ),
    )
    return parser


def parse_jobs(args: argparse.Namespace) -> list[Job]:
    if args.jobs_tsv and (args.start is not None or args.end is not None):
        raise ValueError("--start/--end are currently supported only in single-job mode.")

    source_url = args.source_url
    if args.youtube_url:
        if source_url and source_url != args.youtube_url:
            raise ValueError("Provide only one of --source-url or --youtube-url (or the same value).")
        source_url = args.youtube_url

    if args.source_file and source_url:
        raise ValueError("Provide only one of --source-url or --source-file.")

    if args.jobs_tsv:
        if source_url or args.source_file or args.text or args.run_name or args.voice:
            raise ValueError(
                "Use either --jobs-tsv or single-job flags "
                "(--source-url/--source-file/--voice/--text/--run-name), not both."
            )
        jobs = load_jobs_from_tsv(args.jobs_tsv)
    else:
        source_path = args.source_file
        if source_url and not source_path:
            candidate = Path(source_url).expanduser()
            if candidate.exists():
                source_path = candidate
                source_url = None
        jobs = [
            Job(
                source_url=source_url,
                source_path=source_path,
                text=args.text,
                run_name=args.run_name,
                voice_name=args.voice,
            )
        ]

    # Validate clip window syntax/order early, before any command execution.
    resolve_prompt_window(args)

    for job in jobs:
        if job.source_url and job.source_path:
            raise ValueError("Provide only one source per job (URL or local file).")
        if job.source_url and not job.source_path:
            candidate = Path(job.source_url).expanduser()
            if candidate.exists():
                job.source_path = candidate
                job.source_url = None

        if not job.source_url and not job.source_path and not job.voice_name:
            raise ValueError(
                "Provide either --source-url/--source-file (build voice from audio) "
                "or --voice (use an existing saved voice)."
            )
        if job.source_url or job.source_path:
            if not job.voice_name:
                raise ValueError(
                    "When using a source audio input, --voice is required "
                    "(base name without '-')."
                )
            validate_clone_voice_name(job.voice_name)
        elif job.voice_name:
            parse_voice_selector(job.voice_name)

        if args.skip_generate and not (job.source_url or job.source_path):
            raise ValueError("--skip-generate is only valid with a source input.")
        if not args.skip_generate and not job.text:
            raise ValueError("Generation requires --text.")
        if (args.start is not None or args.end is not None) and not (job.source_url or job.source_path):
            raise ValueError("--start/--end are only valid with source cloning runs.")
        if job.source_path:
            validate_source_file(job.source_path)
    return jobs


def load_jobs_from_tsv(path: Path) -> list[Job]:
    if not path.exists():
        raise FileNotFoundError(f"jobs TSV not found: {path}")

    jobs: list[Job] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 2 columns: "
                    "<source_url> <text> [optional_run_name] [optional_voice_name]"
                )

            source_url = row[0].strip() or None
            text = row[1].strip() or None
            run_name = row[2].strip() if len(row) > 2 and row[2].strip() else None
            voice_name = row[3].strip() if len(row) > 3 and row[3].strip() else None

            jobs.append(
                Job(
                    source_url=source_url,
                    source_path=None,
                    text=text,
                    run_name=run_name,
                    voice_name=voice_name,
                )
            )

    if not jobs:
        raise ValueError(f"No jobs found in {path}")
    return jobs


def ensure_dependencies(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    requires_source_audio = any(job.source_url or job.source_path for job in jobs)
    requires_download = any(job.source_url for job in jobs)

    required_tools: list[str] = []
    if requires_source_audio:
        required_tools.append("ffmpeg")

    if args.use_system_tools:
        if requires_download:
            required_tools.append("yt-dlp")
        if not args.skip_generate:
            required_tools.append("pocket-tts")
    else:
        required_tools.append("uv")
    missing = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Missing required commands: "
            + ", ".join(missing)
            + ". Install them and retry."
        )
    if args.use_system_tools and requires_download and importlib.util.find_spec("yt_dlp") is None:
        # This is only a hint for mixed-Python environments.
        log_warning(
            "Warning: `yt_dlp` module is not importable from this Python. "
            "If downloads fail, prefer UV mode (default) or run the script via `uv run`."
        )


def run_command(
    args: argparse.Namespace,
    cmd: Sequence[str],
    dry_run: bool,
    *,
    step_message: str | None = None,
) -> None:
    if args.verbose_command_output or dry_run:
        print_command(cmd)
    if dry_run:
        return

    status_cm = step_status(args, step_message) if step_message else nullcontext()
    with status_cm:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if args.verbose_command_output:
        if result.stdout:
            console_print(result.stdout, end="")
        if result.stderr:
            console_print(result.stderr, stderr=True, end="")
    if result.returncode != 0:
        raise CommandFailure(
            cmd=cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def sanitize_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = value.strip("-._")
    return value or "job"


def validate_clone_voice_name(voice_name: str) -> str:
    raw = voice_name.strip()
    if "-" in raw:
        raise ValueError(
            "Clone voice names cannot contain '-'. Use a base name like "
            "`--voice goofy` when creating a clone. To select a version at generation "
            "time, use `--voice goofy-1`, `--voice goofy-2`, etc."
        )
    if not VOICE_BASE_PATTERN.fullmatch(raw):
        raise ValueError(
            "Clone voice names must match [A-Za-z0-9_]+. "
            "Example: `--voice goofy_voice`."
        )
    return raw.lower()


def validate_source_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Source audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            "Unsupported source audio extension. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )


def parse_voice_selector(selector: str) -> tuple[str, int]:
    match = VOICE_SELECTOR_PATTERN.fullmatch(selector.strip())
    if not match:
        raise ValueError(
            "Invalid --voice selector. Use `goofy` (defaults to version 1) or "
            "`goofy-2`. Base names may only contain letters, numbers, and underscores."
        )
    base = match.group(1).lower()
    version = int(match.group(2)) if match.group(2) else 1
    return base, version


def parse_timecode_to_seconds(value: str) -> float:
    raw = value.strip()
    if raw == "":
        raise ValueError("Empty time value is not allowed.")

    if ":" not in raw:
        try:
            seconds = float(raw)
        except ValueError as error:
            raise ValueError(
                f"Invalid time value {value!r}. Use seconds or MM:SS / HH:MM:SS."
            ) from error
        if seconds < 0:
            raise ValueError(f"Invalid time value {value!r}. Time cannot be negative.")
        return seconds

    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Invalid time value {value!r}. Use MM:SS or HH:MM:SS."
        )

    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            hours = 0
        else:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
    except ValueError as error:
        raise ValueError(
            f"Invalid time value {value!r}. Use MM:SS or HH:MM:SS with numeric fields."
        ) from error

    if hours < 0 or minutes < 0 or seconds < 0:
        raise ValueError(f"Invalid time value {value!r}. Time cannot be negative.")
    if minutes >= 60 and len(parts) == 3:
        raise ValueError(
            f"Invalid time value {value!r}. In HH:MM:SS format, MM must be < 60."
        )
    if seconds >= 60:
        raise ValueError(
            f"Invalid time value {value!r}. SS must be < 60."
        )

    return hours * 3600 + minutes * 60 + seconds


def resolve_prompt_window(args: argparse.Namespace) -> tuple[float, float | None]:
    using_time_window = args.start is not None or args.end is not None
    using_trim_window = args.trim_start_seconds > 0 or args.trim_duration_seconds is not None

    if using_time_window and using_trim_window:
        raise ValueError(
            "Use either --start/--end or --trim-start-seconds/--trim-duration-seconds, not both."
        )

    if using_time_window:
        start_seconds = parse_timecode_to_seconds(args.start) if args.start else 0.0
        if args.end:
            end_seconds = parse_timecode_to_seconds(args.end)
            if end_seconds <= start_seconds:
                raise ValueError(
                    f"Invalid clip window: --end ({args.end}) must be greater than "
                    f"--start ({args.start or '0'})."
                )
            return start_seconds, end_seconds - start_seconds
        if args.start:
            return start_seconds, START_ONLY_DEFAULT_WINDOW_SECONDS
        return start_seconds, None

    prompt_duration = args.trim_duration_seconds
    if prompt_duration is None and args.auto_truncate_seconds > 0:
        prompt_duration = args.auto_truncate_seconds
    return args.trim_start_seconds, prompt_duration


def clone_version_dir(output_root: Path, voice_base: str, version: int) -> Path:
    return output_root / voice_base / str(version)


def next_clone_version(output_root: Path, voice_base: str) -> int:
    base_dir = output_root / voice_base
    max_version = 0
    if base_dir.exists():
        for path in base_dir.iterdir():
            if path.is_dir() and path.name.isdigit():
                max_version = max(max_version, int(path.name))
    return max_version + 1


def available_clone_versions(output_root: Path, voice_base: str) -> list[int]:
    base_dir = output_root / voice_base
    versions: list[int] = []
    if base_dir.exists():
        for path in base_dir.iterdir():
            if path.is_dir() and path.name.isdigit():
                versions.append(int(path.name))
    return sorted(versions)


def generation_run_dir(
    output_root: Path, voice_base: str, version: int, run_name: str | None
) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    leaf = f"{sanitize_slug(run_name)}_{timestamp}" if run_name else timestamp
    return clone_version_dir(output_root, voice_base, version) / "runs" / leaf


def print_job_header(
    ordinal: int,
    total: int,
    job: Job,
    voice_profile_name: str,
    voice_profile_version: int,
    run_dir: Path,
    generation_dir: Path | None,
) -> None:
    if job.source_path:
        mode = "file->voice"
    else:
        mode = "source->voice" if job.source_url else "voice->tts"
    if OUT_CONSOLE is not None and Panel is not None and Table is not None:
        details = Table.grid(padding=(0, 1))
        details.add_row("[bold bright_magenta]Job[/]", f"[bold]{ordinal}/{total}[/]")
        details.add_row("[bold bright_magenta]Mode[/]", f"[bold cyan]{mode}[/]")
        if job.source_url:
            details.add_row("[bold bright_magenta]Source URL[/]", f"[deep_sky_blue1]{job.source_url}[/]")
        if job.source_path:
            details.add_row("[bold bright_magenta]Source file[/]", f"[deep_sky_blue1]{job.source_path}[/]")
        details.add_row(
            "[bold bright_magenta]Voice profile[/]",
            f"[bold yellow]{voice_profile_name}/{voice_profile_version}[/]",
        )
        if generation_dir is not None and generation_dir != run_dir:
            details.add_row("[bold bright_magenta]Clone dir[/]", f"[dim]{run_dir}[/]")
            details.add_row("[bold bright_magenta]Generation dir[/]", f"[dim]{generation_dir}[/]")
        else:
            details.add_row("[bold bright_magenta]Output dir[/]", f"[dim]{run_dir}[/]")
        OUT_CONSOLE.print()
        OUT_CONSOLE.print(
            Panel.fit(
                details,
                title="[bold white]Pocket TTS Pipeline[/]",
                border_style="bright_blue",
            )
        )
        return

    print("\n" + "=" * 80, flush=True)
    print(f"Job {ordinal}/{total}", flush=True)
    print(f"Mode: {mode}", flush=True)
    if job.source_url:
        print(f"Source URL: {job.source_url}", flush=True)
    if job.source_path:
        print(f"Source file: {job.source_path}", flush=True)
    print(f"Voice profile: {voice_profile_name}/{voice_profile_version}", flush=True)
    if generation_dir is not None and generation_dir != run_dir:
        print(f"Clone dir: {run_dir}", flush=True)
        print(f"Generation dir: {generation_dir}", flush=True)
    else:
        print(f"Output dir: {run_dir}", flush=True)
    print("=" * 80, flush=True)


def print_completion_summary(completed: Sequence[Path]) -> None:
    if OUT_CONSOLE is not None and Panel is not None and Table is not None:
        details = Table.grid(padding=(0, 1))
        for path in completed:
            details.add_row("[green]-[/]", f"[bold green]{path}[/]")
        OUT_CONSOLE.print()
        OUT_CONSOLE.print(
            Panel.fit(
                details,
                title="[bold green]All Jobs Completed[/]",
                border_style="green",
            )
        )
        return

    print("\nAll jobs completed.", flush=True)
    for path in completed:
        print(f"- {path}", flush=True)


def download_cache_stem(source_url: str) -> str:
    cache_key = hashlib.sha256(source_url.strip().encode("utf-8")).hexdigest()[:20]
    return f"source_{cache_key}"


def legacy_youtube_cache_stem(source_url: str) -> str:
    cache_key = hashlib.sha256(source_url.strip().encode("utf-8")).hexdigest()[:20]
    return f"youtube_{cache_key}"


def find_downloaded_audio(
    search_dir: Path,
    stem: str = "source",
    *,
    required: bool = True,
) -> Path | None:
    pattern = f"{stem}.*"
    candidates = sorted(
        p
        for p in search_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    if not candidates:
        if required:
            raise FileNotFoundError(
                f"Could not find downloaded source audio file matching {pattern}"
            )
        return None
    return candidates[0]


def voice_profile_paths_for_clone_dir(clone_dir: Path) -> tuple[Path, Path]:
    return clone_dir / "voice.wav", clone_dir / "voice.safetensors"


def export_voice_safetensors(
    args: argparse.Namespace,
    voice_prompt_wav: Path,
    output_safetensors: Path,
) -> None:
    if args.verbose_command_output or args.dry_run:
        print_command(["export-voice-embedding", str(voice_prompt_wav), str(output_safetensors)])
    if args.dry_run:
        return

    try:
        import safetensors.torch
        import torch
        from pocket_tts import TTSModel
        from pocket_tts.data.audio import audio_read
        from pocket_tts.data.audio_utils import convert_audio
        from pocket_tts.default_parameters import (
            DEFAULT_EOS_THRESHOLD,
            DEFAULT_LSD_DECODE_STEPS,
            DEFAULT_NOISE_CLAMP,
            DEFAULT_TEMPERATURE,
            DEFAULT_VARIANT,
        )
    except ImportError as error:
        raise RuntimeError(
            "Failed to import pocket-tts/safetensors for voice export. "
            "Run this script with `uv run ...` after `uv sync`."
        ) from error

    with step_status(args, "Exporting voice embedding..."):
        variant = args.variant or DEFAULT_VARIANT
        temperature = args.temperature if args.temperature is not None else DEFAULT_TEMPERATURE
        lsd_decode_steps = (
            args.lsd_decode_steps
            if args.lsd_decode_steps is not None
            else DEFAULT_LSD_DECODE_STEPS
        )
        noise_clamp = args.noise_clamp if args.noise_clamp is not None else DEFAULT_NOISE_CLAMP
        eos_threshold = args.eos_threshold if args.eos_threshold is not None else DEFAULT_EOS_THRESHOLD

        tts_model = TTSModel.load_model(
            variant=variant,
            temp=temperature,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
        )
        tts_model.to(args.device)

        audio, sample_rate = audio_read(voice_prompt_wav)
        if args.auto_truncate_seconds > 0:
            max_samples = int(sample_rate * args.auto_truncate_seconds)
            if audio.shape[-1] > max_samples:
                audio = audio[..., :max_samples]

        audio_conditioning = convert_audio(audio, sample_rate, tts_model.sample_rate, 1)
        with torch.no_grad():
            prompt = tts_model._encode_audio(audio_conditioning.unsqueeze(0).to(tts_model.device))

        output_safetensors.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file({"audio_prompt": prompt.cpu()}, str(output_safetensors))


def tool_prefix(args: argparse.Namespace) -> list[str]:
    if args.use_system_tools:
        return []
    prefix = ["uv", "run"]
    if args.uv_no_sync:
        prefix.append("--no-sync")
    return prefix


def build_yt_dlp_command(
    args: argparse.Namespace,
    job: Job,
    output_stem_path: Path,
) -> list[str]:
    if not job.source_url:
        raise ValueError("Internal error: build_yt_dlp_command requires job.source_url.")
    cmd = tool_prefix(args) + [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
    ]
    if args.force_ipv4:
        cmd.append("--force-ipv4")
    if args.cookies_from_browser:
        cmd.extend(["--cookies-from-browser", args.cookies_from_browser])
    if args.cookies_file:
        cmd.extend(["--cookies", str(args.cookies_file)])
    cmd.extend(
        [
            "--output",
            f"{output_stem_path}.%(ext)s",
            job.source_url,
        ]
    )
    return cmd


def build_ffmpeg_command(
    args: argparse.Namespace, source_audio: Path, voice_prompt_wav: Path
) -> list[str]:
    start_seconds, prompt_duration = resolve_prompt_window(args)

    cmd = ["ffmpeg", "-y"]
    if start_seconds > 0:
        cmd.extend(["-ss", str(start_seconds)])
    cmd.extend(["-i", str(source_audio)])
    if prompt_duration is not None:
        cmd.extend(["-t", str(prompt_duration)])
    cmd.extend(
        [
            "-ac",
            "1",
            "-ar",
            str(args.prompt_sample_rate),
            "-vn",
            "-sn",
            "-dn",
            str(voice_prompt_wav),
        ]
    )
    return cmd


def build_pocket_tts_command(
    args: argparse.Namespace,
    text: str,
    voice_reference: str | Path,
    output_wav: Path,
) -> list[str]:
    cmd = tool_prefix(args) + [
        "pocket-tts",
        "generate",
        "--text",
        text,
        "--voice",
        str(voice_reference),
        "--output-path",
        str(output_wav),
        "--device",
        args.device,
    ]
    if not args.verbose_command_output:
        cmd.append("--quiet")
    if args.variant:
        cmd.extend(["--variant", args.variant])
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.lsd_decode_steps is not None:
        cmd.extend(["--lsd-decode-steps", str(args.lsd_decode_steps)])
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.noise_clamp is not None:
        cmd.extend(["--noise-clamp", str(args.noise_clamp)])
    if args.eos_threshold is not None:
        cmd.extend(["--eos-threshold", str(args.eos_threshold)])
    if args.frames_after_eos is not None:
        cmd.extend(["--frames-after-eos", str(args.frames_after_eos)])
    return cmd


def write_manifest(
    run_dir: Path,
    job: Job,
    source_audio: Path | None,
    voice_prompt_wav: Path | None,
    voice_profile_name: str | None,
    voice_profile_version: int | None,
    voice_profile_wav: Path | None,
    voice_profile_safetensors: Path | None,
    voice_reference: str | Path | None,
    output_wav: Path | None,
    args: argparse.Namespace,
) -> None:
    clip_start_seconds, clip_duration_seconds = resolve_prompt_window(args)

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_url": job.source_url,
        "source_path": str(job.source_path) if job.source_path else None,
        "text": job.text,
        "run_name": job.run_name,
        "voice_profile_name": voice_profile_name,
        "voice_profile_version": voice_profile_version,
        "artifacts": {
            "source_audio": str(source_audio) if source_audio else None,
            "voice_prompt_wav": str(voice_prompt_wav) if voice_prompt_wav else None,
            "voice_profile_wav": str(voice_profile_wav) if voice_profile_wav else None,
            "voice_profile_safetensors": (
                str(voice_profile_safetensors) if voice_profile_safetensors else None
            ),
            "voice_reference_used": str(voice_reference) if voice_reference else None,
            "cloned_output_wav": str(output_wav) if output_wav else None,
        },
        "options": {
            "voice": args.voice,
            "device": args.device,
            "variant": args.variant,
            "temperature": args.temperature,
            "lsd_decode_steps": args.lsd_decode_steps,
            "max_tokens": args.max_tokens,
            "noise_clamp": args.noise_clamp,
            "eos_threshold": args.eos_threshold,
            "frames_after_eos": args.frames_after_eos,
            "trim_start_seconds": args.trim_start_seconds,
            "trim_duration_seconds": args.trim_duration_seconds,
            "start": args.start,
            "end": args.end,
            "resolved_clip_start_seconds": clip_start_seconds,
            "resolved_clip_duration_seconds": clip_duration_seconds,
            "auto_truncate_seconds": args.auto_truncate_seconds,
            "prompt_sample_rate": args.prompt_sample_rate,
            "cookies_from_browser": args.cookies_from_browser,
            "cookies_file": str(args.cookies_file) if args.cookies_file else None,
            "force_ipv4": args.force_ipv4,
            "skip_generate": args.skip_generate,
        },
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def run_job(args: argparse.Namespace, job: Job, ordinal: int, total: int) -> Path:
    if not job.voice_name:
        raise ValueError("Internal error: voice_name is required.")

    if job.source_url or job.source_path:
        voice_profile_name = validate_clone_voice_name(job.voice_name)
        voice_profile_version = next_clone_version(args.output_root, voice_profile_name)
        run_dir = clone_version_dir(args.output_root, voice_profile_name, voice_profile_version)
        generation_dir = (
            generation_run_dir(
                args.output_root,
                voice_profile_name,
                voice_profile_version,
                job.run_name,
            )
            if not args.skip_generate
            else None
        )
    else:
        voice_profile_name, voice_profile_version = parse_voice_selector(job.voice_name)
        run_dir = generation_run_dir(
            args.output_root,
            voice_profile_name,
            voice_profile_version,
            job.run_name,
        )
        generation_dir = run_dir

    cloned_output_wav = generation_dir / "cloned_output.wav" if generation_dir else None
    clone_dir_for_profile = clone_version_dir(
        args.output_root,
        voice_profile_name,
        voice_profile_version,
    )
    voice_profile_wav, voice_profile_safetensors = voice_profile_paths_for_clone_dir(
        clone_dir_for_profile
    )

    print_job_header(
        ordinal=ordinal,
        total=total,
        job=job,
        voice_profile_name=voice_profile_name,
        voice_profile_version=voice_profile_version,
        run_dir=run_dir,
        generation_dir=generation_dir,
    )

    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=False)
        if generation_dir is not None and generation_dir != run_dir:
            generation_dir.mkdir(parents=True, exist_ok=False)

    source_audio: Path | None = None
    voice_prompt_wav: Path | None = None
    voice_reference: str | Path | None = None

    if job.source_url or job.source_path:
        voice_prompt_wav = run_dir / "voice_prompt.wav"
        if job.source_path:
            source_audio = job.source_path
        else:
            cached_source_stem = download_cache_stem(job.source_url)
            cached_download_dir = DEFAULT_DOWNLOAD_CACHE_DIR
            source_audio = find_downloaded_audio(
                cached_download_dir,
                stem=cached_source_stem,
                required=False,
            )
            if source_audio is None:
                legacy_stem = legacy_youtube_cache_stem(job.source_url)
                source_audio = find_downloaded_audio(
                    cached_download_dir,
                    stem=legacy_stem,
                    required=False,
                )

            if source_audio is not None:
                log_success(f"Reusing cached source audio: {source_audio}")
            else:
                if not args.dry_run:
                    cached_download_dir.mkdir(parents=True, exist_ok=True)
                run_command(
                    args,
                    build_yt_dlp_command(
                        args,
                        job,
                        cached_download_dir / cached_source_stem,
                    ),
                    dry_run=args.dry_run,
                    step_message="Downloading source audio...",
                )
                source_audio = (
                    cached_download_dir / f"{cached_source_stem}.mp3"
                    if args.dry_run
                    else find_downloaded_audio(cached_download_dir, stem=cached_source_stem)
                )

        run_command(
            args,
            build_ffmpeg_command(args, source_audio, voice_prompt_wav),
            dry_run=args.dry_run,
            step_message="Extracting voice prompt clip...",
        )

        if args.dry_run:
            print_command(["cp", str(voice_prompt_wav), str(voice_profile_wav)])
        else:
            clone_dir_for_profile.mkdir(parents=True, exist_ok=True)
            shutil.copy2(voice_prompt_wav, voice_profile_wav)
        export_voice_safetensors(args, voice_prompt_wav, voice_profile_safetensors)
        log_success(f"Saved voice WAV profile: {voice_profile_wav}")
        log_success(f"Saved voice safetensors profile: {voice_profile_safetensors}")
        voice_reference = voice_profile_wav

    else:
        if args.dry_run:
            voice_reference = voice_profile_wav
        elif voice_profile_wav.exists():
            voice_reference = voice_profile_wav
        elif voice_profile_name in PREDEFINED_VOICES and voice_profile_version == 1:
            voice_reference = voice_profile_name
        elif voice_profile_safetensors.exists():
            raise RuntimeError(
                f"Found {voice_profile_safetensors} but no WAV voice profile at {voice_profile_wav}. "
                "Current pipeline generation path expects a WAV profile. Rebuild from a source URL "
                "or restore the WAV profile."
            )
        else:
            versions = available_clone_versions(args.output_root, voice_profile_name)
            versions_hint = ", ".join(str(v) for v in versions) if versions else "none"
            raise FileNotFoundError(
                f"Voice profile not found for --voice {job.voice_name!r}. "
                f"Expected {voice_profile_wav}. "
                f"Available versions for {voice_profile_name!r}: {versions_hint}."
            )

    output_path: Path | None = None
    if args.skip_generate:
        log_warning("Skipping pocket-tts generation (--skip-generate).")
    else:
        if not job.text:
            raise ValueError("Generation requires text.")
        if voice_reference is None:
            raise RuntimeError("Internal error: missing voice reference for generation.")
        if cloned_output_wav is None:
            raise RuntimeError("Internal error: missing generation output path.")

        run_command(
            args,
            build_pocket_tts_command(args, job.text, voice_reference, cloned_output_wav),
            dry_run=args.dry_run,
            step_message="Generating cloned snippet...",
        )
        output_path = cloned_output_wav

    if not args.dry_run:
        write_manifest(
            run_dir=run_dir,
            job=job,
            source_audio=source_audio,
            voice_prompt_wav=voice_prompt_wav,
            voice_profile_name=voice_profile_name,
            voice_profile_version=voice_profile_version,
            voice_profile_wav=voice_profile_wav,
            voice_profile_safetensors=voice_profile_safetensors,
            voice_reference=voice_reference,
            output_wav=output_path,
            args=args,
        )
        if generation_dir is not None and generation_dir != run_dir:
            write_manifest(
                run_dir=generation_dir,
                job=job,
                source_audio=source_audio,
                voice_prompt_wav=voice_prompt_wav,
                voice_profile_name=voice_profile_name,
                voice_profile_version=voice_profile_version,
                voice_profile_wav=voice_profile_wav,
                voice_profile_safetensors=voice_profile_safetensors,
                voice_reference=voice_reference,
                output_wav=output_path,
                args=args,
            )
    return generation_dir if generation_dir is not None else run_dir


def print_failure_hints(error: CommandFailure) -> None:
    stderr = (error.stderr or "") + "\n" + (error.stdout or "")
    cmd_text = " ".join(error.cmd)

    if "yt-dlp" in cmd_text:
        log_warning(
            "\nHint: yt-dlp failures are often fixed by updating yt-dlp and/or "
            "providing cookies. If using UV, run `uv lock --upgrade-package yt-dlp && uv sync` "
            "then retry with `--cookies-from-browser` or `--cookies-file`."
        )
    if "We could not download the weights for the model with voice cloning" in stderr:
        log_warning(
            "\nHint: voice cloning requires gated model access. Accept terms at "
            "https://huggingface.co/kyutai/pocket-tts, then run `uvx hf auth login`."
        )
    if "The expanded size of the tensor (1000)" in stderr:
        log_warning(
            "\nHint: the voice prompt is too long for the model context. "
            "Use a shorter reference (for example 30s), or pass "
            "`--trim-duration-seconds 30` / `--auto-truncate-seconds 30`. "
            "This matches pocket-tts export-voice `--truncate` guidance: "
            "https://github.com/kyutai-labs/pocket-tts/blob/main/docs/export_voice.md"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        jobs = parse_jobs(args)
        ensure_dependencies(args, jobs)

        if not args.dry_run:
            args.output_root.mkdir(parents=True, exist_ok=True)

        completed: list[Path] = []
        for index, job in enumerate(jobs, start=1):
            completed.append(run_job(args, job, index, len(jobs)))

    except (ValueError, FileNotFoundError, RuntimeError) as error:
        log_error(f"ERROR: {error}")
        return 2
    except CommandFailure as error:
        log_error(f"\nERROR: {error}")
        print_failure_hints(error)
        return error.returncode

    print_completion_summary(completed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
