#!/usr/bin/env python3
"""Interactive CLI for cloning voices, generating snippets, and publishing to Hugging Face."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.table import Table
except ImportError as error:
    raise SystemExit(
        "This script requires `rich`. Run via `uv run scripts/voice_workbench.py` "
        "after `uv sync`."
    ) from error


VOICE_BASE_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
DEFAULT_SAMPLE_TEXT = "This line is synthesized in the cloned voice."
PATH_VALUE_FLAGS = {
    "--output-path",
    "--output-root",
    "--voice",
    "--source-url",
    "--pipeline-script",
    "--cookies",
    "--cookies-file",
    "--jobs-tsv",
}


@dataclass(frozen=True)
class VoiceProfile:
    base: str
    version: int
    directory: Path
    voice_wav: Path
    voice_safetensors: Path
    manifest: Path

    @property
    def selector(self) -> str:
        return f"{self.base}-{self.version}"


@dataclass(frozen=True)
class VoiceSnippet:
    run_dir: Path
    manifest: Path | None
    created_at: str
    run_name: str | None
    text: str
    output_wav: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guided CLI for this project: clone voices from URLs (YouTube/Reddit), generate snippets "
            "with saved voices, and upload voices to Hugging Face."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Pipeline output root. Default: runs",
    )
    parser.add_argument(
        "--pipeline-script",
        type=Path,
        default=Path(__file__).resolve().with_name("pocket_tts_youtube_pipeline.py"),
        help="Path to pocket_tts_youtube_pipeline.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without running pipeline commands or uploading to HF.",
    )
    parser.add_argument(
        "--verbose-pipeline",
        action="store_true",
        help="Show full pipeline stdout/stderr instead of quiet mode.",
    )
    parser.add_argument(
        "--device",
        help="Initial active device (for example: cpu, mps, cuda). If omitted, auto-detected.",
    )
    return parser.parse_args()


def resolve_pipeline_script(path: Path) -> Path:
    candidate = path if path.is_absolute() else (Path.cwd() / path)
    if not candidate.exists():
        raise FileNotFoundError(f"Pipeline script not found: {candidate}")
    return candidate


def detect_device_capabilities() -> dict[str, bool]:
    capabilities = {"cpu": True, "mps": False, "cuda": False}
    try:
        import torch
    except ImportError:
        return capabilities

    try:
        capabilities["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        mps_backend = getattr(torch.backends, "mps", None)
        capabilities["mps"] = bool(mps_backend and mps_backend.is_available())
    except Exception:
        pass
    return capabilities


def detect_default_device() -> str:
    capabilities = detect_device_capabilities()
    if capabilities.get("cuda"):
        return "cuda"
    # Pocket TTS is CPU-first and MPS can be unstable across torch/macOS versions.
    # Keep CPU as the default on Apple Silicon; users can still opt into MPS.
    return "cpu"


def discover_voice_profiles(output_root: Path) -> list[VoiceProfile]:
    root = output_root / "voice-clones"
    if not root.exists():
        return []

    profiles: list[VoiceProfile] = []
    for base_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        for version_dir in sorted(
            (p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ):
            voice_wav = version_dir / "voice.wav"
            voice_safetensors = version_dir / "voice.safetensors"
            manifest = version_dir / "run_manifest.json"
            if not voice_wav.exists() and not voice_safetensors.exists():
                continue
            profiles.append(
                VoiceProfile(
                    base=base_dir.name,
                    version=int(version_dir.name),
                    directory=version_dir,
                    voice_wav=voice_wav,
                    voice_safetensors=voice_safetensors,
                    manifest=manifest,
                )
            )
    return profiles


def print_banner(console: Console) -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Voice Workbench[/]\n"
            "[dim]Clone voices, synthesize snippets, and publish to Hugging Face.[/]",
            border_style="bright_blue",
        )
    )


def print_main_menu(console: Console, active_device: str, auto_device: str) -> None:
    table = Table(
        title="Actions",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("#", style="bold cyan", width=4, justify="right")
    table.add_column("Action", style="white")
    table.add_row("1", "Clone a new voice from URL")
    table.add_row("2", "See existing voices (browse or generate)")
    table.add_row("3", "Push an existing voice to Hugging Face")
    table.add_row(
        "4",
        f"Set active device (current: {active_device}, auto-detected: {auto_device})",
    )
    table.add_row("5", "Exit")
    console.print(table)


def set_active_device_flow(console: Console, current_device: str) -> str:
    capabilities = detect_device_capabilities()
    auto_device = detect_default_device()

    table = Table(
        title="Set Active Device",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("#", style="bold cyan", width=4, justify="right")
    table.add_column("Choice", style="white")
    table.add_row("1", f"Use auto-detected ({auto_device})")
    table.add_row("2", "cpu")
    table.add_row("3", f"mps ({'available' if capabilities.get('mps') else 'unavailable'})")
    table.add_row("4", f"cuda ({'available' if capabilities.get('cuda') else 'unavailable'})")
    table.add_row("5", "Custom device string")
    table.add_row("6", "Cancel")
    console.print(table)

    choice = IntPrompt.ask("Choose option", choices=["1", "2", "3", "4", "5", "6"], default="1")
    if choice == 6:
        return current_device

    if choice == 1:
        new_device = auto_device
    elif choice == 2:
        new_device = "cpu"
    elif choice == 3:
        new_device = "mps"
    elif choice == 4:
        new_device = "cuda"
    else:
        new_device = Prompt.ask("Custom device", default=current_device).strip()
        if new_device == "":
            console.print("Device unchanged.", style="bold yellow")
            return current_device

    if new_device in ("mps", "cuda") and not capabilities.get(new_device, False):
        console.print(
            f"Warning: {new_device} is not currently available on this machine.",
            style="bold yellow",
        )
    console.print(f"Active device set to: {new_device}", style="bold green")
    return new_device


def prompt_voice_base(console: Console) -> str:
    while True:
        raw = Prompt.ask("Voice base name ([A-Za-z0-9_])").strip().lower()
        if VOICE_BASE_PATTERN.fullmatch(raw):
            return raw
        console.print(
            "Invalid voice base name. Use only letters, numbers, and underscores.",
            style="bold red",
        )


def prompt_optional(text: str) -> str | None:
    value = Prompt.ask(text, default="").strip()
    return value or None


def select_voice_profile(console: Console, output_root: Path) -> VoiceProfile | None:
    profiles = discover_voice_profiles(output_root)
    if not profiles:
        console.print(
            f"No voice profiles found under {output_root / 'voice-clones'}.",
            style="bold yellow",
        )
        return None

    table = Table(
        title="Available Voice Profiles",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("#", style="bold cyan", width=4, justify="right")
    table.add_column("Voice")
    table.add_column("Version", justify="right")
    table.add_column("WAV", justify="center")
    table.add_column("Safetensors", justify="center")
    table.add_column("Directory", style="dim")
    for index, profile in enumerate(profiles, start=1):
        table.add_row(
            str(index),
            profile.base,
            str(profile.version),
            "yes" if profile.voice_wav.exists() else "no",
            "yes" if profile.voice_safetensors.exists() else "no",
            str(profile.directory),
        )
    console.print(table)

    while True:
        choice = IntPrompt.ask(
            "Choose profile number",
            default=1,
        )
        if 1 <= choice <= len(profiles):
            return profiles[choice - 1]
        console.print(f"Enter a number between 1 and {len(profiles)}.", style="bold red")


def discover_voice_snippets(output_root: Path, profile: VoiceProfile) -> list[VoiceSnippet]:
    snippets_root = output_root / "voices" / profile.base / str(profile.version)
    if not snippets_root.exists():
        return []

    snippets: list[VoiceSnippet] = []
    run_dirs = sorted(
        (path for path in snippets_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        manifest_data: dict[str, object] | None = None
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest_data = json.load(handle)
            except (json.JSONDecodeError, OSError):
                manifest_data = None

        created_at = (
            str(manifest_data.get("created_at"))
            if manifest_data and manifest_data.get("created_at")
            else run_dir.name
        )
        run_name_value = (
            str(manifest_data.get("run_name"))
            if manifest_data and manifest_data.get("run_name")
            else None
        )
        text_value = (
            str(manifest_data.get("text"))
            if manifest_data and manifest_data.get("text")
            else ""
        )

        output_wav: Path | None = None
        if manifest_data:
            artifacts = manifest_data.get("artifacts")
            if isinstance(artifacts, dict):
                cloned_output = artifacts.get("cloned_output_wav")
                if isinstance(cloned_output, str) and cloned_output.strip():
                    candidate = Path(cloned_output)
                    output_wav = candidate if candidate.is_absolute() else Path.cwd() / candidate

        if output_wav is None:
            fallback = run_dir / "cloned_output.wav"
            output_wav = fallback if fallback.exists() else None

        snippets.append(
            VoiceSnippet(
                run_dir=run_dir,
                manifest=manifest_path if manifest_path.exists() else None,
                created_at=created_at,
                run_name=run_name_value,
                text=text_value,
                output_wav=output_wav,
            )
        )
    return snippets


def build_audio_play_command(audio_path: Path) -> list[str] | None:
    if shutil.which("afplay"):
        return ["afplay", str(audio_path)]
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(audio_path)]
    if shutil.which("mpv"):
        return ["mpv", "--really-quiet", "--no-video", str(audio_path)]
    if shutil.which("play"):
        return ["play", "-q", str(audio_path)]
    if shutil.which("aplay"):
        return ["aplay", str(audio_path)]
    return None


def play_audio_path(console: Console, audio_path: Path, dry_run: bool) -> bool:
    if not audio_path.exists():
        console.print(f"Audio file missing: {audio_path}", style="bold red")
        return False

    command = build_audio_play_command(audio_path)
    if command is None:
        console.print(
            "No audio player found (tried afplay, ffplay, mpv, play, aplay).",
            style="bold yellow",
        )
        return False

    console.print(f"Playing: {audio_path}", style="bold green")
    if dry_run:
        console.print(f"Dry run: {shlex.join(command)}", style="bold yellow")
        return True

    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        console.print(
            f"Audio player exited with code {result.returncode}.",
            style="bold red",
        )
        return False
    return True


def play_snippet_audio(console: Console, snippet: VoiceSnippet, dry_run: bool) -> None:
    if snippet.output_wav is None:
        console.print("No output audio file recorded for this snippet.", style="bold red")
        return

    play_audio_path(console, snippet.output_wav, dry_run=dry_run)


def browse_snippets_flow(console: Console, args: argparse.Namespace, profile: VoiceProfile) -> None:
    snippets = discover_voice_snippets(args.output_root, profile)
    if not snippets:
        console.print(
            f"No generated snippets found for {profile.selector} under "
            f"{args.output_root / 'voices' / profile.base / str(profile.version)}.",
            style="bold yellow",
        )
        return

    while True:
        table = Table(
            title=f"Snippets for {profile.selector}",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
        )
        table.add_column("#", style="bold cyan", width=4, justify="right")
        table.add_column("Created")
        table.add_column("Run name")
        table.add_column("Text", overflow="fold")
        table.add_column("Audio", style="dim")
        for index, snippet in enumerate(snippets, start=1):
            table.add_row(
                str(index),
                snippet.created_at,
                snippet.run_name or "",
                snippet.text or "",
                str(snippet.output_wav) if snippet.output_wav else "(missing)",
            )
        console.print(table)

        choice_raw = Prompt.ask("Select snippet # to play (blank to go back)", default="").strip()
        if choice_raw == "":
            return
        if not choice_raw.isdigit():
            console.print("Enter a valid snippet number.", style="bold red")
            continue

        choice = int(choice_raw)
        if not 1 <= choice <= len(snippets):
            console.print(f"Enter a number between 1 and {len(snippets)}.", style="bold red")
            continue

        play_snippet_audio(console, snippets[choice - 1], dry_run=args.dry_run)


def print_voice_actions_menu(console: Console, profile: VoiceProfile) -> None:
    table = Table(
        title=f"Voice {profile.selector}",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("#", style="bold cyan", width=4, justify="right")
    table.add_column("Action", style="white")
    table.add_row("1", "Browse snippets and play one")
    table.add_row("2", "Generate a new snippet")
    table.add_row("3", "Generate and play (no save)")
    table.add_row("4", "Choose another voice")
    table.add_row("5", "Back to main menu")
    console.print(table)


def classify_command_value(token: str, previous_flag: str | None) -> str:
    if token.startswith(("http://", "https://", "hf://")):
        return "url"
    if previous_flag in PATH_VALUE_FLAGS:
        return "path"
    if "/" in token or token.startswith(("./", "../", "~")):
        return "path"
    return "arg"


def render_command_preview(cmd: list[str]) -> Table:
    table = Table(
        show_header=True,
        header_style="bold white",
        border_style="cyan",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
    )
    table.add_column("#", justify="right", width=3, style="dim")
    table.add_column("Kind", width=8)
    table.add_column("Token", overflow="fold")

    token_styles = {
        "command": "bold green",
        "flag": "bold magenta",
        "arg": "cyan",
        "path": "bold yellow",
        "url": "bold blue",
    }
    previous_flag: str | None = None
    for index, token in enumerate(cmd):
        if index == 0:
            kind = "command"
            previous_flag = None
        elif token.startswith("-"):
            kind = "flag"
            previous_flag = token
        else:
            kind = classify_command_value(token, previous_flag)
            previous_flag = None

        style = token_styles.get(kind, "white")
        quoted_token = escape(shlex.quote(token))
        table.add_row(
            str(index),
            f"[{style}]{kind}[/{style}]",
            f"[{style}]{quoted_token}[/{style}]",
        )
    return table


def run_command_with_output_handling(
    console: Console,
    cmd: list[str],
    *,
    title: str,
    status_text: str,
    dry_run: bool,
    verbose_output: bool,
    success_message: str | None = None,
) -> bool:
    console.print(
        Panel(
            render_command_preview(cmd),
            title=title,
            border_style="cyan",
            expand=False,
        )
    )
    if dry_run:
        console.print("Dry run enabled; command not executed.", style="bold yellow")
        return True

    if verbose_output:
        try:
            result = subprocess.run(cmd, check=False)
        except FileNotFoundError as error:
            console.print(f"Command not found: {cmd[0]}", style="bold red")
            console.print(str(error), style="dim")
            return False
    else:
        try:
            with console.status(status_text):
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                )
        except FileNotFoundError as error:
            console.print(f"Command not found: {cmd[0]}", style="bold red")
            console.print(str(error), style="dim")
            return False

    if result.returncode != 0:
        combined_output = ""
        if not verbose_output:
            combined_output = ((result.stderr or "").strip() + "\n" + (result.stdout or "").strip()).strip()

            # Retry once on CPU for a known MPS runtime mismatch.
            if cmd and Path(cmd[0]).name == "pocket-tts" and "--device" in cmd:
                device_arg_index = cmd.index("--device")
                if device_arg_index + 1 < len(cmd) and cmd[device_arg_index + 1] == "mps":
                    if "Passed CPU tensor to MPS op" in combined_output:
                        fallback_cmd = cmd.copy()
                        fallback_cmd[device_arg_index + 1] = "cpu"
                        console.print(
                            "MPS runtime mismatch detected. Retrying once on CPU...",
                            style="bold yellow",
                        )
                        with console.status("Retrying on CPU..."):
                            retry_result = subprocess.run(
                                fallback_cmd,
                                check=False,
                                capture_output=True,
                                text=True,
                            )
                        if retry_result.returncode == 0:
                            if success_message:
                                console.print(
                                    f"{success_message} (CPU fallback after MPS error).",
                                    style="bold green",
                                )
                            return True
                        result = retry_result
                        combined_output = (
                            ((result.stderr or "").strip() + "\n" + (result.stdout or "").strip()).strip()
                        )

        console.print(
            f"Command failed with exit code {result.returncode}.",
            style="bold red",
        )
        if not verbose_output and combined_output:
            tail_lines = "\n".join(combined_output.splitlines()[-25:])
            console.print(
                Panel.fit(
                    tail_lines,
                    title="Command Output (last 25 lines)",
                    border_style="red",
                )
            )
        return False

    if not verbose_output and success_message:
        console.print(success_message, style="bold green")
    return True


def run_pipeline(
    console: Console,
    pipeline_script: Path,
    pipeline_args: list[str],
    dry_run: bool,
    verbose_output: bool,
) -> bool:
    effective_args = list(pipeline_args)
    if verbose_output and "--verbose-command-output" not in effective_args:
        effective_args.append("--verbose-command-output")
    cmd = [sys.executable, str(pipeline_script), *effective_args]
    return run_command_with_output_handling(
        console,
        cmd,
        title="Pipeline Command",
        status_text="Running pipeline...",
        dry_run=dry_run,
        verbose_output=verbose_output,
        success_message="Pipeline run completed.",
    )


def clone_voice_flow(
    console: Console,
    args: argparse.Namespace,
    pipeline_script: Path,
    active_device: str,
) -> None:
    console.print(Panel.fit("Clone New Voice", border_style="bright_blue"))

    source_url = Prompt.ask("Source URL (YouTube/Reddit)").strip()
    voice_base = prompt_voice_base(console)
    start = prompt_optional("Start time (optional, e.g. 21:34)")
    end = prompt_optional("End time (optional, e.g. 23:01)")
    run_name = prompt_optional("Run name (optional)")
    generate_now = Confirm.ask("Generate a sample snippet now?", default=True)

    pipeline_args = [
        "--output-root",
        str(args.output_root),
        "--source-url",
        source_url,
        "--voice",
        voice_base,
        "--device",
        active_device,
    ]
    if start:
        pipeline_args.extend(["--start", start])
    if end:
        pipeline_args.extend(["--end", end])
    if run_name:
        pipeline_args.extend(["--run-name", run_name])

    if generate_now:
        text = Prompt.ask("Text to synthesize", default=DEFAULT_SAMPLE_TEXT).strip()
        pipeline_args.extend(["--text", text])
    else:
        pipeline_args.append("--skip-generate")

    if args.dry_run:
        pipeline_args.append("--dry-run")

    run_pipeline(
        console,
        pipeline_script,
        pipeline_args,
        dry_run=args.dry_run,
        verbose_output=args.verbose_pipeline,
    )


def generate_for_profile_flow(
    console: Console,
    args: argparse.Namespace,
    pipeline_script: Path,
    profile: VoiceProfile,
    active_device: str,
) -> None:
    console.print(Panel.fit("Generate Snippet", border_style="bright_blue"))

    text = Prompt.ask("Text to synthesize").strip()
    if text == "":
        console.print("Text is required.", style="bold red")
        return

    run_name = prompt_optional("Run name (optional)")

    pipeline_args = [
        "--output-root",
        str(args.output_root),
        "--voice",
        profile.selector,
        "--text",
        text,
        "--device",
        active_device,
    ]
    if run_name:
        pipeline_args.extend(["--run-name", run_name])
    if args.dry_run:
        pipeline_args.append("--dry-run")

    run_pipeline(
        console,
        pipeline_script,
        pipeline_args,
        dry_run=args.dry_run,
        verbose_output=args.verbose_pipeline,
    )


def generate_and_play_temp_for_profile_flow(
    console: Console,
    args: argparse.Namespace,
    profile: VoiceProfile,
    active_device: str,
) -> None:
    console.print(Panel.fit("Generate + Play (No Save)", border_style="bright_blue"))
    if not profile.voice_wav.exists():
        console.print(
            f"Voice WAV is required but missing: {profile.voice_wav}",
            style="bold red",
        )
        return

    text = Prompt.ask("Text to synthesize").strip()
    if text == "":
        console.print("Text is required.", style="bold red")
        return

    if args.dry_run:
        preview_path = Path("/tmp/voice_workbench_preview.wav")
        cmd = [
            "pocket-tts",
            "generate",
            "--text",
            text,
            "--voice",
            str(profile.voice_wav),
            "--output-path",
            str(preview_path),
            "--device",
            active_device,
        ]
        success = run_command_with_output_handling(
            console,
            cmd,
            title="Temporary Generate Command",
            status_text="Generating temporary snippet...",
            dry_run=True,
            verbose_output=args.verbose_pipeline,
            success_message=None,
        )
        if success:
            console.print(
                "Dry run: this would auto-play the temporary audio and then delete it.",
                style="bold yellow",
            )
        return

    with tempfile.TemporaryDirectory(prefix="voice_workbench_") as tmp_dir:
        temp_audio_path = Path(tmp_dir) / "temp_snippet.wav"
        cmd = [
            "pocket-tts",
            "generate",
            "--text",
            text,
            "--voice",
            str(profile.voice_wav),
            "--output-path",
            str(temp_audio_path),
            "--device",
            active_device,
        ]
        success = run_command_with_output_handling(
            console,
            cmd,
            title="Temporary Generate Command",
            status_text="Generating temporary snippet...",
            dry_run=False,
            verbose_output=args.verbose_pipeline,
            success_message="Temporary snippet generated.",
        )
        if not success:
            return

        play_audio_path(console, temp_audio_path, dry_run=False)
    console.print("Temporary snippet deleted.", style="bold green")


def voice_library_flow(
    console: Console,
    args: argparse.Namespace,
    pipeline_script: Path,
    active_device: str,
) -> None:
    profile = select_voice_profile(console, args.output_root)
    if profile is None:
        return

    while True:
        print_voice_actions_menu(console, profile)
        action = IntPrompt.ask("Choose option", choices=["1", "2", "3", "4", "5"], default="1")
        if action == 1:
            browse_snippets_flow(console, args, profile)
        elif action == 2:
            generate_for_profile_flow(console, args, pipeline_script, profile, active_device)
        elif action == 3:
            generate_and_play_temp_for_profile_flow(console, args, profile, active_device)
        elif action == 4:
            profile = select_voice_profile(console, args.output_root)
            if profile is None:
                return
        else:
            return


def push_to_hugging_face_flow(console: Console, args: argparse.Namespace) -> None:
    console.print(Panel.fit("Push Voice to Hugging Face", border_style="bright_blue"))
    profile = select_voice_profile(console, args.output_root)
    if profile is None:
        return

    if not profile.voice_safetensors.exists():
        console.print(
            f"Missing required file: {profile.voice_safetensors}",
            style="bold red",
        )
        return

    repo_id = Prompt.ask("Hugging Face repo id (owner/name)").strip()
    if "/" not in repo_id:
        console.print("Repo id must look like owner/name.", style="bold red")
        return

    private_repo = Confirm.ask("Create repo as private?", default=False)
    upload_file = profile.voice_safetensors
    remote_filename = f"{profile.selector}.safetensors"
    remote_path = f"embeddings/{remote_filename}"

    table = Table(
        title="Upload Plan",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("Local file", style="white")
    table.add_column("Remote path", style="cyan")
    table.add_row(str(upload_file), remote_path)
    console.print(table)

    if not Confirm.ask("Proceed with upload?", default=True):
        console.print("Upload canceled.", style="bold yellow")
        return

    if args.dry_run:
        console.print("Dry run enabled; skipping Hugging Face upload.", style="bold yellow")
        return

    try:
        from huggingface_hub import HfApi, get_token
        from huggingface_hub.utils import (
            HfHubHTTPError,
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except ImportError as error:
        console.print(
            "This flow requires `huggingface_hub`. Run `uv sync` to install dependencies.",
            style="bold red",
        )
        console.print(str(error), style="dim")
        return

    token = get_token()
    if not token:
        console.print(
            "No Hugging Face token found. Run `uvx hf auth login` first, "
            "or paste a token below.",
            style="bold yellow",
        )
        token = Prompt.ask("HF token (blank to cancel)", default="", password=True).strip()
        if token == "":
            console.print("Upload canceled.", style="bold yellow")
            return

    api = HfApi(token=token)
    progress_bars_were_disabled = are_progress_bars_disabled()
    try:
        disable_progress_bars()
        with console.status("Creating or reusing model repo..."):
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private_repo,
                exist_ok=True,
            )
        with console.status(f"Uploading {upload_file.name}..."):
            api.upload_file(
                path_or_fileobj=str(upload_file),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )
    except HfHubHTTPError as error:
        console.print(f"Hugging Face upload failed: {error}", style="bold red")
        return
    except Exception as error:  # pragma: no cover - defensive path
        console.print(f"Unexpected upload failure: {error}", style="bold red")
        return
    finally:
        if not progress_bars_were_disabled:
            enable_progress_bars()

    hf_voice_url = f"hf://{repo_id}/{remote_path}"
    console.print(
        f"Uploaded voice embedding to https://huggingface.co/{repo_id}/blob/main/{remote_path}",
        style="bold green",
    )
    console.print(
        f"Embedding URL: {hf_voice_url}",
        style="bold cyan",
    )


def main() -> int:
    args = parse_args()
    console = Console()
    auto_device = detect_default_device()
    active_device = args.device.strip() if args.device else auto_device

    try:
        pipeline_script = resolve_pipeline_script(args.pipeline_script)
    except FileNotFoundError as error:
        console.print(str(error), style="bold red")
        return 2

    print_banner(console)
    console.print(f"Active device: [bold cyan]{active_device}[/] (auto-detected: {auto_device})")

    while True:
        print_main_menu(console, active_device=active_device, auto_device=auto_device)
        choice = IntPrompt.ask("Choose action", choices=["1", "2", "3", "4", "5"], default="1")
        if choice == 1:
            clone_voice_flow(console, args, pipeline_script, active_device)
        elif choice == 2:
            voice_library_flow(console, args, pipeline_script, active_device)
        elif choice == 3:
            push_to_hugging_face_flow(console, args)
        elif choice == 4:
            active_device = set_active_device_flow(console, active_device)
        else:
            console.print("Bye.", style="bold cyan")
            return 0


if __name__ == "__main__":
    sys.exit(main())
