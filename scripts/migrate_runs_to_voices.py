#!/usr/bin/env python3
"""Migrate legacy runs/* outputs to the new voices/* + media/downloads layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate runs/voice-clones + runs/voices + runs/downloads to voices/ + media/downloads.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Default: parent of this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without moving or writing files.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def move_tree(src: Path, dst: Path, dry_run: bool) -> None:
    """Merge-move src into dst, keeping existing files in dst."""
    if not src.exists():
        return

    if not dst.exists():
        if dry_run:
            log(f"[dry-run] move {src} -> {dst}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return

    if src.is_file():
        if dry_run:
            if dst.exists():
                log(f"[dry-run] skip existing file {dst} (source: {src})")
            else:
                log(f"[dry-run] move file {src} -> {dst}")
            return
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return

    for child in src.iterdir():
        move_tree(child, dst / child.name, dry_run=dry_run)

    if dry_run:
        return
    try:
        src.rmdir()
    except OSError:
        pass


def rewrite_path_string(value: str) -> str:
    updated = value.replace("runs/downloads/", "media/downloads/")
    updated = updated.replace("/runs/downloads/", "/media/downloads/")

    idx = updated.find("runs/voice-clones/")
    if idx != -1:
        prefix = updated[:idx]
        rest = updated[idx + len("runs/voice-clones/") :]
        updated = f"{prefix}voices/{rest}"

    idx = updated.find("runs/voices/")
    if idx != -1:
        prefix = updated[:idx]
        rest = updated[idx + len("runs/voices/") :]
        parts = rest.split("/", 2)
        if len(parts) >= 3:
            voice, version, tail = parts
            updated = f"{prefix}voices/{voice}/{version}/runs/{tail}"
    return updated


def rewrite_manifest_obj(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        rewritten = rewrite_path_string(value)
        return rewritten, rewritten != value
    if isinstance(value, list):
        changed = False
        items: list[Any] = []
        for item in value:
            rewritten_item, item_changed = rewrite_manifest_obj(item)
            items.append(rewritten_item)
            changed = changed or item_changed
        return items, changed
    if isinstance(value, dict):
        changed = False
        updated: dict[str, Any] = {}
        for key, child in value.items():
            rewritten_child, child_changed = rewrite_manifest_obj(child)
            updated[key] = rewritten_child
            changed = changed or child_changed
        return updated, changed
    return value, False


def rewrite_manifests(voices_root: Path, dry_run: bool) -> None:
    manifests = sorted(voices_root.glob("*/*/**/run_manifest.json"))
    for manifest_path in manifests:
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            log(f"Skipping unreadable manifest: {manifest_path}")
            continue

        rewritten, changed = rewrite_manifest_obj(data)
        if not changed:
            continue

        if dry_run:
            log(f"[dry-run] rewrite manifest paths: {manifest_path}")
            continue

        manifest_path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
        log(f"Rewrote manifest paths: {manifest_path}")


def remove_if_empty(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir():
            remove_if_empty(child)
    try:
        path.rmdir()
    except OSError:
        pass


def migrate(repo_root: Path, dry_run: bool) -> None:
    runs_root = repo_root / "runs"
    old_clones_root = runs_root / "voice-clones"
    old_generations_root = runs_root / "voices"
    old_downloads_root = runs_root / "downloads"

    voices_root = repo_root / "voices"
    media_downloads_root = repo_root / "media" / "downloads"

    # 1) Clone profiles: runs/voice-clones/<voice>/<version> -> voices/<voice>/<version>
    if old_clones_root.exists():
        for voice_dir in sorted(p for p in old_clones_root.iterdir() if p.is_dir()):
            for version_dir in sorted(p for p in voice_dir.iterdir() if p.is_dir()):
                target = voices_root / voice_dir.name / version_dir.name
                move_tree(version_dir, target, dry_run=dry_run)

    # 2) Generation runs: runs/voices/<voice>/<version>/<run> -> voices/<voice>/<version>/runs/<run>
    if old_generations_root.exists():
        for voice_dir in sorted(p for p in old_generations_root.iterdir() if p.is_dir()):
            for version_dir in sorted(p for p in voice_dir.iterdir() if p.is_dir()):
                for run_dir in sorted(p for p in version_dir.iterdir() if p.is_dir()):
                    target = voices_root / voice_dir.name / version_dir.name / "runs" / run_dir.name
                    move_tree(run_dir, target, dry_run=dry_run)

    # 3) Download cache: runs/downloads/* -> media/downloads/*
    if old_downloads_root.exists():
        for download_entry in sorted(old_downloads_root.iterdir()):
            target = media_downloads_root / download_entry.name
            move_tree(download_entry, target, dry_run=dry_run)

    # 4) Rewrite manifest artifact paths to the new layout.
    rewrite_manifests(voices_root=voices_root, dry_run=dry_run)

    if dry_run:
        return

    # 5) Best-effort cleanup for now-empty legacy directories.
    for ds_store in (
        runs_root / ".DS_Store",
        old_clones_root / ".DS_Store",
    ):
        if ds_store.exists():
            try:
                ds_store.unlink()
            except OSError:
                pass
    remove_if_empty(old_clones_root)
    remove_if_empty(old_generations_root)
    remove_if_empty(old_downloads_root)
    remove_if_empty(runs_root)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        print(f"ERROR: repo root does not exist: {repo_root}", flush=True)
        return 2

    log(f"Repo root: {repo_root}")
    log(f"Dry run: {args.dry_run}")
    migrate(repo_root=repo_root, dry_run=args.dry_run)
    log("Migration completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
