import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdtemp, readdir, rm, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import type { RequestHandler } from "./$types";

import { listLocalVoices } from "$lib/server/voices";

type CloneSourceMode = "url" | "file";

const VOICE_NAME_PATTERN = /^[A-Za-z0-9_]+$/;

function resolveRepoRoot(): string {
  const cwd = process.cwd();
  const candidates = [cwd, path.resolve(cwd, "..")];
  for (const candidate of candidates) {
    const pipelinePath = path.join(candidate, "src", "pocket_tts_youtube_pipeline.py");
    if (existsSync(pipelinePath)) {
      return candidate;
    }
  }
  throw new Error("Could not resolve repo root from current working directory.");
}

function runCommand(
  command: string,
  args: string[],
  cwd: string,
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

function makeCacheStem(sourceUrl: string): string {
  const digest = createHash("sha256")
    .update(sourceUrl.trim(), "utf8")
    .digest("hex")
    .slice(0, 20);
  return `source_${digest}`;
}

async function clearDownloadCacheForUrl(repoRoot: string, sourceUrl: string): Promise<void> {
  const downloadsDir = path.join(repoRoot, "media", "downloads");
  const cacheStem = makeCacheStem(sourceUrl);
  const legacyStem = cacheStem.replace(/^source_/, "youtube_");
  const entries = await readdir(downloadsDir).catch(() => []);
  const targets = entries.filter(
    (name) =>
      name.startsWith(`${cacheStem}.`) ||
      name.startsWith(`${legacyStem}.`) ||
      name === cacheStem ||
      name === legacyStem,
  );
  await Promise.all(
    targets.map((name) =>
      unlink(path.join(downloadsDir, name)).catch(() => {
        /* ignore cleanup errors */
      }),
    ),
  );
}

export const POST: RequestHandler = async ({ request }) => {
  let tempDir: string | null = null;

  try {
    const formData = await request.formData();
    const sourceMode = (formData.get("sourceMode")?.toString() ?? "url") as CloneSourceMode;
    const sourceUrl = formData.get("sourceUrl")?.toString().trim() ?? "";
    const voiceNameRaw = formData.get("voiceName")?.toString().trim() ?? "";
    const start = formData.get("start")?.toString().trim() ?? "";
    const end = formData.get("end")?.toString().trim() ?? "";
    const cacheDownloads = formData.get("cacheDownloads")?.toString() !== "false";

    if (!VOICE_NAME_PATTERN.test(voiceNameRaw) || voiceNameRaw.includes("-")) {
      return new Response(
        JSON.stringify({
          error:
            "Invalid voice name. Use only letters, numbers, and underscores (no hyphens).",
        }),
        { status: 400, headers: { "Content-Type": "application/json" } },
      );
    }
    const voiceName = voiceNameRaw.toLowerCase();

    const repoRoot = resolveRepoRoot();
    const outputRoot = path.join(repoRoot, "voices");

    const args = [
      "run",
      path.join("src", "pocket_tts_youtube_pipeline.py"),
      "--output-root",
      outputRoot,
      "--voice",
      voiceName,
      "--skip-generate",
    ];
    if (start) args.push("--start", start);
    if (end) args.push("--end", end);

    if (sourceMode === "url") {
      if (sourceUrl === "") {
        return new Response(JSON.stringify({ error: "Source URL is required." }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
      args.push("--source-url", sourceUrl);
    } else if (sourceMode === "file") {
      const sourceFile = formData.get("sourceFile");
      if (!(sourceFile instanceof File) || sourceFile.size === 0) {
        return new Response(JSON.stringify({ error: "Local source file is required." }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        });
      }
      tempDir = await mkdtemp(path.join(os.tmpdir(), "pocket-tts-site-"));
      const fileName = sourceFile.name.replace(/[^A-Za-z0-9._-]+/g, "_") || "upload.bin";
      const tempSourcePath = path.join(tempDir, fileName);
      const fileBytes = new Uint8Array(await sourceFile.arrayBuffer());
      await writeFile(tempSourcePath, fileBytes);
      args.push("--source-file", tempSourcePath);
    } else {
      return new Response(JSON.stringify({ error: "Invalid source mode." }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const result = await runCommand("uv", args, repoRoot);
    if (result.code !== 0) {
      const message = (result.stderr || result.stdout || "Voice clone failed.").trim();
      return new Response(
        JSON.stringify({
          error: message.split("\n").slice(-25).join("\n"),
        }),
        { status: 500, headers: { "Content-Type": "application/json" } },
      );
    }

    if (sourceMode === "url" && !cacheDownloads) {
      await clearDownloadCacheForUrl(repoRoot, sourceUrl);
    }

    const voices = await listLocalVoices(path.join(repoRoot, "voices"));
    const createdVersions = voices
      .filter((voice) => voice.name === voiceName)
      .map((voice) => voice.version);
    const latestVersion = createdVersions.length > 0 ? Math.max(...createdVersions) : null;
    if (latestVersion === null) {
      return new Response(
        JSON.stringify({
          error: "Clone completed but could not locate saved voice profile.",
        }),
        { status: 500, headers: { "Content-Type": "application/json" } },
      );
    }

    return new Response(
      JSON.stringify({
        voiceName,
        version: latestVersion,
        cloneDir: `voices/${voiceName}/${latestVersion}`,
      }),
      { headers: { "Content-Type": "application/json" } },
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected server error.";
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  } finally {
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true }).catch(() => {
        /* ignore cleanup errors */
      });
    }
  }
};
