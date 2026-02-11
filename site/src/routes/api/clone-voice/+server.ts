import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdtemp, readdir, rm, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import type { RequestHandler } from "./$types";

import { listLocalVoices } from "$lib/server/voices";

type CloneSourceMode = "url" | "file";
type CloneStage =
  | "starting"
  | "downloading_media"
  | "extracting_audio"
  | "cloning_voice"
  | "saving_profile"
  | "finalizing";

type CloneStreamEvent =
  | {
      type: "stage";
      stage: CloneStage;
      message: string;
    }
  | {
      type: "result";
      voiceName: string;
      version: number;
      cloneDir: string;
    }
  | {
      type: "error";
      error: string;
    };

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

function runCommandStreaming(
  command: string,
  args: string[],
  cwd: string,
  onLine: (line: string) => void,
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let stdoutBuffer = "";
    let stderrBuffer = "";

    const flushBuffer = (isStderr: boolean, force = false) => {
      const currentBuffer = isStderr ? stderrBuffer : stdoutBuffer;
      if (!force && currentBuffer === "") return;

      const parts = currentBuffer.split(/\r\n|\n|\r/);
      const remainder = force ? "" : (parts.pop() ?? "");
      if (isStderr) {
        stderrBuffer = remainder;
      } else {
        stdoutBuffer = remainder;
      }
      for (const part of parts) {
        const line = part.trim();
        if (line !== "") {
          onLine(line);
        }
      }
    };

    child.stdout.on("data", (chunk) => {
      const text = String(chunk);
      stdout += text;
      stdoutBuffer += text;
      flushBuffer(false);
    });
    child.stderr.on("data", (chunk) => {
      const text = String(chunk);
      stderr += text;
      stderrBuffer += text;
      flushBuffer(true);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      flushBuffer(false, true);
      flushBuffer(true, true);
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

function inferCloneStageFromLine(line: string): CloneStage | null {
  const lower = line.toLowerCase();
  if (
    lower.includes("downloading source audio") ||
    lower.includes("yt-dlp") ||
    lower.includes("[download]")
  ) {
    return "downloading_media";
  }
  if (lower.includes("extracting voice prompt clip") || lower.includes("ffmpeg")) {
    return "extracting_audio";
  }
  if (lower.includes("exporting voice embedding") || lower.includes("export-voice-embedding")) {
    return "cloning_voice";
  }
  if (
    lower.includes("saved voice wav profile") ||
    lower.includes("saved voice safetensors profile")
  ) {
    return "saving_profile";
  }
  if (lower.includes("all jobs completed")) {
    return "finalizing";
  }
  return null;
}

function stageMessage(stage: CloneStage): string {
  if (stage === "starting") return "Starting clone job...";
  if (stage === "downloading_media") return "Downloading media...";
  if (stage === "extracting_audio") return "Extracting audio clip...";
  if (stage === "cloning_voice") return "Cloning voice embedding...";
  if (stage === "saving_profile") return "Saving voice profile...";
  return "Finalizing clone...";
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
      "--verbose-command-output",
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

    const encoder = new TextEncoder();

    return new Response(
      new ReadableStream<Uint8Array>({
        async start(controller) {
          let streamClosed = false;
          let currentStage: CloneStage | null = null;

          const closeStream = () => {
            if (!streamClosed) {
              streamClosed = true;
              controller.close();
            }
          };

          const emit = (event: CloneStreamEvent) => {
            if (streamClosed) return;
            controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
          };

          const setStage = (stage: CloneStage) => {
            if (currentStage === stage) return;
            currentStage = stage;
            emit({
              type: "stage",
              stage,
              message: stageMessage(stage),
            });
          };

          try {
            setStage("starting");
            if (sourceMode === "url") {
              setStage("downloading_media");
            } else {
              setStage("extracting_audio");
            }

            const result = await runCommandStreaming("uv", args, repoRoot, (line) => {
              const inferredStage = inferCloneStageFromLine(line);
              if (inferredStage !== null) {
                setStage(inferredStage);
              }
            });

            if (result.code !== 0) {
              const message = (result.stderr || result.stdout || "Voice clone failed.").trim();
              emit({
                type: "error",
                error: message.split("\n").slice(-25).join("\n"),
              });
              closeStream();
              return;
            }

            if (sourceMode === "url" && !cacheDownloads) {
              await clearDownloadCacheForUrl(repoRoot, sourceUrl);
            }

            setStage("saving_profile");

            const voices = await listLocalVoices(path.join(repoRoot, "voices"));
            const createdVersions = voices
              .filter((voice) => voice.name === voiceName)
              .map((voice) => voice.version);
            const latestVersion = createdVersions.length > 0 ? Math.max(...createdVersions) : null;

            if (latestVersion === null) {
              emit({
                type: "error",
                error: "Clone completed but could not locate saved voice profile.",
              });
              closeStream();
              return;
            }

            setStage("finalizing");
            emit({
              type: "result",
              voiceName,
              version: latestVersion,
              cloneDir: `voices/${voiceName}/${latestVersion}`,
            });
          } catch (error) {
            const message = error instanceof Error ? error.message : "Unexpected server error.";
            emit({ type: "error", error: message });
          } finally {
            if (tempDir) {
              await rm(tempDir, { recursive: true, force: true }).catch(() => {
                /* ignore cleanup errors */
              });
              tempDir = null;
            }
            closeStream();
          }
        },
      }),
      {
        headers: {
          "Content-Type": "application/x-ndjson; charset=utf-8",
          "Cache-Control": "no-cache",
        },
      },
    );
  } catch (error) {
    if (tempDir) {
      await rm(tempDir, { recursive: true, force: true }).catch(() => {
        /* ignore cleanup errors */
      });
    }
    const message = error instanceof Error ? error.message : "Unexpected server error.";
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
};
