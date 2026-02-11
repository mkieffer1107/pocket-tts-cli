import { readFile } from "node:fs/promises";
import path from "node:path";

import type { RequestHandler } from "./$types";

import {
  isValidVersion,
  isValidVoiceName,
  resolveVoiceClonesRoot,
} from "$lib/server/voice-clones";

export const GET: RequestHandler = async ({ params }) => {
  const { voice, version } = params;
  if (!isValidVoiceName(voice) || !isValidVersion(version)) {
    return new Response("Not found", { status: 404 });
  }

  const voiceClonesRoot = resolveVoiceClonesRoot();
  const safetensorsPath = path.join(
    voiceClonesRoot,
    voice,
    version,
    "voice.safetensors",
  );

  let data: Uint8Array<ArrayBuffer>;
  try {
    data = await readFile(safetensorsPath);
  } catch {
    return new Response("Not found", { status: 404 });
  }

  return new Response(data, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Cache-Control": "no-store",
    },
  });
};
