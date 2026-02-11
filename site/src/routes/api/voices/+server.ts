import type { RequestHandler } from "./$types";

import {
  listLocalVoices,
  resolveVoiceClonesRoot,
} from "$lib/server/voice-clones";

export const GET: RequestHandler = async () => {
  const voiceClonesRoot = resolveVoiceClonesRoot();
  const voices = await listLocalVoices(voiceClonesRoot);

  const payload = voices.map((voice) => ({
    name: voice.name,
    version: voice.version,
    url: `/api/voices/${encodeURIComponent(voice.name)}/${voice.version}`,
  }));

  return new Response(JSON.stringify(payload), {
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
  });
};
