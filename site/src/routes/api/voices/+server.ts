import type { RequestHandler } from "./$types";

import {
  listLocalVoices,
  resolveVoicesRoot,
} from "$lib/server/voices";

export const GET: RequestHandler = async () => {
  const voicesRoot = resolveVoicesRoot();
  const voices = await listLocalVoices(voicesRoot);

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
