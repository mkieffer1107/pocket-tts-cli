import { existsSync } from "node:fs";
import { readdir } from "node:fs/promises";
import path from "node:path";

export type LocalVoice = {
  name: string;
  version: number;
  relativeSafetensorsPath: string;
};

const VOICE_NAME_PATTERN = /^[A-Za-z0-9_]+$/;
const VERSION_PATTERN = /^[1-9][0-9]*$/;

export function resolveVoicesRoot(): string {
  const override =
    process.env.POCKET_TTS_VOICES_DIR ?? process.env.POCKET_TTS_VOICE_CLONES_DIR;
  if (override) {
    return path.resolve(override);
  }

  const cwd = process.cwd();
  const candidates = [path.resolve(cwd, "voices"), path.resolve(cwd, "..", "voices")];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[0];
}

export function isValidVoiceName(value: string): boolean {
  return VOICE_NAME_PATTERN.test(value);
}

export function isValidVersion(value: string): boolean {
  return VERSION_PATTERN.test(value);
}

export async function listLocalVoices(
  voicesRoot: string = resolveVoicesRoot(),
): Promise<LocalVoice[]> {
  const voiceEntries = await readdir(voicesRoot, { withFileTypes: true }).catch(
    () => [],
  );
  const voices: LocalVoice[] = [];

  for (const voiceEntry of voiceEntries) {
    if (!voiceEntry.isDirectory()) continue;
    if (!isValidVoiceName(voiceEntry.name)) continue;

    const versionsDir = path.join(voicesRoot, voiceEntry.name);
    const versionEntries = await readdir(versionsDir, { withFileTypes: true }).catch(
      () => [],
    );

    const versions = versionEntries
      .filter((entry) => entry.isDirectory() && isValidVersion(entry.name))
      .map((entry) => Number(entry.name))
      .sort((a, b) => b - a);

    for (const version of versions) {
      const relativeSafetensorsPath = path.join(
        voiceEntry.name,
        String(version),
        "voice.safetensors",
      );
      const absoluteSafetensorsPath = path.join(
        voicesRoot,
        relativeSafetensorsPath,
      );
      if (!existsSync(absoluteSafetensorsPath)) continue;
      voices.push({
        name: voiceEntry.name,
        version,
        relativeSafetensorsPath,
      });
    }
  }

  return voices.sort((a, b) => {
    const nameOrder = a.name.localeCompare(b.name);
    if (nameOrder !== 0) return nameOrder;
    return b.version - a.version;
  });
}
