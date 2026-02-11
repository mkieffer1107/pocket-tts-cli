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

export function resolveVoiceClonesRoot(): string {
  const override = process.env.POCKET_TTS_VOICE_CLONES_DIR;
  if (override) {
    return path.resolve(override);
  }

  const cwd = process.cwd();
  const candidates = [
    path.resolve(cwd, "runs", "voice-clones"),
    path.resolve(cwd, "..", "runs", "voice-clones"),
  ];
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
  voiceClonesRoot: string = resolveVoiceClonesRoot(),
): Promise<LocalVoice[]> {
  const voiceEntries = await readdir(voiceClonesRoot, { withFileTypes: true }).catch(
    () => [],
  );
  const voices: LocalVoice[] = [];

  for (const voiceEntry of voiceEntries) {
    if (!voiceEntry.isDirectory()) continue;
    if (!isValidVoiceName(voiceEntry.name)) continue;

    const versionsDir = path.join(voiceClonesRoot, voiceEntry.name);
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
        voiceClonesRoot,
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
