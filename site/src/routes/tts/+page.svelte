<script lang="ts">
  import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
  import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
  import { AudioLinesIcon, DownloadIcon, GithubIcon } from "@lucide/svelte";
  import { onMount } from "svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { createStreamingPlayer } from "./audio";
  import { playTTS } from "./inference";
  import { fromSafetensors, type PocketTTS } from "./pocket-tts";

  // Cached large objects to download.
  let _weights: safetensors.File | null = null;
  let _model: any | null = null;
  let _tokenizer: any | null = null;

  let downloadManager: DownloadManager;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  let isDownloadingWeights = $state(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  let hasModel = $state(false);

  type LocalVoiceOption = {
    name: string;
    version: number;
    url: string;
  };

  let prompt = $state("The sun is shining, and the birds are singing.");
  let availableVoices = $state<LocalVoiceOption[]>([]);
  let selectedVoiceName = $state("");
  let selectedVoiceVersion = $state("");
  let loadingVoices = $state(false);
  let voicesError = $state<string | null>(null);
  let playing = $state(false);
  let audioBlob = $state<Blob | null>(null);

  // Advanced options
  let seed = $state<number | null>(null);
  let temperature = $state(0.7);
  let lsdDecodeSteps = $state(1);

  async function downloadClipWeights(): Promise<safetensors.File> {
    if (_weights) return _weights;
    isDownloadingWeights = true;
    try {
      const weightsUrl =
        "https://huggingface.co/ekzhang/jax-js-models/resolve/main/kyutai-pocket-tts_b6369a24-fp16.safetensors";

      const data = await downloadManager.fetch("model weights", weightsUrl);
      const result = safetensors.parse(data);
      _weights = result;
      return result;
    } catch (error) {
      alert("Error downloading weights: " + error);
      throw error;
    } finally {
      isDownloadingWeights = false;
    }
  }

  async function getModel(): Promise<PocketTTS> {
    if (_model) return _model;
    const weights = await downloadClipWeights();
    _model = fromSafetensors(weights);
    hasModel = true;
    return _model;
  }

  async function getTokenizer(): Promise<tokenizers.Unigram> {
    if (!_tokenizer)
      _tokenizer = await tokenizers.loadSentencePiece(
        "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280/tokenizer.model",
      );
    return _tokenizer;
  }

  const voiceNames = $derived(
    Array.from(new Set(availableVoices.map((voice) => voice.name))).sort(),
  );
  const selectedVoiceVersions = $derived(
    availableVoices
      .filter((voice) => voice.name === selectedVoiceName)
      .sort((a, b) => a.version - b.version),
  );

  function defaultVersionForVoice(
    voices: LocalVoiceOption[],
    voiceName: string,
  ): string {
    const versions = voices
      .filter((voice) => voice.name === voiceName)
      .map((voice) => voice.version)
      .sort((a, b) => a - b);
    if (versions.length === 0) return "";
    if (versions.includes(1)) return "1";
    return String(versions[0]);
  }

  async function loadVoices() {
    loadingVoices = true;
    voicesError = null;
    try {
      const response = await fetch("/api/voices", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Failed to load cloned voices: HTTP ${response.status}`);
      }
      const voices = (await response.json()) as LocalVoiceOption[];
      availableVoices = voices;
      const selectedStillExists = voices.some(
        (voice) =>
          voice.name === selectedVoiceName &&
          String(voice.version) === selectedVoiceVersion,
      );
      if (!selectedStillExists) {
        const fallbackName = Array.from(new Set(voices.map((voice) => voice.name))).sort()[0];
        selectedVoiceName = fallbackName ?? "";
        selectedVoiceVersion = fallbackName
          ? defaultVersionForVoice(voices, fallbackName)
          : "";
      }
    } catch (error) {
      voicesError =
        error instanceof Error ? error.message : "Failed to load cloned voices.";
      availableVoices = [];
      selectedVoiceName = "";
      selectedVoiceVersion = "";
    } finally {
      loadingVoices = false;
    }
  }

  function handleVoiceNameChange(voiceName: string) {
    selectedVoiceName = voiceName;
    selectedVoiceVersion = defaultVersionForVoice(availableVoices, voiceName);
  }

  onMount(() => {
    void loadVoices();
  });

  function prepareTextPrompt(text: string): [string, number] {
    // Ported from the Python repository.
    text = text.trim();
    if (text === "") throw new Error("Prompt cannot be empty");
    text = text.replace(/\s+/g, " ");
    const numberOfWords = text.split(" ").length;
    let framesAfterEosGuess = 3;
    if (numberOfWords <= 4) {
      framesAfterEosGuess = 5;
    }

    // Make sure it starts with an uppercase letter
    text = text.replace(/^(\p{Ll})/u, (c) => c.toLocaleUpperCase());

    // Let's make sure it ends with some kind of punctuation
    // If it ends with a letter or digit, we add a period.
    if (/[\p{L}\p{N}]$/u.test(text)) {
      text = text + ".";
    }

    // The model does not perform well when there are very few tokens, so
    // we can add empty spaces at the beginning to increase the token count.
    if (text.split(" ").length < 5) {
      text = " ".repeat(8) + text;
    }

    return [text, framesAfterEosGuess];
  }

  async function run() {
    const devices = await init();
    if (devices.includes("webgpu")) {
      defaultDevice("webgpu");
    } else {
      alert("WebGPU not supported on this device, required for inference");
      return;
    }

    const model = await getModel();
    const tokenizer = await getTokenizer();
    console.log("Model:", model);

    const [text, framesAfterEos] = prepareTextPrompt(prompt);
    const tokens = tokenizer.encode(text);
    console.log("Tokens:", tokens);

    const selectedVoiceOption = availableVoices.find(
      (voice) =>
        voice.name === selectedVoiceName &&
        String(voice.version) === selectedVoiceVersion,
    );
    if (!selectedVoiceOption) {
      throw new Error(
        "No cloned voice embedding selected. Create a voice clone first in runs/voice-clones.",
      );
    }

    const audioPrompt = safetensors.parse(
      await cachedFetch(selectedVoiceOption.url),
    ).tensors.audio_prompt;
    const voiceEmbed = np
      .array(audioPrompt.data as Float32Array<ArrayBuffer>, {
        shape: audioPrompt.shape,
        dtype: np.float32,
      })
      .slice(0)
      .astype(np.float16);

    const tokensAr = np.array(tokens, { dtype: np.uint32 });
    let embeds = model.flowLM.conditionerEmbed.ref.slice(tokensAr); // [seq_len, 1024]
    embeds = np.concatenate([voiceEmbed, embeds]);

    const player = createStreamingPlayer();
    try {
      await playTTS(player, tree.ref(model), embeds, {
        framesAfterEos,
        seed,
        temperature,
        lsdDecodeSteps,
      });
      audioBlob = player.toWav();
    } finally {
      await player.close();
    }
  }
</script>

<title>Kyutai Pocket TTS (Web)</title>

<DownloadManager bind:this={downloadManager} />

<main class="mx-4 my-8">
  <h1 class="text-2xl font-semibold mb-1">
    Kyutai Pocket TTS
    <a
      target="_blank"
      href="https://github.com/ekzhang/jax-js/tree/main/website/src/routes/tts"
    >
      <GithubIcon class="inline-block ml-2 -mt-1" />
    </a>
  </h1>
  <p class="text-lg text-gray-500">
    Text-to-speech AI voice model, running in your browser with <a
      href="https://jax-js.com/"
      class="text-primary hover:underline">jax-js</a
    >.
  </p>

  <form
    class="mt-6"
    onsubmit={async (event) => {
      event.preventDefault();
      audioBlob = null;
      playing = true;
      try {
        await run();
      } finally {
        playing = false;
      }
    }}
  >
    <textarea
      class="border-2 rounded p-2 w-full max-w-md"
      rows={6}
      placeholder="Enter your prompt here…"
      bind:value={prompt}
    ></textarea>

    <div class="flex gap-3 mt-1 h-9">
      <select
        class="border-2 rounded p-1"
        bind:value={selectedVoiceName}
        onchange={(event) =>
          handleVoiceNameChange((event.currentTarget as HTMLSelectElement).value)}
        disabled={loadingVoices || availableVoices.length === 0}
      >
        {#each voiceNames as voiceName}
          <option value={voiceName}>{voiceName}</option>
        {/each}
      </select>
      <select
        class="border-2 rounded p-1"
        bind:value={selectedVoiceVersion}
        disabled={loadingVoices || availableVoices.length === 0}
      >
        {#each selectedVoiceVersions as voice}
          <option value={String(voice.version)}>v{voice.version}</option>
        {/each}
      </select>
      <button
        class="btn"
        type="submit"
        disabled={
          playing ||
          prompt.trim() === "" ||
          loadingVoices ||
          availableVoices.length === 0
        }
      >
        {#if playing}
          <AudioLinesIcon size={20} class="animate-pulse" />
        {:else}
          Play
        {/if}
      </button>

      {#if audioBlob}
        <a
          class="btn"
          href={URL.createObjectURL(audioBlob)}
          download="tts_output.wav"
        >
          <DownloadIcon size={20} />
        </a>
      {/if}
    </div>

    {#if loadingVoices}
      <p class="mt-2 text-sm text-gray-600">Loading cloned voices...</p>
    {:else if voicesError}
      <p class="mt-2 text-sm text-red-600">{voicesError}</p>
    {:else if availableVoices.length === 0}
      <p class="mt-2 text-sm text-gray-600">
        No cloned voices found in <code>runs/voice-clones</code>.
      </p>
    {/if}

    <details class="mt-8 max-w-md">
      <summary class="cursor-pointer text-gray-600 hover:text-gray-800"
        >Advanced options</summary
      >
      <div class="mt-3 space-y-4 pl-2">
        <div>
          <label class="block text-sm text-gray-700">
            Seed
            <input
              type="number"
              class="block mt-1 border-2 rounded p-1 w-32"
              placeholder="(random)"
              bind:value={seed}
            />
          </label>
        </div>

        <div>
          <label class="block text-sm text-gray-700">
            Temperature: {temperature.toFixed(2)}
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              class="mt-1 w-full"
              bind:value={temperature}
            />
          </label>
        </div>

        <div>
          <label class="block text-sm text-gray-700">
            LSD Decode Steps: {lsdDecodeSteps}
            <input
              type="range"
              min="1"
              max="4"
              step="1"
              class="mt-1 w-full"
              bind:value={lsdDecodeSteps}
            />
          </label>
        </div>
      </div>
    </details>
  </form>
</main>

<style lang="postcss">
  @reference "$app.css";

  .btn {
    @apply flex items-center justify-center gap-2 px-3 rounded py-1 border-2 border-black;
    @apply disabled:opacity-50 disabled:cursor-wait transition-colors;
    @apply not-disabled:hover:bg-black not-disabled:hover:text-white;
  }
</style>
