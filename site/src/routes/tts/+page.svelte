<script lang="ts">
  import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
  import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
  import { AudioLinesIcon, DownloadIcon, GithubIcon } from "@lucide/svelte";
  import { onDestroy, onMount } from "svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { createStreamingPlayer, type AudioPlayer } from "./audio";
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
  type WorkflowMode = "synthesis" | "clone";
  type CloneSourceMode = "url" | "file" | "record";
  type CloneVoiceResponse = {
    voiceName: string;
    version: number;
    cloneDir: string;
  };
  type CloneProgressStage =
    | "starting"
    | "downloading_media"
    | "extracting_audio"
    | "cloning_voice"
    | "saving_profile"
    | "finalizing";
  type CloneProgressEvent =
    | {
        type: "stage";
        stage: CloneProgressStage;
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
  type CloneSuccessState = {
    voiceName: string;
    version: number;
    cloneDir: string;
  };
  const CLONE_VOICE_NAME_PATTERN = /^[A-Za-z0-9_]+$/;
  const STEFAN_EXAMPLE_SOURCE_URL = "https://www.youtube.com/watch?v=UF8uR6Z6KLc";
  const STEFAN_EXAMPLE_START = "2:31";
  const STEFAN_EXAMPLE_VOICE_NAME = "stefan";

  let prompt = $state("The sun is shining, and the birds are singing.");
  let availableVoices = $state<LocalVoiceOption[]>([]);
  let selectedVoiceName = $state("");
  let selectedVoiceVersion = $state("");
  let loadingVoices = $state(false);
  let voicesError = $state<string | null>(null);
  let playing = $state(false);
  let audioBlob = $state<Blob | null>(null);
  let synthesisAbortController: AbortController | null = null;
  let activeAudioPlayer: AudioPlayer | null = null;

  // Advanced options
  let seed = $state<number | null>(null);
  let temperature = $state(0.7);
  let lsdDecodeSteps = $state(1);

  // Top-level workflow selector
  let workflowMode = $state<WorkflowMode>("synthesis");

  // Voice clone panel
  let cloneSourceMode = $state<CloneSourceMode>("url");
  let cloneSourceUrl = $state("");
  let cloneSourceFile = $state<File | null>(null);
  let cloneRecordedFile = $state<File | null>(null);
  let cloneRecordedDurationSec = $state<number | null>(null);
  let recording = $state(false);
  let cloneStart = $state("");
  let cloneEnd = $state("");
  let cloneVoiceName = $state("");
  let cloneCacheDownloads = $state(true);
  let cloning = $state(false);
  let cloneProgress = $state<string | null>(null);
  let cloneError = $state<string | null>(null);
  let cloneSuccess = $state<CloneSuccessState | null>(null);
  let activeMediaRecorder: MediaRecorder | null = null;
  let activeMediaStream: MediaStream | null = null;
  let recordingStartedAtMs = 0;

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

  function setWorkflowMode(mode: WorkflowMode) {
    workflowMode = mode;
  }

  function cleanupRecordingResources() {
    if (activeMediaStream !== null) {
      for (const track of activeMediaStream.getTracks()) {
        track.stop();
      }
    }
    activeMediaStream = null;
    activeMediaRecorder = null;
  }

  function guessRecordedExtension(mimeType: string): string {
    if (mimeType.includes("ogg")) return "ogg";
    if (mimeType.includes("wav")) return "wav";
    if (mimeType.includes("mp4") || mimeType.includes("m4a")) return "m4a";
    return "webm";
  }

  async function startRecording() {
    cloneError = null;
    cloneSuccess = null;

    if (typeof window === "undefined") return;
    if (recording) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      cloneError = "Microphone recording is not available in this browser.";
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      cloneError = "MediaRecorder is not available in this browser.";
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
      ];
      const mimeType =
        mimeCandidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? "";
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      activeMediaStream = stream;
      activeMediaRecorder = recorder;
      recordingStartedAtMs = Date.now();
      recording = true;
      cloneRecordedFile = null;
      cloneRecordedDurationSec = null;

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      recorder.onerror = () => {
        recording = false;
        cloneError = "Microphone recording failed.";
        cleanupRecordingResources();
      };
      recorder.onstop = () => {
        const durationSec = (Date.now() - recordingStartedAtMs) / 1000;
        recording = false;

        const finalMimeType = recorder.mimeType || mimeType || "audio/webm";
        const recordedBlob = new Blob(chunks, { type: finalMimeType });
        if (recordedBlob.size === 0) {
          cloneRecordedFile = null;
          cloneRecordedDurationSec = null;
          cloneError = "Recorded audio was empty. Please try again.";
          cleanupRecordingResources();
          return;
        }

        const extension = guessRecordedExtension(finalMimeType);
        cloneRecordedFile = new File([recordedBlob], `microphone-recording.${extension}`, {
          type: finalMimeType,
        });
        cloneRecordedDurationSec = Number(durationSec.toFixed(1));
        cleanupRecordingResources();
      };

      recorder.start(250);
    } catch (error) {
      recording = false;
      cloneRecordedFile = null;
      cloneRecordedDurationSec = null;
      cloneError =
        error instanceof Error
          ? `Microphone access failed: ${error.message}`
          : "Microphone access failed.";
      cleanupRecordingResources();
    }
  }

  function stopRecording() {
    if (!recording || activeMediaRecorder === null) return;
    if (activeMediaRecorder.state === "inactive") return;
    try {
      activeMediaRecorder.stop();
    } catch {
      recording = false;
      cleanupRecordingResources();
    }
  }

  function setCloneSourceMode(mode: CloneSourceMode) {
    if (cloneSourceMode === "record" && mode !== "record" && recording) {
      stopRecording();
    }
    cloneSourceMode = mode;
    cloneProgress = null;
    cloneError = null;
    cloneSuccess = null;
  }

  function openStefanClonePreset() {
    if (recording) {
      stopRecording();
    }
    workflowMode = "clone";
    cloneSourceMode = "url";
    cloneSourceUrl = STEFAN_EXAMPLE_SOURCE_URL;
    cloneSourceFile = null;
    cloneRecordedFile = null;
    cloneRecordedDurationSec = null;
    cloneStart = STEFAN_EXAMPLE_START;
    cloneEnd = "";
    cloneVoiceName = STEFAN_EXAMPLE_VOICE_NAME;
    cloneCacheDownloads = false;
    cloneProgress = null;
    cloneError = null;
    cloneSuccess = null;
  }

  function tryClonedVoiceNow(voiceName: string, version: number) {
    selectedVoiceName = voiceName;
    selectedVoiceVersion = String(version);
    workflowMode = "synthesis";
  }

  function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === "AbortError";
  }

  async function stopSynthesis() {
    synthesisAbortController?.abort();
    if (activeAudioPlayer !== null) {
      try {
        await activeAudioPlayer.stop();
      } catch {
        // Best-effort stop; run cleanup handles remaining state.
      }
    }
  }

  function handleCloneFileChange(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    cloneSourceFile = input.files?.[0] ?? null;
  }

  async function readCloneProgressStream(response: Response): Promise<CloneVoiceResponse> {
    if (!response.body) {
      throw new Error("Clone response did not include a progress stream.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result: CloneVoiceResponse | null = null;

    const handleLine = (line: string) => {
      let event: CloneProgressEvent;
      try {
        event = JSON.parse(line) as CloneProgressEvent;
      } catch {
        return;
      }

      if (event.type === "stage") {
        cloneProgress = event.message;
        return;
      }
      if (event.type === "error") {
        throw new Error(event.error || "Voice cloning failed.");
      }
      result = {
        voiceName: event.voiceName,
        version: event.version,
        cloneDir: event.cloneDir,
      };
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newlineIndex = buffer.indexOf("\n");
      while (newlineIndex >= 0) {
        const rawLine = buffer.slice(0, newlineIndex).trim();
        buffer = buffer.slice(newlineIndex + 1);
        if (rawLine !== "") {
          handleLine(rawLine);
        }
        newlineIndex = buffer.indexOf("\n");
      }
    }

    const trailing = (buffer + decoder.decode()).trim();
    if (trailing !== "") {
      handleLine(trailing);
    }

    if (result === null) {
      throw new Error("Voice cloning finished without a result.");
    }
    return result;
  }

  async function submitCloneVoice() {
    cloneProgress = null;
    cloneError = null;
    cloneSuccess = null;

    const voiceName = cloneVoiceName.trim();
    if (!CLONE_VOICE_NAME_PATTERN.test(voiceName) || voiceName.includes("-")) {
      cloneError =
        "Voice name must contain only letters, numbers, and underscores (no hyphens).";
      return;
    }
    if (cloneSourceMode === "url" && cloneSourceUrl.trim() === "") {
      cloneError = "Source URL is required.";
      return;
    }
    if (cloneSourceMode === "file" && cloneSourceFile === null) {
      cloneError = "Local source file is required.";
      return;
    }
    if (cloneSourceMode === "record" && cloneRecordedFile === null) {
      cloneError = "Record a microphone clip first.";
      return;
    }
    if (cloneSourceMode === "record" && recording) {
      cloneError = "Stop recording before cloning.";
      return;
    }

    cloning = true;
    try {
      cloneProgress = "Starting clone job...";
      const formData = new FormData();
      formData.set("sourceMode", cloneSourceMode === "url" ? "url" : "file");
      formData.set("voiceName", voiceName);
      if (cloneSourceMode === "url") {
        formData.set("sourceUrl", cloneSourceUrl.trim());
      } else if (cloneSourceMode === "file" && cloneSourceFile) {
        formData.set("sourceFile", cloneSourceFile);
      } else if (cloneSourceMode === "record" && cloneRecordedFile) {
        formData.set("sourceFile", cloneRecordedFile);
      }
      if (cloneSourceMode !== "record" && cloneStart.trim() !== "") {
        formData.set("start", cloneStart.trim());
      }
      if (cloneSourceMode !== "record" && cloneEnd.trim() !== "") {
        formData.set("end", cloneEnd.trim());
      }
      formData.set("cacheDownloads", String(cloneCacheDownloads));

      const response = await fetch("/api/clone-voice", {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type") ?? "";
      if (!response.ok) {
        const responseText = await response.text();
        let payload: Record<string, unknown> = {};
        if (responseText.trim() !== "") {
          try {
            payload = JSON.parse(responseText) as Record<string, unknown>;
          } catch {
            payload = { error: responseText };
          }
        }
        const message =
          typeof payload.error === "string"
            ? payload.error
            : `Voice cloning failed (HTTP ${response.status}).`;
        throw new Error(message);
      }

      let result: CloneVoiceResponse;
      if (contentType.includes("application/x-ndjson")) {
        result = await readCloneProgressStream(response);
      } else {
        const responseText = await response.text();
        if (responseText.trim() === "") {
          throw new Error("Voice cloning failed: empty response from server.");
        }
        const payload = JSON.parse(responseText) as CloneVoiceResponse;
        result = payload;
      }

      cloneSuccess = {
        voiceName: result.voiceName,
        version: result.version,
        cloneDir: result.cloneDir,
      };
      await loadVoices();
      selectedVoiceName = result.voiceName;
      selectedVoiceVersion = String(result.version);
      cloneSourceUrl = "";
      cloneSourceFile = null;
      cloneRecordedFile = null;
      cloneRecordedDurationSec = null;
      cloneStart = "";
      cloneEnd = "";
      cloneVoiceName = "";
      cloneProgress = null;
    } catch (error) {
      cloneError = error instanceof Error ? error.message : "Voice cloning failed.";
      cloneProgress = null;
    } finally {
      cloning = false;
    }
  }

  onMount(() => {
    void loadVoices();
  });

  onDestroy(() => {
    if (activeMediaRecorder !== null && activeMediaRecorder.state !== "inactive") {
      try {
        activeMediaRecorder.stop();
      } catch {
        // Ignore teardown failures.
      }
    }
    cleanupRecordingResources();
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

  async function run(signal: AbortSignal) {
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
        "No cloned voice embedding selected. Create a voice clone first in voices/<name>/<version>.",
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
    activeAudioPlayer = player;
    try {
      await playTTS(player, tree.ref(model), embeds, {
        framesAfterEos,
        seed,
        temperature,
        lsdDecodeSteps,
        signal,
      });
      if (!signal.aborted) {
        audioBlob = player.toWav();
      }
    } finally {
      try {
        if (signal.aborted) {
          await player.stop();
        } else {
          await player.close();
        }
      } finally {
        if (activeAudioPlayer === player) {
          activeAudioPlayer = null;
        }
      }
    }
  }
</script>

<title>Kyutai Pocket TTS (Web)</title>

<DownloadManager bind:this={downloadManager} />

<main class="mx-4 py-8 min-h-screen flex flex-col items-center justify-center">
  <h1 class="text-2xl font-semibold mb-1 text-center">
    Kyutai Pocket TTS
    <a
      target="_blank"
      href="https://github.com/ekzhang/jax-js/tree/main/website/src/routes/tts"
    >
      <GithubIcon class="inline-block ml-2 -mt-1" />
    </a>
  </h1>
  <p class="text-lg text-gray-500 text-center max-w-2xl">
    Text-to-speech AI voice model, running in your browser with <a
      href="https://jax-js.com/"
      class="text-primary hover:underline">jax-js</a
    >.
  </p>

  <div class="mt-6 w-full max-w-2xl flex flex-col items-center">
    <div class="inline-flex w-full max-w-md border-2 border-black rounded-md overflow-hidden">
      <button
        type="button"
        class={`mode-btn ${workflowMode === "synthesis" ? "mode-btn-active" : ""}`}
        onclick={() => setWorkflowMode("synthesis")}
      >
        Voice Synthesis
      </button>
      <button
        type="button"
        class={`mode-btn ${workflowMode === "clone" ? "mode-btn-active" : ""}`}
        onclick={() => setWorkflowMode("clone")}
      >
        Voice Cloning
      </button>
    </div>

    <div class="mt-4 w-full max-w-md workflow-shell">
      {#if workflowMode === "synthesis"}
      <form
        id="voice-synthesis"
        class="w-full"
        onsubmit={async (event) => {
          event.preventDefault();
          if (playing) return;
          audioBlob = null;
          playing = true;
          const abortController = new AbortController();
          synthesisAbortController = abortController;
          try {
            await run(abortController.signal);
          } catch (error) {
            if (!isAbortError(error)) {
              console.error(error);
              alert(
                error instanceof Error
                  ? `TTS generation failed: ${error.message}`
                  : "TTS generation failed.",
              );
            }
          } finally {
            if (synthesisAbortController === abortController) {
              synthesisAbortController = null;
            }
            playing = false;
          }
        }}
      >
        <textarea
          class="border-2 rounded p-2 w-full"
          rows={6}
          placeholder="Enter your prompt here…"
          bind:value={prompt}
        ></textarea>

        <div class="flex gap-3 mt-1 h-9 flex-wrap justify-center">
          {#if availableVoices.length > 0}
            <select
              class="border-2 rounded p-1"
              bind:value={selectedVoiceName}
              onchange={(event) =>
                handleVoiceNameChange((event.currentTarget as HTMLSelectElement).value)}
              disabled={loadingVoices}
            >
              {#each voiceNames as voiceName}
                <option value={voiceName}>{voiceName}</option>
              {/each}
            </select>
            <select
              class="border-2 rounded p-1"
              bind:value={selectedVoiceVersion}
              disabled={loadingVoices}
            >
              {#each selectedVoiceVersions as voice}
                <option value={String(voice.version)}>v{voice.version}</option>
              {/each}
            </select>
          {/if}
          {#if playing}
            <button class="btn" type="button" onclick={stopSynthesis}>
              Stop
            </button>
          {:else}
            <button
              class="btn"
              type="submit"
              disabled={
                prompt.trim() === "" ||
                loadingVoices ||
                availableVoices.length === 0
              }
            >
              <AudioLinesIcon size={20} />
              Play
            </button>
          {/if}

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
          <p class="mt-2 text-sm text-gray-600 text-center">Loading cloned voices...</p>
        {:else if voicesError}
          <p class="mt-2 text-sm text-red-600 text-center">{voicesError}</p>
        {:else if availableVoices.length === 0}
          <p class="mt-2 text-sm text-gray-600 text-center">
            No cloned voices found.
            <a
              href="#voice-cloning"
              class="font-semibold underline underline-offset-2 hover:no-underline"
              onclick={openStefanClonePreset}
            >
              Set up the Stefan example now
            </a>
          </p>
        {/if}

        <details class="mt-8 w-full">
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
      {:else}
        <section id="voice-cloning" class="border-2 border-black rounded-lg p-4 bg-white/80 w-full">
        <h2 class="text-lg font-semibold text-center">Clone Voice</h2>
        <p class="text-sm text-gray-600 mt-1 text-center">
          Add new voice embeddings to <code>voices/&lt;name&gt;/&lt;version&gt;</code>.
        </p>

        <form
          class="mt-4 space-y-3"
          onsubmit={async (event) => {
            event.preventDefault();
            await submitCloneVoice();
          }}
        >
          <div class="inline-flex w-full border-2 border-black rounded-md overflow-hidden">
            <button
              type="button"
              class={`mode-btn ${cloneSourceMode === "url" ? "mode-btn-active" : ""}`}
              onclick={() => setCloneSourceMode("url")}
              disabled={cloning || recording}
            >
              Video URL
            </button>
            <button
              type="button"
              class={`mode-btn ${cloneSourceMode === "file" ? "mode-btn-active" : ""}`}
              onclick={() => setCloneSourceMode("file")}
              disabled={cloning || recording}
            >
              Local file
            </button>
            <button
              type="button"
              class={`mode-btn ${cloneSourceMode === "record" ? "mode-btn-active" : ""}`}
              onclick={() => setCloneSourceMode("record")}
              disabled={cloning || recording}
            >
              Record mic
            </button>
          </div>

          {#if cloneSourceMode === "url"}
            <label class="block text-sm text-gray-700">
              Source URL
              <input
                type="url"
                class="block mt-1 w-full border-2 rounded p-2"
                placeholder="https://www.youtube.com/watch?v=..."
                bind:value={cloneSourceUrl}
                disabled={cloning}
              />
            </label>
          {:else if cloneSourceMode === "file"}
            <label class="block text-sm text-gray-700">
              Local media file
              <input
                type="file"
                class="block mt-1 w-full border-2 rounded p-2 text-sm"
                accept=".wav,.mp3,.m4a,.mp4,.webm,.opus,.aac,.flac"
                onchange={handleCloneFileChange}
                disabled={cloning}
              />
            </label>
          {:else}
            <div class="space-y-2">
              <p class="text-sm text-gray-700">
                Record a clip from your microphone and use it as the clone source.
              </p>
              <div class="flex gap-2 flex-wrap">
                {#if recording}
                  <button
                    type="button"
                    class="btn"
                    onclick={stopRecording}
                    disabled={cloning}
                  >
                    Stop recording
                  </button>
                  <span class="text-sm text-red-700 self-center">Recording...</span>
                {:else}
                  <button
                    type="button"
                    class="btn"
                    onclick={startRecording}
                    disabled={cloning}
                  >
                    Start recording
                  </button>
                {/if}
              </div>
              {#if cloneRecordedFile}
                <p class="text-sm text-gray-700">
                  Recorded clip ready ({cloneRecordedDurationSec ?? 0}s):
                  <code>{cloneRecordedFile.name}</code>
                </p>
              {:else}
                <p class="text-sm text-gray-600">No microphone clip recorded yet.</p>
              {/if}
            </div>
          {/if}

          {#if cloneSourceMode !== "record"}
            <div class="grid grid-cols-2 gap-2">
              <label class="block text-sm text-gray-700">
                Start (optional)
                <input
                  type="text"
                  class="block mt-1 w-full border-2 rounded p-2"
                  placeholder="2:31"
                  bind:value={cloneStart}
                  disabled={cloning}
                />
              </label>
              <label class="block text-sm text-gray-700">
                End (optional)
                <input
                  type="text"
                  class="block mt-1 w-full border-2 rounded p-2"
                  placeholder="2:45"
                  bind:value={cloneEnd}
                  disabled={cloning}
                />
              </label>
            </div>
          {/if}

          <label class="block text-sm text-gray-700">
            Voice name
            <input
              type="text"
              class="block mt-1 w-full border-2 rounded p-2"
              placeholder="stefan"
              bind:value={cloneVoiceName}
              disabled={cloning}
            />
          </label>

          <label class="flex items-start gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              class="mt-1 accent-black"
              bind:checked={cloneCacheDownloads}
              disabled={cloning || cloneSourceMode !== "url"}
            />
            <span>Cache source downloads in <code>media/downloads</code></span>
          </label>

          <button class="btn w-full justify-center" type="submit" disabled={cloning || recording}>
            {cloning ? "Cloning..." : "Clone voice"}
          </button>

          {#if cloning && cloneProgress}
            <p class="text-sm text-gray-700">{cloneProgress}</p>
          {/if}
        </form>

        {#if cloneError}
          <p class="mt-3 text-sm text-red-600 whitespace-pre-wrap">{cloneError}</p>
        {/if}
        {#if cloneSuccess}
          <p class="mt-3 text-sm text-green-700">
            Saved voice <span class="font-semibold">{cloneSuccess.voiceName}</span> (v{cloneSuccess.version}).
            <a
              href="#voice-synthesis"
              class="font-semibold underline underline-offset-2 hover:no-underline"
              onclick={() => tryClonedVoiceNow(cloneSuccess.voiceName, cloneSuccess.version)}
            >
              Try it now
            </a>
          </p>
        {/if}
        </section>
      {/if}
    </div>
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";

  .btn {
    @apply flex items-center justify-center gap-2 px-3 rounded py-1 border-2 border-black;
    @apply disabled:opacity-50 disabled:cursor-wait transition-colors;
    @apply not-disabled:hover:bg-black not-disabled:hover:text-white;
  }

  .mode-btn {
    @apply flex-1 px-3 py-1 text-sm transition-colors bg-white text-black;
    @apply disabled:opacity-50 disabled:cursor-not-allowed;
    @apply not-disabled:hover:bg-black not-disabled:hover:text-white;
  }

  .mode-btn-active {
    @apply bg-black text-white;
  }

  .workflow-shell {
    min-height: clamp(32rem, 62vh, 38rem);
  }
</style>
