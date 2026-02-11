import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";
import lucidePreprocess from "vite-plugin-lucide-preprocess";

export default defineConfig({
  plugins: [
    sveltekit(),
    tailwindcss(),
    lucidePreprocess(),
    process.env.BASIC_SSL ? basicSsl() : null,
  ],
  optimizeDeps: {
    exclude: ["@rollup/browser", "onnxruntime-web"],
  },
  build: {
    chunkSizeWarningLimit: 4000,
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
