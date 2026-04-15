import { describe, expect, it } from "vitest";

import {
  calculateTotalParams,
  extractMaxContextLength,
  normalizeRepoInput,
  parseQuantizationBits,
  resolveBaseModel,
  resolveModelPayload,
  flattenTextConfig,
} from "./huggingface.js";

function createJsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return payload;
    },
  };
}

describe("huggingface helpers", () => {
  it("normalizes full Hugging Face URLs to owner/repo format", () => {
    expect(
      normalizeRepoInput("https://huggingface.co/unsloth/Qwen3.5-4B-GGUF"),
    ).toBe("unsloth/Qwen3.5-4B-GGUF");
  });

  it("prefers nested text_config fields", () => {
    expect(
      flattenTextConfig({
        num_hidden_layers: 12,
        text_config: {
          num_hidden_layers: 36,
        },
      }).num_hidden_layers,
    ).toBe(36);
  });

  it("extracts max context length from rope scaling metadata", () => {
    expect(
      extractMaxContextLength({
        rope_scaling: {
          original_max_position_embeddings: 32_768,
          factor: 8,
        },
      }),
    ).toBe(262_144);
  });

  it("parses marketed parameter counts before rough architecture estimation", () => {
    expect(
      calculateTotalParams(
        {
          model_name: "qwen/Qwen3.5-27B",
        },
        36,
        4,
        128,
        "Qwen/Qwen3.5-27B",
      ),
    ).toBe(27);
  });

  it("parses quantization bits from GGUF filenames", () => {
    expect(parseQuantizationBits("model-Q4_K_M.gguf")).toBe(4);
    expect(parseQuantizationBits("model-F16.gguf")).toBe(16);
  });

  it("follows base_model metadata recursively", async () => {
    const fetchImpl = async (url) => {
      if (url.endsWith("/api/models/unsloth/Qwen3.5-4B-GGUF")) {
        return createJsonResponse({
          cardData: {
            base_model: ["Qwen/Qwen3.5-4B-Instruct"],
          },
        });
      }

      if (url.endsWith("/api/models/Qwen/Qwen3.5-4B-Instruct")) {
        return createJsonResponse({
          cardData: {
            base_model: ["Qwen/Qwen3.5-4B"],
          },
        });
      }

      if (url.endsWith("/api/models/Qwen/Qwen3.5-4B")) {
        return createJsonResponse({
          cardData: {},
        });
      }

      throw new Error(`Unhandled URL: ${url}`);
    };

    await expect(
      resolveBaseModel("unsloth/Qwen3.5-4B-GGUF", fetchImpl),
    ).resolves.toEqual({
      baseModel: "Qwen/Qwen3.5-4B",
      ggufRepo: "unsloth/Qwen3.5-4B-GGUF",
    });
  });

  it("builds the client payload from Hugging Face responses", async () => {
    const fetchImpl = async (url: string) => {
      if (url.endsWith("/api/models/unsloth/Qwen3.5-4B-GGUF")) {
        return createJsonResponse({
          cardData: {
            base_model: ["Qwen/Qwen3.5-4B"],
          },
          siblings: [
            { rfilename: "model-Q8_0.gguf" },
            { rfilename: "model-Q4_K_M.gguf" },
          ],
        });
      }

      if (url.endsWith("/api/models/Qwen/Qwen3.5-4B")) {
        return createJsonResponse({
          cardData: {},
        });
      }

      if (url.endsWith("/Qwen/Qwen3.5-4B/resolve/main/config.json")) {
        return createJsonResponse({
          model_name: "Qwen/Qwen3.5-4B",
          text_config: {
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 128,
            max_position_embeddings: 131_072,
            vocab_size: 151_936,
          },
        });
      }

      throw new Error(`Unhandled URL: ${url}`);
    };

    const payload = await resolveModelPayload(
      "unsloth/Qwen3.5-4B-GGUF",
      fetchImpl,
    );

    expect(payload).toMatchObject({
      modelName: "Qwen/Qwen3.5-4B",
      totalParamsBillions: 4,
      numLayers: 36,
      numKvHeads: 4,
      headDim: 128,
      maxContextLength: 131_072,
      quantizations: [
        { filename: "model-Q8_0.gguf", bits: 8 },
        { filename: "model-Q4_K_M.gguf", bits: 4 },
      ],
    });
    expect(payload.contexts).toContainEqual({ label: "4K", tokens: 4_096 });
    expect(payload.contexts).toContainEqual({ label: "128K", tokens: 131_072 });
  });
});
