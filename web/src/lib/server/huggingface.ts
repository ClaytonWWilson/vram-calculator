import { getAvailableContexts, sortQuantizations } from "$lib/calculator.js";
import type { ModelInfo } from "../../types/huggingface";
import { cachedFetch } from "./cachedFetch";

const HF_ORIGIN = "https://huggingface.co";
const MAX_BASE_MODEL_DEPTH = 10;

export function normalizeRepoInput(input: string) {
  const value = input.trim().replace(/\/+$/, "");

  if (!value) {
    throw new Error("A Hugging Face GGUF repository is required. Format: owner/repo-name.");
  }

  if (value.includes("://")) {
    let url;

    try {
      url = new URL(value);
    } catch {
      throw new Error(`Invalid URL format: "${input}". Please provide a valid Hugging Face repo URL.`);
    }

    if (!url.hostname.endsWith("huggingface.co")) {
      throw new Error(
        `Only Hugging Face model URLs are supported. Got hostname: ${url.hostname}.`,
      );
    }

    const parts = url.pathname.split("/").filter(Boolean);

    if (parts.length < 2) {
      throw new Error(
        `Invalid repository path in URL. Expected owner/repo format, got: ${url.pathname}`,
      );
    }

    return `${parts[0]}/${parts[1]}`;
  }

  const parts = value.split("/").filter(Boolean);

  if (parts.length !== 2) {
    throw new Error(
      `Invalid repository format: "${input}". Expected owner/repo-name format (e.g., meta-llama/Llama-3-GGUF).`,
    );
  }

  return `${parts[0]}/${parts[1]}`;
}

async function requestJson(path: string, fetchImpl: typeof fetch) {
  const response = await fetchImpl(`${HF_ORIGIN}${path}`, {
    headers: {
      accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${url}: HTTP ${response.status}. Check that the repository exists and is accessible.`,
    );
  }

  }

  return response.json();
}

async function requestConfig(repo: string, fetchImpl: typeof fetch) {
  const response = await fetchImpl(
    `${HF_ORIGIN}/${repo}/resolve/main/config.json`,
    {
      headers: {
        accept: "application/json",
      },
    },
  );

  if (!response.ok) {
    throw new Error(
      `Failed to fetch config.json for ${repo}: HTTP ${response.status}. The repository may not exist or config.json may be missing.`,
    );
  }
  }

  return response.json();
}

export function inferOwnerFromModel(modelName: string) {
  const value = modelName.toLowerCase();

  if (value.includes("llama")) {
    return "meta-llama";
  }

  if (value.includes("qwen")) {
    return "Qwen";
  }

  if (value.includes("mistral")) {
    return "mistralai";
  }

  if (value.includes("gemma")) {
    return "google";
  }

  if (value.includes("phi")) {
    return "microsoft";
  }

  if (value.includes("olmo")) {
    return "allenai";
  }

  return "meta-llama";
}

export function inferBaseModel(repo: string) {
  const [owner, name] = normalizeRepoInput(repo).split("/");
  const baseRepo = name.replace(/-GGUF$/i, "");
  const inferredOwner = new Set([
    "unsloth",
    "bartowski",
    "thebloke",
    "maziyarpanahi",
  ]).has(owner.toLowerCase())
    ? inferOwnerFromModel(name)
    : owner;

  return `${inferredOwner}/${baseRepo}`;
}

function getBaseModels(modelInfo) {
  const cardData = modelInfo.cardData ?? modelInfo.card_data;
  const value = cardData?.base_model ?? cardData?.baseModel ?? [];

  if (Array.isArray(value)) {
    return value.filter((item) => typeof item === "string" && item);
  }

  if (typeof value === "string" && value) {
    return [value];
  }

  return [];
}

export async function resolveBaseModel(
  ggufRepo: string,
  fetchImpl: typeof fetch,
) {
  let currentRepo = normalizeRepoInput(ggufRepo);
  let resolvedBaseModel: string | null = null;
  const visited = new Set();

  for (let depth = 0; depth < MAX_BASE_MODEL_DEPTH; depth += 1) {
    if (visited.has(currentRepo)) {
      break;
    }

    visited.add(currentRepo);

    let modelInfo;

    try {
      modelInfo = await requestJson(`/api/models/${currentRepo}`, fetchImpl);
    } catch {
      break;
    }

    const baseModels = getBaseModels(modelInfo);

    if (!baseModels.length) {
      break;
    }

    resolvedBaseModel = baseModels[0];
    currentRepo = resolvedBaseModel;
  }

  return {
    baseModel: resolvedBaseModel ?? inferBaseModel(ggufRepo),
    ggufRepo: normalizeRepoInput(ggufRepo),
  };
}

export function flattenTextConfig(configData: any) {
  // Return empty object if configData is null/undefined or not an object (Issue #15)
  if (!configData || typeof configData !== "object") {
    return {};
  }

  // Text config data is inside a text_config key
  if (configData.text_config && typeof configData.text_config === "object") {
    return {
      ...configData,
      ...configData.text_config,
    };
  }

  // Text config is at the top level.
  return configData;
}

function getPositiveInt(value: any): number | null {
  return Number.isInteger(value) && value > 0 ? value : null;
}

export function extractMaxContextLength(configData: any) {
  const candidateKeys = [
    "max_position_embeddings",
    "max_sequence_length",
    "model_max_length",
    "max_seq_len",
    "max_context_length",
    "seq_len",
    "n_positions",
  ];

  const candidates = candidateKeys
    .map((key) => getPositiveInt(configData[key]))
    .filter(Boolean);

  const ropeScaling = configData.rope_scaling;

  if (ropeScaling && typeof ropeScaling === "object") {
    const originalMax = getPositiveInt(
      ropeScaling.original_max_position_embeddings,
    );

    if (originalMax) {
      candidates.push(originalMax);

      if (typeof ropeScaling.factor === "number" && ropeScaling.factor > 0) {
        candidates.push(Math.ceil(originalMax * ropeScaling.factor));
      }
    }
  }

  return Math.max(...candidates, 4_096);
}

export function extractParamCountFromIdentifiers(...identifiers) {
  const matches = [];

  for (const identifier of identifiers) {
    if (typeof identifier !== "string") {
      continue;
    }

    for (const match of identifier.matchAll(
      /(?<!\d)(\d+(?:\.\d+)?)B(?![A-Za-z])/gi,
    )) {
      matches.push(Number(match[1]));
    }
  }

  return matches.length ? Math.max(...matches) : null;
}

export function calculateTotalParams(
  configData,
  numLayers,
  numHeads,
  headDim,
  ggufRepoName, // Renamed from 'baseModel' for clarity - this is the GGUF repo name (Issue #14)
) {
  if (
    typeof configData.num_parameters === "number" &&
    configData.num_parameters > 0
  ) {
    return configData.num_parameters / 1e9;
  }

  if (
    typeof configData.parameter_count === "number" &&
    configData.parameter_count > 0
  ) {
    return configData.parameter_count / 1e9;
  }

  const inferred = extractParamCountFromIdentifiers(
    configData.model_name,
    configData._name_or_path,
    configData.name,
    ggufRepoName,
  );

  if (inferred !== null) {
    return inferred;
  }

  const hiddenSize = configData.hidden_size ?? headDim * numHeads;
  const intermediateSize =
    configData.intermediate_size ??
    configData.ffn_hidden_size ??
    hiddenSize * 4;
  const vocabSize = configData.vocab_size ?? 32_000;
  const paramsPerLayer =
    numHeads * headDim * headDim * 4 +
    hiddenSize * intermediateSize * 2 +
    hiddenSize * 2;

  return (paramsPerLayer * numLayers + hiddenSize * vocabSize) / 1e9;
}

export function parseQuantizationBits(filename: string) {
  const value = filename.toUpperCase();

  if (value.includes("FP32") || value.includes("F32")) {
    return 32;
  }

  if (value.includes("FP16") || value.includes("F16")) {
    return 16;
  }

  if (value.includes("Q8")) {
    return 8;
  }

  if (value.includes("Q6")) {
    return 6;
  }

  if (value.includes("Q5")) {
    return 5;
  }

  if (value.includes("Q4")) {
    return 4;
  }

  if (value.includes("Q3")) {
    return 3;
  }

  if (value.includes("Q2")) {
    return 2;
  }

  const match = value.match(/(\d+)/);
  return match ? Number(match[1]) : 0;
}

export async function fetchQuantizations(
  ggufRepo: string,
  fetchImpl: typeof fetch,
) {
  const modelInfo: ModelInfo = await requestJson(
    `/api/models/${normalizeRepoInput(ggufRepo)}`,
    fetchImpl,
  );

  const siblings = Array.isArray(modelInfo.siblings) ? modelInfo.siblings : [];

  const quantizations = siblings
    .map((item) => item?.rfilename)
    .filter(
      (filename) =>
        typeof filename === "string" &&
        filename.endsWith(".gguf") &&
        !filename.startsWith("."),
    )
    .map((filename) => ({ filename, bits: parseQuantizationBits(filename) }))
    .filter((item) => item.bits > 0);

  return sortQuantizations(quantizations);
}

export async function fetchModelPayload(
  baseModel: string,
  fetchImpl: typeof fetch,
) {
  const configData = await requestConfig(baseModel, fetchImpl);

  const textConfig = flattenTextConfig(configData);
  const numLayers = parseInt(
    String(
      textConfig.num_hidden_layers ??
        textConfig.n_layer ??
        textConfig.hidden_layers ??
        32,
    ),
  );
  const numKvHeads = parseInt(
    String(
      textConfig.num_key_value_heads ??
        textConfig.num_attention_heads ??
        textConfig.n_head ??
        32,
    ),
  );
  // Fix: headDim should be calculated using num_attention_heads, not num_kv_heads (Issue #9)
  // For GQA models (like Llama 3, Gemma), these are different
  const numAttnHeads = parseInt(
    String(
      textConfig.num_attention_heads ??
        textConfig.n_head ??
        numKvHeads,
    ),
  );
  const headDim = parseInt(
    String(
      textConfig.head_dim ??
        Math.floor(
          (textConfig.hidden_size ?? textConfig.n_embd ?? 4_096) /
            Math.max(numAttnHeads, 1),
        ),
    ),
  );
  const maxContextLength = extractMaxContextLength(textConfig);
  const totalParamsBillions = calculateTotalParams(
    textConfig,
    numLayers,
    numKvHeads,
    headDim,
    baseModel,
  );

  return {
    modelName: baseModel,
    totalParamsBillions,
    numLayers,
    numKvHeads,
    headDim,
    maxContextLength,
    contexts: getAvailableContexts(maxContextLength),
  };
}

export async function resolveModelPayload(
  repoInput: string,
  fetchImpl = cachedFetch,
) {
  const { baseModel, ggufRepo } = await resolveBaseModel(repoInput, fetchImpl);

  // Use Promise.allSettled to handle each promise separately and preserve error context (Issue #6)
  const [modelResult, quantResult] = await Promise.allSettled([
    fetchModelPayload(baseModel, fetchImpl),
    fetchQuantizations(ggufRepo, fetchImpl),
  ]);

  // Handle model payload errors first (more critical)
  if (modelResult.status === "rejected") {
    const error = modelResult.reason;
    throw new Error(
      `Failed to fetch model metadata for ${baseModel}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  // Handle quantization errors
  if (quantResult.status === "rejected") {
    const error = quantResult.reason;
    throw new Error(
      `Failed to fetch quantizations for ${ggufRepo}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  const model = modelResult.value;
  const quantizations = quantResult.value;

  if (!quantizations.length) {
    throw new Error(
      `No GGUF quantizations were found in repository ${ggufRepo}. Please verify the repository contains .gguf files.`,
    );
  }

  return {
    ...model,
    quantizations,
  };
}
