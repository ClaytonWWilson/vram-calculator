<script lang="ts">
  import Formula from "$lib/Formula.svelte";
  import { calculateEstimate, formatQuantizationLabel } from "$lib/calculator";
  import {
    resolvedTheme,
    setThemePreference,
    themePreference,
    type ThemePreference,
  } from "$lib/theme";
  import type { ApiResponse, ModelResponse } from "./api/model/+server";

  let repoInput = "unsloth/Qwen3.5-4B-GGUF";
  let vramGb = 12;
  let ramGb = 64;
  let loading = false;
  let error = "";
  let model: ModelResponse | null = null;
  let selectedContextIndex = 0;
  let selectedQuantizationIndex = 0;

  $: contexts = model?.contexts ?? [];
  $: quantizations = model?.quantizations ?? [];

  $: if (contexts.length > 0 && selectedContextIndex >= contexts.length) {
    selectedContextIndex = 0;
  }

  $: if (
    quantizations.length > 0 &&
    selectedQuantizationIndex >= quantizations.length
  ) {
    selectedQuantizationIndex = 0;
  }

  $: selectedContext = contexts[selectedContextIndex] ?? null;
  $: selectedQuantization = quantizations[selectedQuantizationIndex] ?? null;

  $: estimate =
    model && selectedContext && selectedQuantization
      ? calculateEstimate({
          paramsBillions: model.totalParamsBillions,
          bitsPerWeight: selectedQuantization.bits,
          numLayers: model.numLayers,
          numKvHeads: model.numKvHeads,
          headDim: model.headDim,
          contextLength: selectedContext.tokens,
          vramTotalGb: vramGb,
          ramTotalGb: ramGb,
        })
      : null;

  async function loadModel() {
    loading = true;
    error = "";

    try {
      const response = await fetch("/api/model", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          repo: repoInput,
        }),
      });

      const payload: ApiResponse = await response.json();

      if (payload.success === false) {
        throw new Error(
          payload.error.message ?? "Unable to load model metadata.",
        );
      }

      model = payload.data;
      selectedContextIndex = 0;
      selectedQuantizationIndex = 0;
    } catch (requestError) {
      model = null;
      error =
        requestError instanceof Error
          ? requestError.message
          : "Unable to load model metadata.";
    } finally {
      loading = false;
    }
  }

  function formatGb(value) {
    return `${value.toFixed(2)} GB`;
  }

  function formatPercent(value) {
    return `${Math.min(value, 100).toFixed(1)}%`;
  }

  const themeOptions: { label: string; value: ThemePreference }[] = [
    { label: "System", value: "system" },
    { label: "Light", value: "light" },
    { label: "Dark", value: "dark" },
  ];
</script>

<svelte:head>
  <title>VRAM Calculator</title>
  <meta
    name="description"
    content="Estimate VRAM and RAM requirements for GGUF models from Hugging Face."
  />
</svelte:head>

<div class="page-shell">
  <section class="hero">
    <form class="control-card" on:submit|preventDefault={loadModel}>
      <div class="hero-copy">
        <div class="hero-meta">
          <div class="theme-toggle" role="group" aria-label="Theme preference">
            {#each themeOptions as option}
              <button
                class:selected={$themePreference === option.value}
                type="button"
                title={option.label}
                aria-label={option.label}
                aria-pressed={$themePreference === option.value}
                on:click={() => setThemePreference(option.value)}
              >
                {#if option.value === "system"}
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <rect x="4.5" y="5.5" width="15" height="10" rx="2" />
                    <path d="M9 18.5h6" />
                    <path d="M12 15.5v3" />
                  </svg>
                {:else if option.value === "light"}
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <circle cx="12" cy="12" r="4" />
                    <path d="M12 2.5v3" />
                    <path d="M12 18.5v3" />
                    <path d="M21.5 12h-3" />
                    <path d="M5.5 12h-3" />
                    <path d="M18.7 5.3l-2.1 2.1" />
                    <path d="M7.4 16.6l-2.1 2.1" />
                    <path d="M18.7 18.7l-2.1-2.1" />
                    <path d="M7.4 7.4L5.3 5.3" />
                  </svg>
                {:else}
                  <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path
                      d="M15.5 3.5a7.8 7.8 0 1 0 5 13.8A8.8 8.8 0 1 1 15.5 3.5Z"
                    />
                  </svg>
                {/if}
              </button>
            {/each}
          </div>
        </div>
        <h1 class="hero-title">GGUF Fit Estimator</h1>
        <p class="lede">
          Load a Hugging Face GGUF repo, choose context size and quantization,
          and see if you can load it on your system.
        </p>
        <p class="theme-caption">Current appearance: {$resolvedTheme}</p>
      </div>
      <div class="field-grid">
        <label class="field field-wide">
          <span>GGUF repo</span>
          <input
            bind:value={repoInput}
            type="text"
            placeholder="unsloth/Qwen3.5-4B-GGUF"
          />
        </label>

        <label class="field">
          <span>VRAM (GB)</span>
          <input bind:value={vramGb} type="number" min="0" step="0.5" />
        </label>

        <label class="field">
          <span>RAM (GB)</span>
          <input bind:value={ramGb} type="number" min="0" step="0.5" />
        </label>
      </div>

      <div class="actions">
        <button class="load-button" type="submit" disabled={loading}>
          {#if loading}Loading model…{:else}Load model{/if}
        </button>

        <!-- <p class="helper-text">
          Uses the Hugging Face model card to resolve the base model, then reads
          <code>config.json</code> and GGUF filenames.
        </p> -->
      </div>

      {#if error}
        <p class="message error">{error}</p>
      {/if}
    </form>
  </section>

  {#if model}
    <section class="dashboard">
      <article class="panel summary-panel">
        <div class="panel-heading">
          <p class="eyebrow">Model</p>
          <h2>{model.modelName}</h2>
        </div>

        <dl class="stats-grid">
          <div>
            <dt>Params</dt>
            <dd>{model.totalParamsBillions.toFixed(2)}B</dd>
          </div>
          <div>
            <dt>Layers</dt>
            <dd>{model.numLayers}</dd>
          </div>
          <div>
            <dt>KV heads</dt>
            <dd>{model.numKvHeads}</dd>
          </div>
          <div>
            <dt>Head dim</dt>
            <dd>{model.headDim}</dd>
          </div>
          <div>
            <dt>Max context</dt>
            <dd>{model.maxContextLength.toLocaleString()}</dd>
          </div>
        </dl>
      </article>

      <article class="panel selector-panel">
        <div class="panel-heading">
          <p class="eyebrow">Selection</p>
          <h2>Context and quantization</h2>
        </div>

        <div class="selector-grid">
          <div class="field context-quant">
            <span>Context size</span>
            <div class="slider-shell">
              <div class="slider-header">
                <strong>{selectedContext?.label}</strong>
                <span>{selectedContext?.tokens.toLocaleString()} tokens</span>
              </div>
              <input
                class="context-slider"
                bind:value={selectedContextIndex}
                type="range"
                min="0"
                max={Math.max(contexts.length - 1, 0)}
                step="1"
              />
              <div class="slider-scale" aria-hidden="true">
                {#each contexts as context}
                  <span
                    class:selected={selectedContext?.tokens === context.tokens}
                    >{context.label}</span
                  >
                {/each}
              </div>
            </div>
          </div>

          <label class="field context-quant">
            <span>Quantization</span>
            <select bind:value={selectedQuantizationIndex}>
              {#each quantizations as quantization, index}
                <option value={index}>
                  {formatQuantizationLabel(quantization.bits)} · {quantization.filename}
                </option>
              {/each}
            </select>
          </label>
        </div>
      </article>

      {#if estimate}
        <article class="panel results-panel">
          <div class="panel-heading">
            <p class="eyebrow">Estimate</p>
            <h2>
              {#if estimate.fits}
                Fits within system capacity
              {:else}
                Capacity shortfall detected
              {/if}
            </h2>
          </div>

          <div class="breakdown-grid">
            <div>
              <span>Model size</span>
              <strong>{formatGb(estimate.modelSizeGb)}</strong>
            </div>
            <div>
              <span>Context size</span>
              <strong>{formatGb(estimate.contextSizeGb)}</strong>
            </div>
            <div>
              <span>Total needed</span>
              <strong>{formatGb(estimate.totalNeededGb)}</strong>
            </div>
            <div>
              <span>VRAM used</span>
              <strong>{formatGb(estimate.vramUsedGb)}</strong>
            </div>
            <div>
              <span>RAM used</span>
              <strong>{formatGb(estimate.ramUsedGb)}</strong>
            </div>
            <div>
              <span>Status</span>
              <strong class:ok={estimate.fits} class:warn={!estimate.fits}>
                {#if estimate.fits}
                  {formatGb(estimate.remainingGb)} remaining
                {:else}
                  {formatGb(estimate.shortfallGb)} shortfall
                {/if}
              </strong>
            </div>
          </div>

          <div class="meter-stack">
            <div class="meter-label">
              <span>VRAM</span>
              <strong
                >{formatGb(estimate.vramUsedGb)} / {Number(vramGb).toFixed(1)} GB</strong
              >
            </div>
            <div class="meter-track">
              <div
                class="meter-fill"
                style={`width: ${Math.min(estimate.vramUtilization, 100)}%`}
              ></div>
            </div>
            <p class="meter-note">
              {formatPercent(estimate.vramUtilization)} utilized
            </p>
          </div>

          <div class="meter-stack">
            <div class="meter-label">
              <span>RAM spillover</span>
              <strong
                >{formatGb(estimate.ramUsedGb)} / {Number(ramGb).toFixed(1)} GB</strong
              >
            </div>
            <div class="meter-track soft">
              <div
                class="meter-fill warm"
                style={`width: ${Math.min(estimate.ramUtilization, 100)}%`}
              ></div>
            </div>
            <p class="meter-note">
              {formatPercent(estimate.ramUtilization)} utilized
            </p>
          </div>
        </article>
      {/if}
    </section>

    <section class="notes panel">
      <p class="eyebrow">How it works</p>
      <p>
        This tool uses the provided huggingface repo and attempts to fetch the
        base model of the GGUF. It pulls the required values from the
        config.json of the base model and uses those to estimate the necessary
        VRAM required for running the model. This is a rough estimate and does
        not account for runtime overhead or batching.
      </p>
    </section>

    <section class="notes panel">
      <p class="eyebrow">Definitions And Calculations</p>
      <div class="explanation-grid">
        <div>
          <h3>Definitions</h3>
          <ul class="info-list">
            <li>
              <strong>Params:</strong> total model parameters in billions.
            </li>
            <li>
              <strong>Layers:</strong> transformer block count from the model config.
            </li>
            <li>
              <strong>KV heads:</strong> the number of key/value heads used for KV
              cache sizing.
            </li>
            <li><strong>Head dim:</strong> width of each attention head.</li>
            <li>
              <strong>Context size:</strong> selected token window for the estimate.
            </li>
            <li>
              <strong>Model size:</strong> estimated weight memory for the chosen
              quantization.
            </li>
            <li>
              <strong>Total needed:</strong> model size plus context memory.
            </li>
            <li>
              <strong>VRAM used:</strong> the portion of total memory filled by GPU
              memory first.
            </li>
            <li>
              <strong>RAM used:</strong> overflow beyond VRAM that spills into system
              memory.
            </li>
          </ul>
        </div>

        <div>
          <h3>Formulas</h3>
          <div class="formula-block">
            <p><strong>Model size (GB)</strong></p>
            <Formula
              expression={"\\mathrm{model\\_size\\_gb} = \\frac{\\mathrm{params\\_billions} \\cdot \\mathrm{bits\\_per\\_weight}}{8}"}
            />
          </div>
          <div class="formula-block">
            <p><strong>Context size (GB)</strong></p>
            <Formula
              expression={"\\mathrm{context\\_size\\_gb} = \\frac{2 \\cdot \\mathrm{layers} \\cdot \\mathrm{kv\\_heads} \\cdot \\mathrm{head\\_dim} \\cdot \\mathrm{context\\_length}}{10^9}"}
            />
          </div>
          <div class="formula-block">
            <p><strong>Total memory needed</strong></p>
            <Formula
              expression={"\\mathrm{total\\_needed\\_gb} = \\mathrm{model\\_size\\_gb} + \\mathrm{context\\_size\\_gb}"}
            />
          </div>
          <div class="formula-block">
            <p><strong>Memory placement</strong></p>
            <Formula
              expression={"\\mathrm{vram\\_used\\_gb} = \\min(\\mathrm{total\\_needed\\_gb}, \\mathrm{vram\\_total\\_gb})"}
            />
            <br />
            <Formula
              expression={"\\mathrm{ram\\_used\\_gb} = \\max(0, \\mathrm{total\\_needed\\_gb} - \\mathrm{vram\\_used\\_gb})"}
            />
          </div>
        </div>
      </div>
    </section>
  {/if}
</div>

<style>
  .page-shell {
    padding: 2rem 1.25rem 3rem;
  }

  .hero,
  .dashboard,
  .notes {
    max-width: 1100px;
    margin: 0 auto;
  }

  .hero {
    display: grid;
    gap: 1.5rem;
    align-items: start;
    max-width: 1100px;
    border-radius: 32px;
    box-shadow: var(--shadow);
  }

  .hero-meta {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: center;
    margin-bottom: 1rem;
    float: right;
  }

  .hero-title {
    max-width: none;
    margin-bottom: 1rem;
    font-size: clamp(4rem, 10vw, 6rem);
    line-height: 0.9;
    letter-spacing: -0.05em;
  }

  .eyebrow {
    margin: 0 0 0.5rem;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
  }

  h1,
  h2,
  p {
    margin-top: 0;
  }

  h1 {
    max-width: 12ch;
    margin-bottom: 1rem;
    font-size: clamp(2.8rem, 7vw, 5.6rem);
    line-height: 0.94;
    letter-spacing: -0.04em;
  }

  .lede {
    max-width: 56ch;
    font-size: 1.05rem;
    line-height: 1.6;
    color: var(--muted);
  }

  .theme-caption {
    margin-top: 0.9rem;
    color: var(--muted);
    font-size: 0.9rem;
    text-transform: capitalize;
  }

  .panel,
  .control-card {
    backdrop-filter: blur(18px);
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 28px;
    box-shadow: var(--shadow);
  }

  .control-card {
    width: 100%;
    padding: 1.25rem;
  }

  .field-grid,
  .selector-grid,
  .breakdown-grid,
  .stats-grid {
    display: grid;
    gap: 1rem;
  }

  .field-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .field-wide {
    grid-column: span 2;
  }

  .field {
    display: grid;
    gap: 0.5rem;
  }

  .field span {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--muted);
  }

  .context-quant {
    display: grid;
    grid-auto-flow: row;
    align-content: start;
  }

  .context-quant > select,
  .context-quant > .slider-shell {
    height: 120px;
  }

  @media (max-width: 640px) {
    .context-quant > select,
    .context-quant > .slider-shell {
      height: 150px;
    }
  }

  .context-quant > select {
    font-size: 16pt;
  }

  input,
  select {
    width: 100%;
    padding: 0.95rem 1rem;
    border: 1px solid var(--field-border);
    border-radius: 16px;
    background: var(--field-bg);
    color: var(--ink);
  }

  .theme-toggle {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    padding: 0.35rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--panel-strong);
    margin-left: auto;
  }

  .theme-toggle button {
    border: 0;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.6rem;
    height: 2.6rem;
    padding: 0;
    background: transparent;
    color: var(--muted);
    transition:
      background 140ms ease,
      color 140ms ease,
      transform 140ms ease;
  }

  .theme-toggle button:hover {
    transform: translateY(-1px);
    color: var(--ink);
  }

  .theme-toggle button.selected {
    background: var(--accent-soft);
    color: var(--accent);
  }

  .theme-toggle svg {
    width: 1.1rem;
    height: 1.1rem;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .actions {
    display: flex;
    gap: 1rem;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
  }

  .load-button {
    border: none;
    border-radius: 999px;
    padding: 0.95rem 1.4rem;
    background: var(--accent-gradient);
    color: white;
    font-weight: 700;
  }

  .load-button:disabled {
    opacity: 0.7;
    cursor: progress;
  }

  .message,
  .meter-note,
  .notes p:last-child {
    color: var(--muted);
    line-height: 1.5;
  }

  .message.error,
  .warn {
    color: var(--warn);
  }

  .ok {
    color: var(--good);
  }

  .dashboard {
    display: grid;
    gap: 1.25rem;
    margin-top: 1.5rem;
  }

  .panel {
    padding: 1.25rem;
  }

  .panel-heading {
    margin-bottom: 1rem;
  }

  .panel-heading h2 {
    margin: 0;
    font-size: clamp(1.4rem, 2vw, 2rem);
    line-height: 1.05;
  }

  .stats-grid {
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }

  .stats-grid dt,
  .breakdown-grid span {
    color: var(--muted);
    font-size: 0.85rem;
  }

  .stats-grid dd,
  .breakdown-grid strong,
  .meter-label strong {
    margin: 0.35rem 0 0;
    font-size: 1.15rem;
  }

  .selector-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .slider-shell {
    padding: 1rem 1rem 0.85rem;
    border: 1px solid var(--field-border);
    border-radius: 22px;
    background: var(--field-bg);
  }

  .slider-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: baseline;
    margin-bottom: 0.9rem;
  }

  .slider-header strong {
    font-size: 1.1rem;
  }

  .slider-header span {
    color: var(--muted);
    font-size: 0.9rem;
  }

  .context-slider {
    padding: 0;
    accent-color: var(--accent);
  }

  .slider-scale {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    margin-top: 0.7rem;
    font-size: 0.78rem;
    color: var(--muted);
  }

  .slider-scale span.selected {
    color: var(--accent);
    font-weight: 700;
  }

  .breakdown-grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-bottom: 1.5rem;
  }

  .breakdown-grid > div {
    padding: 1rem;
    border-radius: 20px;
    background: var(--panel-strong);
    border: 1px solid var(--surface-border);
  }

  .meter-stack + .meter-stack {
    margin-top: 1rem;
  }

  .meter-label {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.5rem;
  }

  .meter-track {
    overflow: hidden;
    height: 14px;
    border-radius: 999px;
    background: var(--track-cool);
  }

  .meter-track.soft {
    background: var(--track-warm);
  }

  .meter-fill {
    height: 100%;
    border-radius: inherit;
    background: var(--fill-cool);
  }

  .meter-fill.warm {
    background: var(--fill-warm);
  }

  .notes {
    margin-top: 1.5rem;
  }

  .explanation-grid {
    display: grid;
    gap: 1.25rem;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .explanation-grid h3 {
    margin: 0 0 0.75rem;
    font-size: 1.05rem;
  }

  .info-list {
    margin: 0;
    padding-left: 1.2rem;
    color: var(--muted);
    line-height: 1.6;
  }

  .formula-block + .formula-block {
    margin-top: 0.85rem;
  }

  .formula-block p {
    margin-bottom: 0.35rem;
    color: var(--muted);
  }

  .formula-block :global(.formula) {
    padding: 0.85rem 1rem;
    border-radius: 16px;
    background: var(--panel-strong);
    border: 1px solid var(--surface-border);
  }

  .formula-block :global(.katex) {
    font-size: 1.02rem;
  }

  @media (max-width: 899px) {
    .field-grid,
    .selector-grid,
    .stats-grid,
    .breakdown-grid,
    .explanation-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .field-wide {
      grid-column: span 2;
    }
  }

  @media (max-width: 640px) {
    .page-shell {
      padding-inline: 0.85rem;
    }

    h1 {
      max-width: none;
    }

    .field-grid,
    .selector-grid,
    .stats-grid,
    .breakdown-grid,
    .explanation-grid {
      grid-template-columns: 1fr;
    }

    .field-wide {
      grid-column: auto;
    }

    .hero-meta,
    .slider-header,
    .actions,
    .meter-label {
      flex-direction: column;
      align-items: stretch;
    }

    .slider-scale {
      flex-wrap: wrap;
    }
  }
</style>
