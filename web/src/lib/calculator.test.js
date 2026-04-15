import { describe, expect, it } from 'vitest';

import {
  calculateEstimate,
  calculateMemoryUsage,
  getAvailableContexts,
  sortQuantizations
} from './calculator.js';

describe('calculator', () => {
  it('caps context presets at the detected max context', () => {
    expect(getAvailableContexts(16_384)).toEqual([
      { label: '4K', tokens: 4_096 },
      { label: '8K', tokens: 8_192 },
      { label: '16K', tokens: 16_384 }
    ]);
  });

  it('sorts quantizations from highest precision to lowest precision', () => {
    expect(
      sortQuantizations([
        { filename: 'model-Q4_K_M.gguf', bits: 4 },
        { filename: 'model-Q8_0.gguf', bits: 8 },
        { filename: 'model-F16.gguf', bits: 16 }
      ]).map((item) => item.filename)
    ).toEqual(['model-F16.gguf', 'model-Q8_0.gguf', 'model-Q4_K_M.gguf']);
  });

  it('fills VRAM first and spills the remainder into RAM', () => {
    expect(calculateMemoryUsage(12, 5, 12, 64)).toMatchObject({
      totalNeededGb: 17,
      vramUsedGb: 12,
      ramUsedGb: 5,
      fits: true,
      remainingGb: 59
    });
  });

  it('reports a shortfall when total memory exceeds system capacity', () => {
    const estimate = calculateEstimate({
      paramsBillions: 27,
      bitsPerWeight: 8,
      numLayers: 64,
      numKvHeads: 4,
      headDim: 256,
      contextLength: 131_072,
      vramTotalGb: 24,
      ramTotalGb: 8
    });

    expect(estimate.modelSizeGb).toBe(27);
    expect(estimate.contextSizeGb).toBeCloseTo(17.179869184);
    expect(estimate.vramUsedGb).toBe(24);
    expect(estimate.ramUsedGb).toBeCloseTo(20.179869184);
    expect(estimate.fits).toBe(false);
    expect(estimate.shortfallGb).toBeCloseTo(12.179869184);
  });
});
