export const CONTEXT_PRESETS = [
  { label: '4K', tokens: 4_096 },
  { label: '8K', tokens: 8_192 },
  { label: '16K', tokens: 16_384 },
  { label: '32K', tokens: 32_768 },
  { label: '64K', tokens: 65_536 },
  { label: '128K', tokens: 131_072 },
  { label: '256K', tokens: 262_144 }
];

export function getAvailableContexts(maxContext) {
  return CONTEXT_PRESETS.filter((context) => context.tokens <= maxContext);
}

export function sortQuantizations(quantizations) {
  return [...quantizations].sort((left, right) => {
    if (right.bits !== left.bits) {
      return right.bits - left.bits;
    }

    return left.filename.localeCompare(right.filename);
  });
}

export function calculateModelSizeGb(paramsBillions, bitsPerWeight) {
  return (paramsBillions * bitsPerWeight) / 8;
}

export function calculateContextSizeGb(numLayers, numKvHeads, headDim, contextLength) {
  return (2 * numLayers * numKvHeads * headDim * contextLength) / 1e9;
}

export function calculateMemoryUsage(modelSizeGb, contextSizeGb, vramTotalGb, ramTotalGb) {
  const safeVramTotal = Math.max(0, Number(vramTotalGb) || 0);
  const safeRamTotal = Math.max(0, Number(ramTotalGb) || 0);
  const totalNeededGb = modelSizeGb + contextSizeGb;
  const vramUsedGb = Math.min(totalNeededGb, safeVramTotal);
  const ramUsedGb = Math.max(0, totalNeededGb - vramUsedGb);
  const totalCapacityGb = safeVramTotal + safeRamTotal;
  const fits = totalNeededGb <= totalCapacityGb;

  return {
    totalNeededGb,
    vramUsedGb,
    ramUsedGb,
    totalCapacityGb,
    fits,
    remainingGb: fits ? totalCapacityGb - totalNeededGb : 0,
    shortfallGb: fits ? 0 : totalNeededGb - totalCapacityGb,
    vramUtilization: safeVramTotal > 0 ? (vramUsedGb / safeVramTotal) * 100 : 0,
    ramUtilization: safeRamTotal > 0 ? (ramUsedGb / safeRamTotal) * 100 : 0
  };
}

export function calculateEstimate(input) {
  const modelSizeGb = calculateModelSizeGb(input.paramsBillions, input.bitsPerWeight);
  const contextSizeGb = calculateContextSizeGb(
    input.numLayers,
    input.numKvHeads,
    input.headDim,
    input.contextLength
  );

  return {
    modelSizeGb,
    contextSizeGb,
    ...calculateMemoryUsage(modelSizeGb, contextSizeGb, input.vramTotalGb, input.ramTotalGb)
  };
}

export function formatQuantizationLabel(bitsPerWeight) {
  return bitsPerWeight >= 16 ? `FP${bitsPerWeight}` : `Q${bitsPerWeight}`;
}
