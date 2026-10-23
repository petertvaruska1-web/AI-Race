// Mirrors ml/src/airace_ml/model/shape.py; the counts must match exactly.

export const VOCAB_SIZE = 4096;
const FFN_STEP = 64;

/** 64 * ceil(8 * dModel / 3 / 64), in exact integer arithmetic. */
export function ffnHidden(dModel: number): number {
  return FFN_STEP * Math.floor((8 * dModel + 3 * FFN_STEP - 1) / (3 * FFN_STEP));
}

/** Unique trainable parameters (the output head is tied to the embedding). */
export function paramCount(nLayer: number, dModel: number, vocab = VOCAB_SIZE): number {
  const d = dModel;
  const f = ffnHidden(d);
  const perBlock = 4 * d * d + 3 * d * f + 2 * d; // q/k/v/o, gate/up/down, two norms
  return vocab * d + nLayer * perBlock + d; // embedding, blocks, final norm
}

export function trainingFlops(params: number, tokens: number): number {
  return 6 * params * tokens;
}
