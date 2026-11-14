export interface Confusion {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export function confusion(pairs: { label: boolean; prediction: boolean }[]): Confusion {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  for (const { label, prediction } of pairs) {
    if (label && prediction) c.tp++;
    else if (!label && prediction) c.fp++;
    else if (!label && !prediction) c.tn++;
    else c.fn++;
  }
  return c;
}

export function f1(c: Confusion) {
  const denominator = 2 * c.tp + c.fp + c.fn;
  return denominator ? (2 * c.tp) / denominator : NaN;
}

export function mcc(c: Confusion) {
  const denominator = Math.sqrt((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn));
  return denominator ? (c.tp * c.tn - c.fp * c.fn) / denominator : 0;
}

export function balancedAccuracy(c: Confusion) {
  const tpr = c.tp + c.fn ? c.tp / (c.tp + c.fn) : NaN;
  const tnr = c.tn + c.fp ? c.tn / (c.tn + c.fp) : NaN;
  return (tpr + tnr) / 2;
}

export const fmt = (value: number, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "–";
