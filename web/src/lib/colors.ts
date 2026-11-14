export const MODEL_COLORS: Record<string, string> = {
  "eth-xgaze": "#f472b6",
  "l2cs-gaze360": "#60a5fa",
  mpiifacegaze: "#fbbf24",
  "head-rotation": "#a78bfa",
};

export const modelColor = (model: string) => MODEL_COLORS[model] ?? "#ececec";

export const COLORS = {
  positive: "#4ade80",
  negative: "#f87171",
  warning: "#fbbf24",
  unsure: "#737373",
  missing: "#2e2e2e",
  accent: "#3ecf8e",
};
