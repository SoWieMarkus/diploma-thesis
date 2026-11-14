import { useEffect, useRef } from "react";
import type { FrameSummary } from "../api/schemas";
import { COLORS } from "../lib/colors";

export interface TimelineRow {
  label: string;
  /** Color per frame (same order as `frames`), null = nothing to draw. */
  colors: (string | null)[];
}

interface Props {
  frames: FrameSummary[];
  rows: TimelineRow[];
  current: number;
  onSelect: (position: number) => void;
}

/** Compact canvas timeline (like the strip plots of the thesis): one row per signal, click to jump. */
export function Timeline({ frames, rows, current, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth * dpr;
    const height = canvas.clientHeight * dpr;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx || frames.length === 0) return;

    const labelWidth = 110 * dpr;
    const rowHeight = height / rows.length;
    const cell = (width - labelWidth) / frames.length;
    ctx.clearRect(0, 0, width, height);
    ctx.font = `${11 * dpr}px system-ui, sans-serif`;
    ctx.textBaseline = "middle";

    rows.forEach((row, r) => {
      const y = r * rowHeight;
      ctx.fillStyle = "#8b98b0";
      ctx.fillText(row.label, 8 * dpr, y + rowHeight / 2);
      row.colors.forEach((color, i) => {
        ctx.fillStyle = color ?? COLORS.missing;
        ctx.fillRect(
          labelWidth + i * cell,
          y + 2 * dpr,
          Math.max(cell - 0.3 * dpr, 1),
          rowHeight - 4 * dpr,
        );
      });
    });

    const x = labelWidth + current * cell + cell / 2;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }, [frames, rows, current]);

  const handleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left - 110;
    if (x < 0) return;
    const position = Math.floor((x / (rect.width - 110)) * frames.length);
    onSelect(Math.min(frames.length - 1, Math.max(0, position)));
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      style={{ height: Math.max(48, rows.length * 22) }}
    />
  );
}

export const lookingColor = (label: number | null | undefined, blink?: boolean | null) => {
  if (label === null || label === undefined) return null;
  if (label === 2) return COLORS.unsure;
  if (blink) return COLORS.warning;
  return label === 1 ? COLORS.positive : COLORS.negative;
};

export const qualityColor = (label: number | null | undefined, detected: boolean | undefined) => {
  if (detected === false) return "#0f172a";
  if (label === null || label === undefined) return null;
  return [COLORS.positive, COLORS.warning, COLORS.negative][label] ?? null;
};
