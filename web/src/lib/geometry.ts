/**
 * Geometry in OpenCV camera coordinates (meters): x right, y down, z forward (towards the cyclist).
 * Mirrors gaze_estimator/core/geometry.py.
 */
import type { Vec3 } from "../api/schemas";

export interface Display {
  width: number;
  height: number;
  /** Distance of the camera below the top edge of the display. */
  offset: number;
}

export const DEFAULT_DISPLAY: Display = { width: 0.07, height: 0.15, offset: 0.01 };

export function displayBounds(d: Display) {
  return { minX: -d.width / 2, maxX: d.width / 2, minY: -d.offset, maxY: d.height - d.offset };
}

/** Intersection of the ray with the camera plane z = 0, or null if it points away. */
export function intersectCameraPlane(origin: Vec3, direction: Vec3): Vec3 | null {
  if (direction[2] === 0) return null;
  const r = -origin[2] / direction[2];
  if (!(r > 0)) return null;
  return [origin[0] + r * direction[0], origin[1] + r * direction[1], 0];
}

export function onDisplay(point: Vec3 | null, d: Display): boolean {
  if (!point) return false;
  const b = displayBounds(d);
  return point[0] >= b.minX && point[0] <= b.maxX && point[1] >= b.minY && point[1] <= b.maxY;
}

export function hitsDisplay(origin: Vec3, direction: Vec3, d: Display): boolean {
  return onDisplay(intersectCameraPlane(origin, direction), d);
}

const normalize = (v: Vec3): Vec3 => {
  const n = Math.hypot(...v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
};

/** Smallest angle (degrees) between the gaze ray and any point of the display (0 when it hits). */
export function angleToDisplay(origin: Vec3, direction: Vec3, d: Display, samples = 64): number {
  if (hitsDisplay(origin, direction, d)) return 0;
  const dir = normalize(direction);
  const b = displayBounds(d);
  let best = -1;
  const visit = (x: number, y: number) => {
    const v = normalize([x - origin[0], y - origin[1], -origin[2]]);
    best = Math.max(best, v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2]);
  };
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    visit(b.minX + t * (b.maxX - b.minX), b.minY);
    visit(b.minX + t * (b.maxX - b.minX), b.maxY);
    visit(b.minX, b.minY + t * (b.maxY - b.minY));
    visit(b.maxX, b.minY + t * (b.maxY - b.minY));
  }
  return (Math.acos(Math.min(1, Math.max(-1, best))) * 180) / Math.PI;
}

/** OpenCV camera coordinates -> three.js world coordinates (y up, z towards the viewer). */
export const toThree = (v: Vec3 | readonly number[]): [number, number, number] => [
  v[0],
  -v[1],
  -v[2],
];

export const degrees = (radians: number) => (radians * 180) / Math.PI;
