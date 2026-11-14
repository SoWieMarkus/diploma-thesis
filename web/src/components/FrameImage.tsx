import type { Camera, Face, Vec3 } from "../api/schemas";
import { imageUrl } from "../api/client";
import { modelColor } from "../lib/colors";

const project = (p: Vec3, c: Camera): [number, number] => [
  (c.fx * p[0]) / p[2] + c.cx,
  (c.fy * p[1]) / p[2] + c.cy,
];

interface Props {
  frameId: number;
  camera: Camera;
  face?: Face;
  gazeModels?: string[];
  showLandmarks?: boolean;
  highlightModel?: string;
  style?: React.CSSProperties;
}

/** Frame JPG with landmarks, bounding box and projected gaze arrows drawn as SVG on top. */
export function FrameImage({
  frameId,
  camera,
  face,
  gazeModels = [],
  showLandmarks = true,
  highlightModel,
  style,
}: Props) {
  const center = face?.center;
  const radius = Math.max(camera.width, camera.height) / 400;
  return (
    <div
      className="frame-image"
      style={{ aspectRatio: `${camera.width} / ${camera.height}`, ...style }}
    >
      <img src={imageUrl(frameId)} alt={`Frame ${frameId}`} draggable={false} />
      <svg viewBox={`0 0 ${camera.width} ${camera.height}`} preserveAspectRatio="xMidYMid meet">
        {face?.bbox && (
          <rect
            x={face.bbox[0]}
            y={face.bbox[1]}
            width={face.bbox[2] - face.bbox[0]}
            height={face.bbox[3] - face.bbox[1]}
            fill="none"
            stroke="rgba(226,232,240,0.5)"
            strokeWidth={radius / 2}
            strokeDasharray={`${radius * 2} ${radius}`}
          />
        )}
        {showLandmarks &&
          face?.landmarks_2d?.map(([x, y], i) => (
            <circle key={i} cx={x} cy={y} r={radius} fill="#5eead4" opacity={0.85} />
          ))}
        {center &&
          face?.gazes
            .filter((g) => gazeModels.includes(g.model))
            .map((g) => {
              const end: Vec3 = [
                center[0] + 0.15 * g.vector[0],
                center[1] + 0.15 * g.vector[1],
                center[2] + 0.15 * g.vector[2],
              ];
              const [x0, y0] = project(center, camera);
              const [x1, y1] = project(end, camera);
              const highlighted = !highlightModel || g.model === highlightModel;
              return (
                <g key={g.id} opacity={highlighted ? 1 : 0.45}>
                  <line
                    x1={x0}
                    y1={y0}
                    x2={x1}
                    y2={y1}
                    stroke={modelColor(g.model)}
                    strokeWidth={radius * (highlighted ? 2.2 : 1.4)}
                    strokeLinecap="round"
                  />
                  <circle cx={x1} cy={y1} r={radius * 2.4} fill={modelColor(g.model)} />
                </g>
              );
            })}
        {center &&
          (() => {
            const [x, y] = project(center, camera);
            return (
              <circle
                cx={x}
                cy={y}
                r={radius * 2.5}
                fill="#fff"
                stroke="#000"
                strokeWidth={radius / 2}
              />
            );
          })()}
      </svg>
    </div>
  );
}
