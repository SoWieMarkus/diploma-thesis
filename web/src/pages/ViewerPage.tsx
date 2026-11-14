import { useCallback, useEffect, useMemo } from "react";
import { useParams, useSearchParams } from "react-router";
import { useFrame, useFrames, useGazeTrack, useModels, useVideo } from "../api/client";
import { LOOKING_NAMES, QUALITY_NAMES, type Face } from "../api/schemas";
import { FrameImage } from "../components/FrameImage";
import { Timeline, lookingColor, qualityColor, type TimelineRow } from "../components/Timeline";
import { COLORS, modelColor } from "../lib/colors";
import { DEFAULT_DISPLAY, angleToDisplay, degrees, hitsDisplay } from "../lib/geometry";
import { balancedAccuracy, confusion, f1, fmt, mcc } from "../lib/metrics";
import { usePersistentState } from "../lib/storage";
import { GazeScene } from "../scene/GazeScene";

interface ViewerSettings {
  display: typeof DEFAULT_DISPLAY;
  square: boolean;
  showLandmarks: boolean;
  showFrustum: boolean;
  showAllModels: boolean;
}

const DEFAULT_SETTINGS: ViewerSettings = {
  display: DEFAULT_DISPLAY,
  square: false,
  showLandmarks: true,
  showFrustum: true,
  showAllModels: true,
};

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field">
      <span className="field-row">
        <span>{label}</span>
        <span className="mono">{(value * 100).toFixed(1)} cm</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function Checkbox({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="checkbox">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

export function ViewerPage() {
  const videoId = Number(useParams().videoId);
  const [search, setSearch] = useSearchParams();
  const { data: video } = useVideo(videoId);
  const { data: frames = [] } = useFrames(videoId);
  const { data: models } = useModels();
  const [settings, setSettings] = usePersistentState("viewer-settings", DEFAULT_SETTINGS);
  const display = settings.display;

  const faceName =
    search.get("face") ?? (models?.faces.includes("mediapipe") ? "mediapipe" : models?.faces[0]);
  const gazeName = search.get("gaze") ?? models?.gazes[0];
  const position = Math.min(
    Math.max(Number(search.get("frame") ?? 0), 0),
    Math.max(frames.length - 1, 0),
  );
  const summary = frames[position];
  const { data: frame } = useFrame(summary?.id);
  const { data: track = [] } = useGazeTrack(videoId, faceName, gazeName);

  const update = useCallback(
    (changes: Record<string, string | number>) => {
      setSearch(
        (previous) => {
          const next = new URLSearchParams(previous);
          Object.entries(changes).forEach(([k, v]) => next.set(k, String(v)));
          return next;
        },
        { replace: true },
      );
    },
    [setSearch],
  );

  const go = useCallback(
    (to: number) => update({ frame: Math.min(Math.max(to, 0), frames.length - 1) }),
    [update, frames.length],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement)
        return;
      const jump = event.shiftKey ? 10 : 1;
      if (event.key === "ArrowRight" || event.key === "d") go(position + jump);
      if (event.key === "ArrowLeft" || event.key === "a") go(position - jump);
      if (event.key === "Home") go(0);
      if (event.key === "End") go(frames.length - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, position, frames.length]);

  const face: Face | undefined = frame?.faces.find((f) => f.detector === faceName);
  const gaze = face?.gazes.find((g) => g.model === gazeName);
  const visibleModels = settings.showAllModels ? (models?.gazes ?? []) : gazeName ? [gazeName] : [];

  // Prediction for every frame of the video with the current display, for the timeline + metrics.
  const predictions = useMemo(() => {
    const byFrame = new Map(
      track.map((t) => [t.frame_id, hitsDisplay(t.center, t.vector, display)]),
    );
    return frames.map((f) => byFrame.get(f.id) ?? false);
  }, [track, frames, display]);

  const stats = useMemo(() => {
    const pairs = frames.flatMap((f, i) =>
      f.label_looking === 0 || f.label_looking === 1
        ? [{ label: f.label_looking === 1 && !f.label_blink, prediction: predictions[i] }]
        : [],
    );
    return { n: pairs.length, ...confusion(pairs) };
  }, [frames, predictions]);

  const rows: TimelineRow[] = useMemo(
    () => [
      {
        label: "Ground truth",
        colors: frames.map((f) => lookingColor(f.label_looking, f.label_blink)),
      },
      {
        label: "Prediction",
        colors: frames.map((f, i) =>
          f.detected[faceName ?? ""] === false
            ? "#171717"
            : predictions[i]
              ? COLORS.positive
              : COLORS.negative,
        ),
      },
      {
        label: `Quality ${faceName ?? ""}`,
        colors: frames.map((f) =>
          qualityColor(f.label_quality[faceName ?? ""], f.detected[faceName ?? ""]),
        ),
      },
    ],
    [frames, predictions, faceName],
  );

  const setDisplay = (changes: Partial<typeof display>) =>
    setSettings((s) => {
      const next = { ...s.display, ...changes };
      if (s.square) {
        if (changes.width !== undefined) next.height = changes.width;
        if (changes.height !== undefined) next.width = changes.height;
      }
      return { ...s, display: next };
    });

  if (!video) return <div className="empty">Loading…</div>;
  if (frames.length === 0) return <div className="empty">This video has no processed frames.</div>;

  const angle =
    face?.center && gaze ? angleToDisplay(face.center, gaze.vector, display) : undefined;
  const hit = face?.center && gaze ? hitsDisplay(face.center, gaze.vector, display) : undefined;

  return (
    <div className="viewer">
      <aside className="sidebar stack">
        <div className="card stack">
          <h3>{video.name}</h3>
          <label className="field">
            Face detector
            <select value={faceName} onChange={(e) => update({ face: e.target.value })}>
              {models?.faces.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <label className="field">
            Gaze model
            <select value={gazeName} onChange={(e) => update({ gaze: e.target.value })}>
              {models?.gazes.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="card stack">
          <h3>Virtual display</h3>
          <Slider
            label="Width"
            value={display.width}
            min={0.01}
            max={0.5}
            step={0.0025}
            onChange={(width) => setDisplay({ width })}
          />
          <Slider
            label="Height"
            value={display.height}
            min={0.01}
            max={0.5}
            step={0.0025}
            onChange={(height) => setDisplay({ height })}
          />
          <Slider
            label="Camera offset"
            value={display.offset}
            min={-0.1}
            max={0.5}
            step={0.0025}
            onChange={(offset) => setDisplay({ offset })}
          />
          <Checkbox
            label="Square (as in the thesis)"
            checked={settings.square}
            onChange={(square) =>
              setSettings((s) => ({
                ...s,
                square,
                display: square ? { ...s.display, height: s.display.width } : s.display,
              }))
            }
          />
          <button
            className="button"
            onClick={() => setSettings((s) => ({ ...s, display: DEFAULT_DISPLAY, square: false }))}
          >
            Reset to smartphone size
          </button>
        </div>

        <div className="card stack">
          <h3>Display</h3>
          <Checkbox
            label="Landmarks"
            checked={settings.showLandmarks}
            onChange={(showLandmarks) => setSettings((s) => ({ ...s, showLandmarks }))}
          />
          <Checkbox
            label="Camera frustum"
            checked={settings.showFrustum}
            onChange={(showFrustum) => setSettings((s) => ({ ...s, showFrustum }))}
          />
          <Checkbox
            label="All gaze models"
            checked={settings.showAllModels}
            onChange={(showAllModels) => setSettings((s) => ({ ...s, showAllModels }))}
          />
        </div>

        <div className="card">
          <h3>This video ({stats.n} labeled)</h3>
          <dl className="stats">
            <dt>F1</dt>
            <dd>{fmt(f1(stats))}</dd>
            <dt>MCC</dt>
            <dd>{fmt(mcc(stats))}</dd>
            <dt>Balanced acc.</dt>
            <dd>{fmt(balancedAccuracy(stats))}</dd>
            <dt>TP / FP</dt>
            <dd>
              {stats.tp} / {stats.fp}
            </dd>
            <dt>FN / TN</dt>
            <dd>
              {stats.fn} / {stats.tn}
            </dd>
          </dl>
        </div>
      </aside>

      <section className="scene">
        <GazeScene
          camera={video.camera}
          face={face}
          display={display}
          gazeModels={visibleModels}
          highlightModel={gazeName}
          showFrustum={settings.showFrustum}
        />
        <div className="legend">
          {visibleModels.map((m) => (
            <span key={m} style={{ fontWeight: m === gazeName ? 700 : 400 }}>
              <span className="swatch" style={{ background: modelColor(m) }} />
              {m}
            </span>
          ))}
        </div>
      </section>

      <section className="right">
        <div className="card" style={{ padding: 8 }}>
          {summary && (
            <FrameImage
              frameId={summary.id}
              camera={video.camera}
              face={face}
              gazeModels={visibleModels}
              highlightModel={gazeName}
              showLandmarks={settings.showLandmarks}
              style={{ maxHeight: "48vh" }}
            />
          )}
        </div>
        <div className="card">
          <h3>Frame {summary?.index}</h3>
          <dl className="stats">
            <dt>Label</dt>
            <dd>
              {summary?.label_looking == null ? "–" : LOOKING_NAMES[summary.label_looking]}
              {summary?.label_blink ? " (blink)" : ""}
            </dd>
            <dt>Prediction</dt>
            <dd>
              {hit === undefined ? (
                <span className="pill neutral">no face</span>
              ) : hit ? (
                <span className="pill positive">on display</span>
              ) : (
                <span className="pill negative">off display</span>
              )}
            </dd>
            <dt>Angle to display</dt>
            <dd>{angle === undefined ? "–" : `${angle.toFixed(1)}°`}</dd>
            <dt>Pitch / yaw</dt>
            <dd>
              {gaze ? `${degrees(gaze.pitch).toFixed(1)}° / ${degrees(gaze.yaw).toFixed(1)}°` : "–"}
            </dd>
            <dt>Distance</dt>
            <dd>{face?.distance ? `${(face.distance * 100).toFixed(1)} cm` : "–"}</dd>
            <dt>EAR (l / r)</dt>
            <dd>
              {face?.ear_left ? `${face.ear_left.toFixed(2)} / ${face.ear_right?.toFixed(2)}` : "–"}
            </dd>
            <dt>Landmark quality</dt>
            <dd>{face?.label_quality == null ? "–" : QUALITY_NAMES[face.label_quality]}</dd>
            <dt>Runtime face / gaze</dt>
            <dd>
              {face ? `${face.runtime_ms.toFixed(0)} ms` : "–"} /{" "}
              {gaze ? `${gaze.runtime_ms.toFixed(0)} ms` : "–"}
            </dd>
          </dl>
        </div>
      </section>

      <section className="bottom card timeline">
        <Timeline frames={frames} rows={rows} current={position} onSelect={go} />
        <div className="controls">
          <button className="button" onClick={() => go(position - 1)}>
            ←
          </button>
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={position}
            onChange={(e) => go(Number(e.target.value))}
          />
          <button className="button" onClick={() => go(position + 1)}>
            →
          </button>
          <span className="mono">
            {position + 1} / {frames.length}
          </span>
          <span className="muted">
            <kbd>←</kbd> <kbd>→</kbd> navigate, <kbd>Shift</kbd> ×10
          </span>
        </div>
      </section>
    </div>
  );
}
