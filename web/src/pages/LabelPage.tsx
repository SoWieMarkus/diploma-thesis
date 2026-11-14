import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router";
import {
  useFrame,
  useFrames,
  useModels,
  useUpdateFaceQuality,
  useUpdateFrameLabels,
  useVideo,
} from "../api/client";
import { LookingLabel, QualityLabel, QUALITY_NAMES } from "../api/schemas";
import { FrameImage } from "../components/FrameImage";
import { Timeline, lookingColor, qualityColor, type TimelineRow } from "../components/Timeline";
import { COLORS } from "../lib/colors";
import { usePersistentState } from "../lib/storage";

const SHORTCUTS: [string, string][] = [
  ["J / Y", "looks at the display"],
  ["N", "does not look at the display"],
  ["U", "unsure (excluded from evaluation)"],
  ["B", "toggle blink (eyes closed)"],
  ["1 / 2 / 3", "landmarks good / inaccurate / wrong"],
  ["Tab", "switch face detector"],
  ["← / →", "previous / next frame"],
  ["Space", "next unlabeled frame"],
  ["Backspace", "clear looking label"],
];

/** Keyboard driven labeling of every processed frame: looking, blink and landmark quality. */
export function LabelPage() {
  const videoId = Number(useParams().videoId);
  const [search, setSearch] = useSearchParams();
  const { data: video } = useVideo(videoId);
  const { data: frames = [] } = useFrames(videoId);
  const { data: models } = useModels();
  const updateFrame = useUpdateFrameLabels(videoId);
  const updateQuality = useUpdateFaceQuality(videoId);
  const [settings, setSettings] = usePersistentState("label-settings", {
    autoAdvance: true,
    landmarks: true,
  });
  const [detectorIndex, setDetectorIndex] = useState<number | null>(null);

  const detectors = useMemo(() => models?.faces ?? [], [models]);
  const preferred = Math.max(detectors.indexOf("mediapipe"), 0);
  const detector = detectors[(detectorIndex ?? preferred) % Math.max(detectors.length, 1)];
  const position = Math.min(
    Math.max(Number(search.get("frame") ?? 0), 0),
    Math.max(frames.length - 1, 0),
  );
  const summary = frames[position];
  const { data: frame } = useFrame(summary?.id);
  const face =
    frame?.id === summary?.id ? frame?.faces.find((f) => f.detector === detector) : undefined;

  const go = useCallback(
    (to: number) => {
      const clamped = Math.min(Math.max(to, 0), frames.length - 1);
      setSearch(
        (p) => {
          const next = new URLSearchParams(p);
          next.set("frame", String(clamped));
          return next;
        },
        { replace: true },
      );
    },
    [frames.length, setSearch],
  );

  const nextUnlabeled = useCallback(() => {
    const after = frames.findIndex((f, i) => i > position && f.label_looking === null);
    const any = after >= 0 ? after : frames.findIndex((f) => f.label_looking === null);
    if (any >= 0) go(any);
  }, [frames, position, go]);

  const setLooking = useCallback(
    (label: number | null) => {
      if (!summary) return;
      updateFrame.mutate({
        frameId: summary.id,
        labels: { label_looking: label, label_blink: summary.label_blink ?? false },
      });
      if (label !== null && settings.autoAdvance) go(position + 1);
    },
    [summary, updateFrame, settings.autoAdvance, go, position],
  );

  const toggleBlink = useCallback(() => {
    if (!summary) return;
    updateFrame.mutate({ frameId: summary.id, labels: { label_blink: !summary.label_blink } });
  }, [summary, updateFrame]);

  const setQuality = useCallback(
    (quality: number) => {
      if (!summary || !face?.detected) return;
      const current = summary.label_quality[face.detector];
      updateQuality.mutate({
        faceId: face.id,
        frameId: summary.id,
        detector: face.detector,
        quality: current === quality ? null : quality,
      });
    },
    [summary, face, updateQuality],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement)
        return;
      const key = event.key.toLowerCase();
      if (key === "j" || key === "y") setLooking(LookingLabel.YES);
      else if (key === "n") setLooking(LookingLabel.NO);
      else if (key === "u") setLooking(LookingLabel.UNSURE);
      else if (key === "backspace") setLooking(null);
      else if (key === "b") toggleBlink();
      else if (key === "1") setQuality(QualityLabel.GOOD);
      else if (key === "2") setQuality(QualityLabel.INACCURATE);
      else if (key === "3") setQuality(QualityLabel.WRONG);
      else if (key === "tab") setDetectorIndex((i) => (i ?? preferred) + 1);
      else if (key === "arrowright") go(position + (event.shiftKey ? 10 : 1));
      else if (key === "arrowleft") go(position - (event.shiftKey ? 10 : 1));
      else if (key === " ") nextUnlabeled();
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setLooking, toggleBlink, setQuality, go, position, nextUnlabeled, preferred]);

  const rows: TimelineRow[] = useMemo(
    () => [
      { label: "Looking", colors: frames.map((f) => lookingColor(f.label_looking)) },
      {
        label: "Blink",
        colors: frames.map((f) =>
          f.label_blink === null ? null : f.label_blink ? COLORS.warning : "#334155",
        ),
      },
      ...detectors.map((d) => ({
        label: `Quality ${d}`,
        colors: frames.map((f) => qualityColor(f.label_quality[d], f.detected[d])),
      })),
    ],
    [frames, detectors],
  );

  if (!video) return <div className="empty">Loading…</div>;
  if (frames.length === 0) return <div className="empty">This video has no processed frames.</div>;

  const labeled = frames.filter((f) => f.label_looking !== null).length;
  const quality = summary?.label_quality[detector ?? ""];

  return (
    <div className="labeler">
      <section className="image-area">
        {summary && (
          <FrameImage
            frameId={summary.id}
            camera={video.camera}
            face={face}
            showLandmarks={settings.landmarks}
          />
        )}
      </section>

      <aside className="panel stack">
        <div className="card stack">
          <div className="field-row">
            <h3 style={{ margin: 0 }}>Frame {summary?.index}</h3>
            <span className="mono muted">
              {labeled} / {frames.length} labeled
            </span>
          </div>
          <div className="bar">
            <div style={{ width: `${(labeled / frames.length) * 100}%` }} />
          </div>
        </div>

        <div className="card stack">
          <h3>Looks at the display?</h3>
          <div className="row">
            <button
              className={`button positive ${summary?.label_looking === LookingLabel.YES ? "active" : ""}`}
              onClick={() => setLooking(LookingLabel.YES)}
            >
              Yes <kbd>J</kbd>
            </button>
            <button
              className={`button negative ${summary?.label_looking === LookingLabel.NO ? "active" : ""}`}
              onClick={() => setLooking(LookingLabel.NO)}
            >
              No <kbd>N</kbd>
            </button>
            <button
              className={`button ${summary?.label_looking === LookingLabel.UNSURE ? "active" : ""}`}
              onClick={() => setLooking(LookingLabel.UNSURE)}
            >
              Unsure <kbd>U</kbd>
            </button>
          </div>
          <div className="row">
            <button
              className={`button warning ${summary?.label_blink ? "active" : ""}`}
              onClick={toggleBlink}
            >
              {summary?.label_blink ? "Eyes closed" : "Eyes open"} <kbd>B</kbd>
            </button>
          </div>
        </div>

        <div className="card stack">
          <h3>Landmark quality</h3>
          <div className="row">
            {detectors.map((d, i) => (
              <button
                key={d}
                className={`button ${d === detector ? "active" : ""}`}
                onClick={() => setDetectorIndex(i)}
              >
                {d}
              </button>
            ))}
            <kbd>Tab</kbd>
          </div>
          {face?.detected ? (
            <div className="row">
              {QUALITY_NAMES.map((name, value) => (
                <button
                  key={name}
                  className={`button ${["positive", "warning", "negative"][value]} ${quality === value ? "active" : ""}`}
                  onClick={() => setQuality(value)}
                >
                  {name} <kbd>{value + 1}</kbd>
                </button>
              ))}
            </div>
          ) : (
            <span className="pill neutral">No face detected by {detector}</span>
          )}
        </div>

        <div className="card stack">
          <h3>Options</h3>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={settings.autoAdvance}
              onChange={(e) => setSettings((s) => ({ ...s, autoAdvance: e.target.checked }))}
            />
            Jump to the next frame after labeling
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={settings.landmarks}
              onChange={(e) => setSettings((s) => ({ ...s, landmarks: e.target.checked }))}
            />
            Show landmarks
          </label>
          <button className="button" onClick={nextUnlabeled}>
            Next unlabeled <kbd>Space</kbd>
          </button>
        </div>

        <div className="card">
          <h3>Shortcuts</h3>
          <div className="shortcut-list">
            {SHORTCUTS.map(([key, text]) => (
              <span key={key} style={{ display: "contents" }}>
                <kbd>{key}</kbd>
                <span>{text}</span>
              </span>
            ))}
          </div>
        </div>
      </aside>

      <section className="bottom card timeline">
        <Timeline frames={frames} rows={rows} current={position} onSelect={go} />
        <div className="controls">
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={position}
            onChange={(e) => go(Number(e.target.value))}
          />
          <span className="mono">
            {position + 1} / {frames.length}
          </span>
        </div>
      </section>
    </div>
  );
}
