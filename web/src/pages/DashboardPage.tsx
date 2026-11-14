import { Link } from "react-router";
import { imageUrl, useVideos } from "../api/client";
import type { VideoSummary } from "../api/schemas";

function Progress({ label, value, total }: { label: string; value: number; total: number }) {
  const ratio = total > 0 ? value / total : 0;
  return (
    <div className="progress">
      <span className="muted">{label}</span>
      <div className="bar">
        <div style={{ width: `${ratio * 100}%` }} />
      </div>
      <span className="mono">{Math.round(ratio * 100)}%</span>
    </div>
  );
}

function VideoCard({ video }: { video: VideoSummary }) {
  const thumbnail = video.thumbnail_frame_id ? imageUrl(video.thumbnail_frame_id) : undefined;
  return (
    <div className="card video-card">
      <div
        className="thumb"
        style={{
          backgroundImage: thumbnail ? `url(${thumbnail})` : undefined,
          aspectRatio: `${video.width} / ${video.height}`,
          maxHeight: 260,
          backgroundSize: "contain",
        }}
      />
      <div className="title">
        <h2 style={{ fontSize: 16 }}>{video.name}</h2>
        <span className="muted mono">
          {video.width}×{video.height} · {video.fps.toFixed(0)} fps · {video.processed_frames}{" "}
          frames
        </span>
      </div>
      <div className="stack" style={{ gap: 6 }}>
        <Progress
          label="Looking labeled"
          value={video.labeled_looking}
          total={video.processed_frames}
        />
        <Progress
          label="Blink labeled"
          value={video.labeled_blink}
          total={video.processed_frames}
        />
        {Object.entries(video.faces).map(([detector, stats]) => (
          <Progress
            key={detector}
            label={`${detector} detected`}
            value={stats.detected}
            total={stats.frames}
          />
        ))}
        {Object.entries(video.faces).map(([detector, stats]) => (
          <Progress
            key={`${detector}-q`}
            label={`${detector} rated`}
            value={stats.quality_labeled}
            total={stats.detected}
          />
        ))}
      </div>
      <div className="actions">
        <Link className="button primary" to={`/videos/${video.id}/view`}>
          Open viewer
        </Link>
        <Link className="button" to={`/videos/${video.id}/label`}>
          Label frames
        </Link>
      </div>
    </div>
  );
}

export function DashboardPage() {
  const { data: videos, isLoading, error } = useVideos();

  if (isLoading) return <div className="empty">Loading…</div>;
  if (error) {
    return (
      <div className="empty">
        <h2>API not reachable</h2>
        <p>
          Start it with <code>uv run gaze-serve</code>.
        </p>
      </div>
    );
  }
  if (!videos?.length) {
    return (
      <div className="empty">
        <h2>No processed videos yet</h2>
        <p>
          Put videos into <code>data/input</code> and run <code>uv run gaze-process</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="page stack">
      <div>
        <h1>Videos</h1>
        <p className="muted">
          {videos.length} processed video{videos.length === 1 ? "" : "s"}. Open the viewer to
          explore the face and gaze estimation in 3D, or label frames for the evaluation.
        </p>
      </div>
      <div className="grid">
        {videos.map((video) => (
          <VideoCard key={video.id} video={video} />
        ))}
      </div>
    </div>
  );
}
