import { z } from "zod";

export const Vec3 = z.tuple([z.number(), z.number(), z.number()]);
export type Vec3 = z.infer<typeof Vec3>;

export const CameraSchema = z.object({
  width: z.number(),
  height: z.number(),
  fx: z.number(),
  fy: z.number(),
  cx: z.number(),
  cy: z.number(),
  distortion: z.array(z.number()),
});
export type Camera = z.infer<typeof CameraSchema>;

export const VideoSchema = z.object({
  id: z.number(),
  name: z.string(),
  fps: z.number(),
  width: z.number(),
  height: z.number(),
  frame_count: z.number(),
  camera: CameraSchema,
});
export type Video = z.infer<typeof VideoSchema>;

export const VideoSummarySchema = VideoSchema.extend({
  processed_frames: z.number(),
  labeled_looking: z.number(),
  labeled_blink: z.number(),
  thumbnail_frame_id: z.number().nullable(),
  faces: z.record(
    z.string(),
    z.object({ frames: z.number(), detected: z.number(), quality_labeled: z.number() }),
  ),
});
export type VideoSummary = z.infer<typeof VideoSummarySchema>;

export const ModelsSchema = z.object({ faces: z.array(z.string()), gazes: z.array(z.string()) });
export type Models = z.infer<typeof ModelsSchema>;

export const FrameSummarySchema = z.object({
  id: z.number(),
  index: z.number(),
  timestamp_ms: z.number(),
  label_looking: z.number().nullable(),
  label_blink: z.boolean().nullable(),
  detected: z.record(z.string(), z.boolean()),
  label_quality: z.record(z.string(), z.number().nullable()),
});
export type FrameSummary = z.infer<typeof FrameSummarySchema>;

export const GazeSchema = z.object({
  id: z.number(),
  face_id: z.number(),
  model: z.string(),
  pitch: z.number(),
  yaw: z.number(),
  vector: Vec3,
  runtime_ms: z.number(),
});
export type Gaze = z.infer<typeof GazeSchema>;

export const FaceSchema = z.object({
  id: z.number(),
  detector: z.string(),
  detected: z.boolean(),
  runtime_ms: z.number(),
  bbox: z.array(z.number()).nullable(),
  landmarks_2d: z.array(z.tuple([z.number(), z.number()])).nullable(),
  landmarks_3d: z.array(Vec3).nullable(),
  rotation_vector: Vec3.nullable(),
  translation_vector: Vec3.nullable(),
  center: Vec3.nullable(),
  normalizing_rotation: z.array(Vec3).nullable(),
  distance: z.number().nullable(),
  ear_left: z.number().nullable(),
  ear_right: z.number().nullable(),
  label_quality: z.number().nullable(),
  gazes: z.array(GazeSchema),
});
export type Face = z.infer<typeof FaceSchema>;

export const FrameDetailSchema = z.object({
  id: z.number(),
  video_id: z.number(),
  index: z.number(),
  timestamp_ms: z.number(),
  brightness: z.number(),
  contrast: z.number(),
  sharpness: z.number(),
  label_looking: z.number().nullable(),
  label_blink: z.boolean().nullable(),
  faces: z.array(FaceSchema),
});
export type FrameDetail = z.infer<typeof FrameDetailSchema>;

export const GazeTrackSchema = z.array(
  z.object({ frame_id: z.number(), index: z.number(), center: Vec3, vector: Vec3 }),
);
export type GazeTrack = z.infer<typeof GazeTrackSchema>;

export const LookingLabel = { NO: 0, YES: 1, UNSURE: 2 } as const;
export const QualityLabel = { GOOD: 0, INACCURATE: 1, WRONG: 2 } as const;
export const QUALITY_NAMES = ["good", "inaccurate", "wrong"] as const;
export const LOOKING_NAMES = ["not looking", "looking", "unsure"] as const;
