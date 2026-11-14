import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { z } from "zod";
import {
  FrameDetailSchema,
  FrameSummarySchema,
  GazeTrackSchema,
  ModelsSchema,
  VideoSchema,
  VideoSummarySchema,
  type FrameSummary,
} from "./schemas";

/** Fetch JSON from the API and validate it against the zod schema. */
async function request<T>(schema: z.ZodType<T>, url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    throw new Error(`${init?.method ?? "GET"} ${url} failed: ${response.status}`);
  }
  return schema.parse(await response.json());
}

export const imageUrl = (frameId: number) => `/api/frames/${frameId}/image`;

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: () => request(ModelsSchema, "/api/models") });
}

export function useVideos() {
  return useQuery({
    queryKey: ["videos"],
    queryFn: () => request(VideoSummarySchema.array(), "/api/videos"),
  });
}

export function useVideo(videoId: number) {
  return useQuery({
    queryKey: ["video", videoId],
    queryFn: () => request(VideoSchema, `/api/videos/${videoId}`),
  });
}

export function useFrames(videoId: number) {
  return useQuery({
    queryKey: ["frames", videoId],
    queryFn: () => request(FrameSummarySchema.array(), `/api/videos/${videoId}/frames`),
  });
}

export function useFrame(frameId: number | undefined) {
  return useQuery({
    queryKey: ["frame", frameId],
    queryFn: () => request(FrameDetailSchema, `/api/frames/${frameId}`),
    enabled: frameId !== undefined,
    placeholderData: (previous) => previous,
  });
}

export function useGazeTrack(videoId: number, face: string | undefined, gaze: string | undefined) {
  return useQuery({
    queryKey: ["track", videoId, face, gaze],
    queryFn: () =>
      request(
        GazeTrackSchema,
        `/api/videos/${videoId}/gazes?face=${encodeURIComponent(face!)}&gaze=${encodeURIComponent(gaze!)}`,
      ),
    enabled: Boolean(face && gaze),
  });
}

type FrameLabels = Partial<Pick<FrameSummary, "label_looking" | "label_blink">>;

/** Update frame labels with an optimistic update of the frame list. */
export function useUpdateFrameLabels(videoId: number) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ frameId, labels }: { frameId: number; labels: FrameLabels }) =>
      fetch(`/api/frames/${frameId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(labels),
      }).then((r) => {
        if (!r.ok) throw new Error(`Saving labels failed: ${r.status}`);
      }),
    onMutate: ({ frameId, labels }) => {
      client.setQueryData<FrameSummary[]>(["frames", videoId], (frames) =>
        frames?.map((f) => (f.id === frameId ? { ...f, ...labels } : f)),
      );
    },
    onSettled: (_data, _error, { frameId }) => {
      void client.invalidateQueries({ queryKey: ["frame", frameId] });
      void client.invalidateQueries({ queryKey: ["videos"] });
    },
  });
}

export function useUpdateFaceQuality(videoId: number) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({
      faceId,
      quality,
    }: {
      faceId: number;
      frameId: number;
      detector: string;
      quality: number | null;
    }) =>
      fetch(`/api/faces/${faceId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label_quality: quality }),
      }).then((r) => {
        if (!r.ok) throw new Error(`Saving quality failed: ${r.status}`);
      }),
    onMutate: ({ frameId, detector, quality }) => {
      client.setQueryData<FrameSummary[]>(["frames", videoId], (frames) =>
        frames?.map((f) =>
          f.id === frameId
            ? { ...f, label_quality: { ...f.label_quality, [detector]: quality } }
            : f,
        ),
      );
    },
    onSettled: (_data, _error, { frameId }) => {
      void client.invalidateQueries({ queryKey: ["frame", frameId] });
      void client.invalidateQueries({ queryKey: ["videos"] });
    },
  });
}
