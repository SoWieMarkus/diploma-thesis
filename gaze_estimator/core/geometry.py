"""Geometry helpers.

All 3D quantities are in the OpenCV camera coordinate system, in meters:
``x`` points right, ``y`` points down and ``z`` points forward (out of the camera, towards the cyclist).
A gaze vector looking at the camera therefore has a negative ``z`` component.

The smartphone records with its front camera, so its display lies in the camera plane ``z = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def vector_from_pitch_yaw(pitch: float, yaw: float) -> np.ndarray:
    """Convert pitch and yaw (radians, ETH-XGaze/MPIIGaze convention) into a unit gaze vector."""
    return -np.array(
        [np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], dtype=np.float64
    )


def pitch_yaw_from_vector(vector: np.ndarray) -> tuple[float, float]:
    """Inverse of :func:`vector_from_pitch_yaw`."""
    v = np.asarray(vector, dtype=np.float64)
    v = v / np.linalg.norm(v)
    pitch = float(np.arcsin(np.clip(-v[1], -1.0, 1.0)))
    yaw = float(np.arctan2(-v[0], -v[2]))
    return pitch, yaw


def normalize(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)


def intersect_camera_plane(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Intersect rays with the plane ``z = 0``.

    Works for a single ray (shape ``(3,)``) or many rays (shape ``(N, 3)``). Rays that are parallel to
    the plane or point away from it yield ``nan``.
    """
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -o[..., 2] / d[..., 2]
    r = np.where(np.isfinite(r) & (r > 0), r, np.nan)
    return o + r[..., None] * d


@dataclass(frozen=True)
class Display:
    """A virtual smartphone display in the camera plane.

    The display is ``width`` wide (centered on the camera in x) and ``height`` tall. The camera sits
    ``offset`` below the top edge of the display, so the display covers ``y in [-offset, height - offset]``.

    The thesis used a square display (``width == height == l``) and called the offset ``p``.
    """

    width: float
    height: float
    offset: float = 0.0

    @classmethod
    def square(cls, size: float, offset: float) -> Display:
        return cls(size, size, offset)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(min_x, max_x, min_y, max_y)``."""
        return (-self.width / 2, self.width / 2, -self.offset, self.height - self.offset)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Whether points in the camera plane (``nan`` = no intersection) lie on the display."""
        p = np.asarray(points, dtype=np.float64)
        min_x, max_x, min_y, max_y = self.bounds
        with np.errstate(invalid="ignore"):
            return (
                (min_x <= p[..., 0])
                & (p[..., 0] <= max_x)
                & (min_y <= p[..., 1])
                & (p[..., 1] <= max_y)
            )

    def hits(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        return self.contains(intersect_camera_plane(origins, directions))

    def boundary(self, samples_per_edge: int = 128) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.bounds
        t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        edges = [
            np.stack([min_x + t * (max_x - min_x), np.full_like(t, min_y)], axis=1),
            np.stack([np.full_like(t, max_x), min_y + t * (max_y - min_y)], axis=1),
            np.stack([max_x - t * (max_x - min_x), np.full_like(t, max_y)], axis=1),
            np.stack([np.full_like(t, min_x), max_y - t * (max_y - min_y)], axis=1),
        ]
        points = np.concatenate(edges)
        return np.concatenate([points, np.zeros((len(points), 1))], axis=1)

    def angular_distance(
        self, origins: np.ndarray, directions: np.ndarray, samples_per_edge: int = 128
    ) -> np.ndarray:
        """Smallest angle (radians) between the gaze ray and any point on the display.

        Zero if the ray hits the display. This is a continuous "how close was the cyclist to looking at
        the display" score, which allows threshold-free metrics such as ROC curves.
        """
        o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        d = normalize(np.atleast_2d(np.asarray(directions, dtype=np.float64)))
        boundary = self.boundary(samples_per_edge)
        result = np.empty(len(o))

        chunk = 2048
        for start in range(0, len(o), chunk):
            oc, dc = o[start : start + chunk], d[start : start + chunk]
            to_boundary = normalize(boundary[None, :, :] - oc[:, None, :])
            cos = np.einsum("nkc,nc->nk", to_boundary, dc).max(axis=1)
            result[start : start + chunk] = np.arccos(np.clip(cos, -1.0, 1.0))

        result[self.hits(o, d)] = 0.0
        result[~np.isfinite(o).all(axis=1) | ~np.isfinite(d).all(axis=1)] = np.nan
        return result if np.ndim(origins) > 1 else result[:1]
