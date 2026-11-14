"""Camera intrinsics.

The videos of the thesis were recorded with a smartphone mounted on the handlebar, using the front
camera. The intrinsics of that camera were never calibrated, so by default they are approximated with
a focal length equal to the image width (horizontal FOV of about 53 degrees) and the principal point in
the image center. This is the same approximation the thesis used.

If the camera is known, a profile YAML can be passed, either globally (``--profile``) or as a sidecar
file next to the video (``<video>.camera.yaml``)::

    # either explicit intrinsics in pixels (for the resolution given by width/height) ...
    width: 1920
    height: 1080
    fx: 1450.2
    fy: 1450.2
    cx: 960.0
    cy: 540.0
    distortion: [0.1, -0.2, 0.0, 0.0, 0.0]
    # ... or only a horizontal field of view in degrees
    hfov_deg: 68
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = field(default=(0.0, 0.0, 0.0, 0.0))

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    @property
    def distortion_coefficients(self) -> np.ndarray:
        return np.array(self.distortion, dtype=np.float64).reshape(-1, 1)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    @classmethod
    def default(cls, width: int, height: int) -> Camera:
        """The approximation used in the thesis: focal length = image width."""
        return cls(width, height, float(width), float(width), width / 2, height / 2)

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> Camera:
        focal = width / (2 * math.tan(math.radians(hfov_deg) / 2))
        return cls(width, height, focal, focal, width / 2, height / 2)

    @classmethod
    def from_profile(cls, profile: dict, width: int, height: int) -> Camera:
        """Create a camera for a video of size ``width`` x ``height`` from a profile dict."""
        if "hfov_deg" in profile:
            return cls.from_hfov(width, height, float(profile["hfov_deg"]))

        missing = {"fx", "fy", "cx", "cy"} - profile.keys()
        if missing:
            raise ValueError(f"Camera profile is missing the keys {sorted(missing)}")

        # Rescale if the profile was calibrated for another resolution.
        sx = width / profile.get("width", width)
        sy = height / profile.get("height", height)
        return cls(
            width,
            height,
            float(profile["fx"]) * sx,
            float(profile["fy"]) * sy,
            float(profile["cx"]) * sx,
            float(profile["cy"]) * sy,
            tuple(float(v) for v in profile.get("distortion", (0.0, 0.0, 0.0, 0.0))),
        )

    @classmethod
    def resolve(cls, width: int, height: int, profile_path: Path | None = None) -> Camera:
        if profile_path is None:
            return cls.default(width, height)
        with open(profile_path, encoding="utf-8") as file:
            return cls.from_profile(yaml.safe_load(file) or {}, width, height)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": list(self.distortion),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Camera:
        return cls(
            int(data["width"]),
            int(data["height"]),
            float(data["fx"]),
            float(data["fy"]),
            float(data["cx"]),
            float(data["cy"]),
            tuple(data.get("distortion", (0.0, 0.0, 0.0, 0.0))),
        )


def find_profile_for_video(video_path: Path, fallback: Path | None = None) -> Path | None:
    """Look for a sidecar ``<video>.camera.yaml`` next to the video, otherwise use ``fallback``."""
    sidecar = video_path.with_suffix(".camera.yaml")
    return sidecar if sidecar.is_file() else fallback
