from dataclasses import dataclass
import numpy


@dataclass
class Gaze:
    pitch: float
    yaw: float
    vector: numpy.array
