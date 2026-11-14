import os

import torch


def select_device() -> torch.device:
    """Pick ``GAZE_DEVICE`` if set, otherwise CUDA, then Apple MPS, then CPU."""
    if requested := os.environ.get("GAZE_DEVICE"):
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
