import logging

import face_detection
import gaze_detection
from lib.estimation.model_manager import ModelManager

model_manager = ModelManager()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    face_detection.main(model_manager)
    gaze_detection.main(model_manager)


if __name__ == "__main__":
    main()
