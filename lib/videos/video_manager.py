import logging
import pathlib

from .video import Video

SCRIPT_PATH = pathlib.Path(__file__).resolve().parent
INPUT_DIRECTORY = SCRIPT_PATH / "../.." / "input"


class VideoManager:
    """Class that manages all the available video files in ./input"""

    def __init__(self):
        video_extensions = [".mp4"]
        logging.info(f"Loading video files of '{INPUT_DIRECTORY}'")
        self.index = -1
        self.video_paths = [
            file
            for file in INPUT_DIRECTORY.iterdir()
            if file.suffix.lower() in video_extensions
        ]

        logging.info(
            f"Loaded '{len(self.video_paths)}' video{'' if len(self.video_paths) == 1 else 's'}"
        )

    def next(self):
        """Return the next video in the folder ./input"""
        self.index += 1
        if self.index >= len(self.video_paths):
            return None
        video_path = self.video_paths[self.index]
        return Video(str(video_path))

    def has_next(self):
        """Checks if there is another video available in the list"""
        return self.index + 1 < len(self.video_paths)

    def get_size(self):
        return len(self.video_paths)
