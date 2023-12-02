import logging
import pathlib
from abc import ABC

import torch

SCRIPT_PATH = pathlib.Path(__file__).resolve().parent
MODELS_PATH = SCRIPT_PATH / "../.." / "models"


class Model(ABC):
    """
    Class that represents a model. This class can be used as a gaze model or a face model
    """

    def __init__(self, name, download_url, file_name):
        # download pretrained weights
        self.download_url = download_url

        # name of the model
        self.name = name

        # file name of the file with the pretrained weights
        self.file_name = file_name

        # directory of the model with the pretrained weights
        self.model_directory = MODELS_PATH / name

    def get_model_path(self):
        return self.model_directory / self.file_name

    def download(self):
        # download the pretrained weights
        if self.exists():
            logging.info(f"No need to download '{self.name}'-model.")
            return

        logging.info(f"Downloading {self.file_name}")
        self.model_directory.mkdir(parents=True, exist_ok=True)

        torch.hub.download_url_to_file(self.download_url, self.get_model_path())

    def exists(self):
        # check hte file that has to be downloaded already exists.
        if self.download_url is None:
            return True
        return self.get_model_path().exists()
