import logging
import pathlib

from .face_models.dlib_face_model import DlibFaceModel
from .face_models.mediapipe_face_model import MediapipeFaceModel
from .gaze_models.eth_gaze_model import ETHGazeModel
from .gaze_models.gaze_360_l2cs_model import Gaze360Model
from .gaze_models.head_rotation_model import HeadRotationModel
from .gaze_models.mpiifacegaze_model import MPIIFaceGazeModel

SCRIPT_PATH = pathlib.Path(__file__).resolve().parent
MODELS_PATH = SCRIPT_PATH / "../.." / "models"


class ModelManager:
    """
    Class that manages the gaze and face models
    """

    def __init__(self) -> None:
        if not MODELS_PATH.exists():
            logging.info("Creating 'models' directory")

        self.face_model_dlib = DlibFaceModel()
        self.face_model_mediapipe = MediapipeFaceModel()

        self.gaze_model_eth = ETHGazeModel()
        self.gaze_model_mpiiface = MPIIFaceGazeModel()
        self.gaze_model_gaze360_l2cs = Gaze360Model()
        self.gaze_head_rotation = HeadRotationModel()

        self.face_models = [
            self.face_model_dlib,
            self.face_model_mediapipe
        ]
        self.gaze_models = [
            self.gaze_model_eth,
            self.gaze_model_mpiiface,
            self.gaze_model_gaze360_l2cs,
            self.gaze_head_rotation
        ]
        self.download()

    def download(self):
        # download all files needed for the models
        logging.info("Downloading models ...")
        for model in self.face_models:
            model.download()

        for model in self.gaze_models:
            model.download()
