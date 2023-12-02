import logging

import cv2
import numpy

from lib.estimation.model_manager import ModelManager
from lib.io.export import Exporter
from lib.utils.time_predictor import TimePredictor
from lib.videos.video_manager import VideoManager

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

THRESHOLD = .2

THRESHOLD_BOUNDING_BOX = [(-THRESHOLD / 2, 0), (THRESHOLD / 2, THRESHOLD)]

MAPPING = {
    "dlib": {
        "eth-gaze": {
            "size": 0.204,
            "position": 0.022
        },
        "l2cs-gaze-360": {
            "size": 0.5,
            "position": 0.047
        },
        "mpiifacegaze": {
            "size": 0.082,
            "position": -0.027
        },
        "head-rotation": {
            "size": 0.265,
            "position": 0.206
        }
    },
    "mediapipe": {
        "eth-gaze": {
            "size": 0.286,
            "position": 0.01
        },
        "l2cs-gaze-360": {
            "size": 0.398,
            "position": 0.047
        },
        "mpiifacegaze": {
            "size": 0.051,
            "position": -0.027
        },
        "head-rotation": {
            "size": 0.296,
            "position": 0.206
        }
    }
}



def is_point_inside_threshold(intersection: numpy.array, threshold, position_threshold):
    x = intersection[0]
    y = intersection[1]

    min_x = -threshold / 2
    max_x = threshold / 2
    min_y = -position_threshold
    max_y = threshold - position_threshold

    # todo stimmt das????
    return min_x <= x <= max_x and min_y <= y <= max_y


def process_video(video, model_manager):
    logging.info(f"Details: {str(video)}")

    count = 0

    exporter = Exporter(video.name)
    exporter.register(model_manager)
    exporter.read_face_data()

    predictor = TimePredictor(video.total_frames)

    while True:
        success, frame = video.video_capture.read()

        if not success:
            break

        if count % 100 == 0 and count != 0:
            logging.info(predictor.predict(count))

        if count % 1000 == 0 and count != 0:
            exporter.save_gaze_data(False)

        if count % 3 != 0:
            count += 1
            continue

        timestamp_frame = int(video.video_capture.get(cv2.CAP_PROP_POS_MSEC))

        # create output directories for normalized images
        for face_model_name in exporter.face_exporter.keys():
            for gaze_model in model_manager.gaze_models:
                directory = gaze_model.get_output_directory(video, face_model_name)
                directory.mkdir(exist_ok=True)

        for face_model_name, face_exporter in exporter.face_exporter.items():
            face = face_exporter.get_by_frame_number(count)

            if face is not None:
                face.normalize()


            for gaze_model in model_manager.gaze_models:
                if face is None or not face.has_detected_face():
                    exporter.save_gaze(
                        gaze_model_name=gaze_model.name,
                        face_model_name=face_exporter.face_model_name,
                        frame_number=count,
                        timestamp=timestamp_frame,
                        pitch=None,
                        yaw=None,
                        vector=None,
                        intersection=None,
                        hit=False
                    )
                    continue

                frame_for_gaze = frame.copy()
                prediction = gaze_model.detect_gaze(image=frame_for_gaze,
                                                    face=face, video=video,
                                                    face_model_name=face_model_name,
                                                    frame_number=count)

                pitch, yaw, gaze_vector = prediction
                gaze_vector = gaze_vector

                cx, cy, cz = face.face_center
                gx, gy, gz = gaze_vector

                r = -cz / gz

                intersection = numpy.array([cx + r * gx, cy + r * gy, 0], dtype="float")

                threshold = MAPPING[face_model_name][gaze_model.name]["size"]
                position = MAPPING[face_model_name][gaze_model.name]["position"]

                hit_threshold = is_point_inside_threshold(intersection, threshold, position)

                exporter.save_gaze(
                    gaze_model_name=gaze_model.name,
                    face_model_name=face_exporter.face_model_name,
                    frame_number=count,
                    timestamp=timestamp_frame,
                    pitch=pitch,
                    yaw=yaw,
                    vector=gaze_vector,
                    intersection=intersection,
                    hit=hit_threshold
                )

        count += 1

    exporter.save_face_data(True)

    video.video_capture.release()

    exporter.save_gaze_data(True)
    logging.info("Done.")


def main(model_manager: ModelManager):
    logging.info("Started analysing gaze.")

    video_manager = VideoManager()

    video_count = 0
    while video_manager.has_next():
        video_count += 1
        video = video_manager.next()
        logging.info(f" {video_count} / {video_manager.get_size()} | {video.name} | {video.total_frames}")
        process_video(video, model_manager)


if __name__ == "__main__":
    main(ModelManager())
