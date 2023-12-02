import logging

import cv2

from lib.estimation.model_manager import ModelManager
from lib.io.export import Exporter
from lib.utils.time_predictor import TimePredictor
from lib.videos.video import Video
from lib.videos.video_manager import VideoManager

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_face_of_frame(video, frame, face_model):
    face = face_model.detect_face(frame)

    if face is None:
        return None

    # apply 3d template model to face
    face.apply(face_model, video)
    return face


def process_video(video: Video, model_manager: ModelManager):
    logging.info(f"Details: {str(video)} ")

    count = 0
    exporter = Exporter(video.name)
    predictor = TimePredictor(video.total_frames)

    exporter.register(model_manager)

    while True:
        success, frame = video.video_capture.read()

        if not success:
            break

        # log current progress
        if count % 100 == 0 and count != 0:
            logging.info(predictor.predict(count))

        # save the output every 1000 frames
        if count % 1000 == 0 and count != 0:
            exporter.save_face_data(False)
            exporter.save_meta_data(False)

        # only process frames with n % 3 == 0
        if count % 3 != 0:
            count += 1
            continue

        timestamp_frame = int(video.video_capture.get(cv2.CAP_PROP_POS_MSEC))
        exporter.save_frame(frame=frame, frame_number=count, timestamp=timestamp_frame)
        for face_model in model_manager.face_models:
            frame_for_face = frame.copy()

            # detect face with face model
            face = get_face_of_frame(
                video=video, frame=frame_for_face, face_model=face_model
            )
            if face is None:
                continue
            exporter.save_face(
                face_model_name=face_model.name,
                face=face,
                frame=frame.copy(),
                frame_number=count,
                timestamp=timestamp_frame,
            )

        count += 1
    video.video_capture.release()

    # save the output files
    exporter.save_face_data(True)
    exporter.save_meta_data(True)
    logging.info("Done.")


def main(model_manager: ModelManager):
    logging.info("Started analysing videos.")

    video_manager = VideoManager()

    video_count = 0
    frames_count = 0
    while video_manager.has_next():
        video_count += 1
        video = video_manager.next()
        frames_count += video.total_frames
        logging.info(f" {video_count} / {video_manager.get_size()} | {video.name}")

        process_video(video, model_manager)
    print(frames_count)


if __name__ == "__main__":
    main(ModelManager())
