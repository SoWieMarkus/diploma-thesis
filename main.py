import argparse
import os
import logging
import cv2
import numpy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

import gaze_estimator.visualization as visualization
import gaze_estimator.io as io
import gaze_estimator.gaze_detection as gaze_detection
import gaze_estimator.face_detection as face_detection
from tqdm import tqdm
from database import create_database_manager
from gaze_estimator.common.camera import get_meta_data_of_video, VideoCaptureMetaData

# Default directory where the videos to be processed are stored
INPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "input")

# Default directory where the processed output will be saved
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "output")

FACE_DETECTORS: list[face_detection.FaceDetector] = [
    face_detection.MediapipeFaceDetector(),
    face_detection.DlibFaceDetector(),
]

GAZE_DETECTORS: list[gaze_detection.GazeDetector] = [
    gaze_detection.ETHXGazeDetector(),
    gaze_detection.HeadRotationGazeDetector(),
    gaze_detection.Gaze360GazeDetector(),
    gaze_detection.MPIIFaceGazeDetector(),
]

database = create_database_manager()


def process_frame(
    frame_number: int,
    frame: numpy.ndarray,
    meta: VideoCaptureMetaData,
    output_directory: str,
    args: argparse.Namespace,
):
    """
    Process a single frame of the video, detect faces and estimate gaze.
    Args:
        frame_number (int): The index of the frame in the video.
        frame (numpy.ndarray): The image frame to process.
        meta (VideoCaptureMetaData): Metadata of the video capture.
        output_directory (str): Directory where the output will be saved.
        args (argparse.Namespace): Command line arguments.
    """

    # Store original frame as jpg
    cv2.imwrite(
        os.path.join(output_directory, f"{frame_number}.jpg"),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, args.image_quality],
    )

    for face_detector in FACE_DETECTORS:
        face = face_detector.detect_face(frame, meta)
        if face is None:
            continue

        # Draw detected landmarks to the frame and store it as jpg
        frame_with_landmarks = visualization.draw_landmarks(
            frame.copy(), face, face_detector.template_model, True
        )
        face_detector_output = io.get_face_detector_directory(
            output_directory, face_detector.name
        )
        cv2.imwrite(
            os.path.join(face_detector_output, f"{frame_number}.jpg"),
            frame_with_landmarks,
            [cv2.IMWRITE_JPEG_QUALITY, args.image_quality],
        )

        for gaze_detector in GAZE_DETECTORS:
            # Normalize the input image for the gaze detector
            # Store the
            normalized_image = gaze_detector.normalize_input_image(
                frame.copy(), face, meta
            )
            output_file = io.get_gaze_detector_directory(
                output_directory,
                face_detector.name,
                gaze_detector.name,
            )
            cv2.imwrite(
                os.path.join(output_file, f"{frame_number}.jpg"),
                normalized_image,
                [cv2.IMWRITE_JPEG_QUALITY, args.image_quality],
            )
            gaze = gaze_detector.detect_gaze(face, normalized_image)


def process_video(video_path: str, args: argparse.Namespace):
    """
    Process a single video file and estimate gaze of every frame.
    Args:
        video_path (str): Path to the video file to process.
        args (argparse.Namespace): Command line arguments.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return

    meta = get_meta_data_of_video(video)

    video_name = os.path.basename(video_path)

    logging.info(f"Processing video: {video_path}")

    output_directory = os.path.join(args.output, video_name)
    io.create_output_directories(output_directory, FACE_DETECTORS, GAZE_DETECTORS)

    db_video = database.insert_video(video_name, video_path)
    # TODO clear entries of videos

    frame_count = 0
    with tqdm(total=meta.total_frames, desc=f"Processing {video_name}") as progress:
        while True:
            success, frame = video.read()
            if not success:
                logging.info(f"Finished processing video: {video_path}")
                break
            frame_count += 1
            progress.update(1)
            if frame_count % args.frame_step != 0:
                continue
            process_frame(
                frame_number=frame_count,
                frame=frame,
                meta=meta,
                output_directory=output_directory,
                args=args,
            )


def main():
    parser = argparse.ArgumentParser(description="Gaze Estimation")
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to directory containing the videos",
        default=INPUT_DIRECTORY,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to directory where the output will be saved",
        default=OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        help="Process every Nth frame (skip frames where frame_count %% N != 0)",
        default=3,
    )
    parser.add_argument(
        "--image_quality",
        type=int,
        help="Quality of the saved images (0-100, where 100 is best quality)",
        default=30,
    )

    args = parser.parse_args()
    video_paths = io.load_videos(args.dir)
    logging.info(f"Found {len(video_paths)} videos in {args.dir}")

    for video_path in video_paths:
        process_video(video_path, args)


if __name__ == "__main__":
    main()
