import argparse
import cv2
from gaze_estimator.gaze_detection import (
    ETHXGazeDetector,
    HeadRotationGazeDetector,
    Gaze360GazeDetector,
    MPIIFaceGazeDetector,
    GazeDetector,
)
from gaze_estimator.face_detection import (
    FaceDetector,
    DlibFaceDetector,
    MediapipeFaceDetector,
)
from gaze_estimator.common.camera import get_meta_data_of_video
from gaze_estimator.visualization import draw_landmarks, draw_gaze_vector
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


FACE_DETECTORS = {"mediapipe": MediapipeFaceDetector, "dlib": DlibFaceDetector}
GAZE_DETECTORS = {
    "eth-xgaze": ETHXGazeDetector,
    "head-rotation": HeadRotationGazeDetector,
    "gaze360": Gaze360GazeDetector,
    "mpiiface": MPIIFaceGazeDetector,
}


def demo(
    face_detector: FaceDetector,
    gaze_detector: GazeDetector,
    args: argparse.Namespace,
):
    """
    Runs a live demo for gaze estimation using the specified face detector and gaze detector.
    This function captures the video feed of the webcam, detects faces and gaze, and visualizes the results in real-time.
    It allows the user to toggle the drawing of landmarks and gaze vectors based on command line arguments.
    This function will continue to run until the user presses 'q' to exit.

    Args:
        face_detector (FaceDetector): The face detector to use.
        gaze_detector (GazeDetector): The gaze estimation model to use.
        args (argparse.Namespace): Command line arguments.
    Raises:
        RuntimeError: If the camera cannot be opened.
        ValueError: If an unsupported face detector or gaze model is specified.
    """
    camera = cv2.VideoCapture(args.cam)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    meta = get_meta_data_of_video(camera)

    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Failed to capture image from camera.")
            break

        face = face_detector.detect_face(frame, meta)
        if face is not None:
            normalized_image = gaze_detector.normalize_input_image(
                frame.copy(), face, meta
            )
            gaze = gaze_detector.detect_gaze(face, normalized_image)

            # Draw detected landmarks if enabled
            if not args.skip_landmarks:
                frame = draw_landmarks(
                    frame,
                    face,
                    face_detector.template_model,
                    args.draw_important_landmarks,
                )

            # Draw gaze vector if enabled
            if not args.skip_gaze_vector:
                frame = draw_gaze_vector(frame, face, gaze, meta)

        # Show the current frame (with landmarks and gaze vector if applicable and enabled)
        # Loops until the user presses 'q' to exit
        cv2.imshow("Gaze Estimation Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            logging.info("Exiting demo.")
            break

    camera.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Gaze Estimation Demo")
    parser.add_argument(
        "--face_detector",
        type=str,
        help="Name of the face detector to use.",
        default="mediapipe",
        choices=list(FACE_DETECTORS.keys()),
    )
    parser.add_argument(
        "--gaze_detector",
        type=str,
        help="Name of the gaze detector to use.",
        choices=list(GAZE_DETECTORS.keys()),
        default="eth-xgaze",
    )
    parser.add_argument(
        "--cam",
        type=int,
        help="Camera index to use for live demo (default is 0)",
        default=0,
    )
    parser.add_argument(
        "--draw_important_landmarks",
        action="store_true",
        help="If set, only draws important landmarks on the face.",
    )
    parser.add_argument(
        "--skip_landmarks",
        action="store_true",
        help="If set, draws landmarks on the face.",
    )
    parser.add_argument(
        "--skip_gaze_vector",
        action="store_true",
        help="If set, skips drawing the gaze vector on the frame.",
    )

    args = parser.parse_args()
    if args.face_detector not in FACE_DETECTORS:
        raise ValueError(f"Unsupported face detector: {args.face_detector}")

    if args.gaze_detector not in GAZE_DETECTORS:
        raise ValueError(f"Unsupported gaze model: {args.gaze_detector}")

    logging.info(f"Using face detector: {args.face_detector}")
    logging.info(f"Using gaze model: {args.gaze_detector}")

    face_detector = FACE_DETECTORS[args.face_detector]()
    gaze_detector = GAZE_DETECTORS[args.gaze_detector]()
    demo(face_detector, gaze_detector, args)


if __name__ == "__main__":
    main()
