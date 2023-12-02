import numpy

from ..estimation.face.face import Face, FaceLandmark
from ..utils.math import vector_to_string, vector_from_string, convert_rotation_vector_to_matrix


class FrameFaceDataLandmark:

    def __init__(self, index, landmark, landmark_3d) -> None:
        self.landmark = landmark
        self.landmark_3d = landmark_3d
        self.index = index

    @staticmethod
    def from_string(string):
        parts = string.split(",")
        index = int(parts[0])
        landmark_3d = numpy.array([float(parts[1]), float(parts[2]), float(parts[3])])

        landmark = FaceLandmark(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8]))
        return FrameFaceDataLandmark(index=index, landmark=landmark, landmark_3d=landmark_3d)

    def __str__(self):
        return f"{self.index},{vector_to_string(self.landmark_3d)},{str(self.landmark)}"


class FrameFaceData:

    def __init__(self, frame_number, face: Face | None, timestamp) -> None:
        self.normalized_rotation = None
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.landmarks = []

        if face is None:
            self.face_distance = None
            self.face_center = None
            self.face_head_rotation_vector = None
            self.face_position = None
            self.average_blink_score = None

        else:
            self.face_distance = face.distance
            self.face_center = face.center
            self.face_head_rotation_vector = face.head_rotation_vector
            self.face_position = face.position
            self.average_blink_score = face.average_blink_score
            amount_of_landmarks = int(face.landmarks_3d.size / 3)
            for point in range(amount_of_landmarks):
                self.landmarks.append(FrameFaceDataLandmark(
                    index=point,
                    landmark=face.landmarks[point],
                    landmark_3d=face.landmarks_3d[point]
                ))

    def get_bounding_box(self):
        image_points = []
        for landmark in self.landmarks:
            image_points.append((landmark.landmark.image_x, landmark.landmark.image_y))

        image_points = numpy.array(image_points)

        min_x = numpy.min(image_points[:, 0])
        max_x = numpy.max(image_points[:, 0])
        min_y = numpy.min(image_points[:, 1])
        max_y = numpy.max(image_points[:, 1])

        return [(min_x, min_y), (max_x, max_y)]

    @staticmethod
    def from_string(string):
        parts = string.split("\n")
        frame_number = int(parts[0])
        timestamp = int(parts[1])

        frame_data = FrameFaceData(timestamp=timestamp, frame_number=frame_number, face=None)

        if len(parts) == 2:
            return frame_data

        frame_data.face_distance = float(parts[2])
        frame_data.face_head_rotation_vector = vector_from_string(parts[3])
        frame_data.face_center = vector_from_string(parts[4])
        frame_data.face_position = vector_from_string(parts[5])
        frame_data.average_blink_score = float(parts[6])

        for landmark in range(7, len(parts)):
            frame_data.landmarks.append(FrameFaceDataLandmark.from_string(parts[landmark]))

        return frame_data

    def normalize(self):
        self.normalized_rotation = self.compute_normalized_rotation()

    # reference: https://github.com/hysts/pytorch_mpiigaze_demo/blob/47cdf68414d20c8281bbb0a03112a298761aaa9b/ptgaze/head_pose_estimation/head_pose_normalizer.py#L52
    def compute_normalized_rotation(self):

        if not self.has_detected_face():
            raise ValueError("Illegal State. Face is None.")

        z_axis = self.face_center / numpy.linalg.norm(self.face_center)

        rotation_matrix = convert_rotation_vector_to_matrix(self.face_head_rotation_vector)
        head_x_axis = rotation_matrix[:, 0]

        cross_y = numpy.cross(z_axis, head_x_axis)
        y_axis = cross_y / numpy.linalg.norm(cross_y)

        cross_x = numpy.cross(y_axis, z_axis)
        x_axis = cross_x / numpy.linalg.norm(cross_x)

        conversion_matrix = numpy.vstack([x_axis, y_axis, z_axis])
        return conversion_matrix

    def has_detected_face(self):
        return len(self.landmarks) > 0

    def export(self):
        content = [str(self.frame_number), str(self.timestamp)]

        # no landmarks = no face detected
        if len(self.landmarks) == 0:
            return "\n".join(content)

        content += [
            str(self.face_distance),
            vector_to_string(self.face_head_rotation_vector),
            vector_to_string(self.face_center),
            vector_to_string(self.face_position),
            str(self.average_blink_score)
        ]

        for landmark in self.landmarks:
            content += [str(landmark)]
        return "\n".join(content)
