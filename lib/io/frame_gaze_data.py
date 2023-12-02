from ..utils.math import vector_to_string, vector_from_string


class GazeFrameData:
    def __init__(self, frame_number, timestamp, pitch, yaw, vector, intersection, hit) -> None:
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.pitch = pitch
        self.yaw = yaw
        self.vector = vector
        self.intersection = intersection
        self.hit = hit

    @staticmethod
    def from_string(string):
        parts = string.split("\n")
        frame_number = int(parts[0])
        timestamp = int(parts[1])

        if len(parts) == 2:
            return GazeFrameData(timestamp=timestamp, frame_number=frame_number, pitch=None, yaw=None, vector=None,
                                 intersection=None, hit=False)

        pitch = float(parts[2])
        yaw = float(parts[3])
        vector = vector_from_string(parts[4])
        intersection = vector_from_string(parts[5])
        hit = parts[6] == "true"

        return GazeFrameData(frame_number=frame_number, timestamp=timestamp, pitch=pitch, yaw=yaw, vector=vector,
                             intersection=intersection, hit=hit)

    #TODO brauche ich das? def __str__(self):
    #    return f"{self.index},{vector_to_string(self.landmark_3d)},{str(self.landmark)}"

    def export(self):
        content = [str(self.frame_number), str(self.timestamp)]

        if self.pitch is None:
            return "\n".join(content)

        content += [str(self.pitch), str(self.yaw), vector_to_string(self.vector), vector_to_string(self.intersection),
                    "true" if self.hit else "false"]
        return "\n".join(content)
