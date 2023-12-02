import logging
import pathlib


class LabelSet:

    def __init__(self, path):
        self.path = path
        self.labels = {}

        if path is not None:
            with open(path, "r") as file:
                content = file.read()
                parts = content.split("\n")
                for index in range(1, len(parts)):
                    frame, label = parts[index].split(";")
                    self.labels[int(frame)] = int(label)

    def export(self, video_path):
        with open(f"{video_path}\\combined.labels", 'w') as file:
            file.write("false\n")
            for key, value in self.labels.items():
                file.write(f'{key};{value}\n')

    def get_diffs(self, label_set):

        found_diff = False

        combined = LabelSet(None)
        for frame, label in self.labels.items():
            label_2 = label_set.labels[frame]
            if label_2 == label and label_2 != 3:
                combined.labels[frame] = label
            elif label_2 != label and label_2 != 3:
                found_diff = True
                combined.labels[frame] = 3
            else:
                if label_2 == 3:
                    return
                raise Exception(f"{label}, {label_2}")

        return combined if found_diff else None

    @staticmethod
    def blink(video_path):
        return LabelSet.read(f"{video_path}/blink.labels")

    @staticmethod
    def base(video_path):
        return LabelSet.read(f"{video_path}/base.labels")

    @staticmethod
    def base_2(video_path):
        return LabelSet.read(f"{video_path}/base_2.labels")

    @staticmethod
    def face(video_path, face_model):
        return LabelSet.read(f"{video_path}/face/{face_model}/{face_model}.labels")

    @staticmethod
    def read(path):
        if not pathlib.Path(path).is_file():
            logging.warning(f"No label set for '{path}'.")
            return
        return LabelSet(path)
