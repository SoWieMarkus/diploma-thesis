import logging
import os
import pathlib

import matplotlib.pyplot as plt

from .label_set import LabelSet
from .visualization import GradientChartPoint, draw_bar_chart, draw_gradient_plot, Scale, draw_scatter_plot, value_to_html_color, get_contrast_color
from ..io.frame_face_data import FrameFaceData
from ..io.frame_meta_data import FrameMetaData





def read_face_file(path):
    if not path.is_file():
        return
    frames_data = {}
    with open(path, "r") as file:
        content = file.read()
        parts = content.split("\n\n")
        for index in range(1, len(parts)):
            data = FrameFaceData.from_string(parts[index])
            frames_data[data.frame_number] = data

    return frames_data


VISUALIZATION_MAPPING = [
    "Gut", "Ungenau", "Schlecht", "Nicht erkannt"
]

THRESHOLD_BLINK_SCALE = Scale(0, 1, 100)


def read_meta_file(path):
    if not path.is_file():
        return
    frames_data = {}
    with open(path, "r") as file:
        content = file.read()
        parts = content.split("\n\n")
        for index in range(1, len(parts)):
            data = FrameMetaData.from_string(parts[index])
            frames_data[data.frame_number] = data

    return frames_data


def calculate_blink_f1_score(labels, predictions):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    for frame, value in labels.labels.items():
        guess_blink = predictions[frame]
        if guess_blink is None:
            continue
        label_blink = value == 0

        if label_blink and guess_blink:
            true_positives += 1

        if label_blink and not guess_blink:
            false_negatives += 1

        if not label_blink and not guess_blink:
            true_negatives += 1

        if not label_blink and guess_blink:
            false_positives += 1

    x = (2 * true_positives)
    y = (2 * true_positives + false_positives + false_negatives)
    if y == 0:
        return 0

    return x / y


def predict_blinks(blink_x, blink_scores, threshold):
    predictions = {}

    for i in range(len(blink_x)):
        score = blink_scores[i]
        frame = blink_x[i]
        if score is None:
            predictions[frame] = None
        elif score <= threshold:
            predictions[frame] = True
        elif score > threshold:
            predictions[frame] = False
        else:
            raise Exception("Illegal State.")
    return predictions

def get_score_from_quality(label):
    if label == 0:
        return 0
    if label == 1:
        return 2
    if label == 2:
        return 1
    raise Exception(f"Illegal Value {label}")


def evaluate_face(video_path, distances_map, total_count_labels, missing, results_blinks, results_quality, position_average, results_videos):
    #logging.info(f"Evaluating face models of {video_path}")

    face_directory_path = f"{video_path}/face"
    directories = [file for file in pathlib.Path(face_directory_path).iterdir()]
    face_paths = [item for item in directories if item.is_dir()]

    gaze_base_labels = LabelSet.base(video_path)
    blink_labels = LabelSet.blink(video_path)

    meta_file = read_meta_file(pathlib.Path(f"{video_path}/base.meta"))

    video_name = os.path.basename(video_path)

    if gaze_base_labels is None:
        logging.info(f"Skipping {video_path} because no gaze labels available.")
        return

    count = []
    plots = []

    table_row_face_quality = ""

    for face_path in face_paths:
        labels_count = [0, 0, 0, 0]

        face_model_name = os.path.basename(face_path)
        face_file = read_face_file(pathlib.Path(f"{face_path}/{face_model_name}.face"))

        face_labels = LabelSet.face(video_path, face_model_name)

        # counts in how many frames there is a detected face
        frame_has_face_count = 0
        # counts how often there was no face detected but the labels say the person looks on the screen
        missed_look_on_screen_count = 0
        count_looks_on_screen = 0
        count_looks_on_screen_and_face = 0
        count_looks_on_screen_and_not_face = 0
        total_frames = len(gaze_base_labels.labels)

        y_axis_labels = []
        gradient_labels = []

        distances_x = []
        distance_colors = []
        distances = []

        meta_x = []
        meta_color = []
        meta_brightness = []

        meta_contrast = []

        meta_sharpness = []

        blink_x = []
        blink_score = []

        quality_score_video = []

        average_pixel_x = []
        average_pixel_y = []

        for frame, value in gaze_base_labels.labels.items():
            looks_on_screen = value == 1
            has_face = False
            blink_x.append(frame)
            meta_x.append(frame)

            meta_brightness.append(meta_file[frame].brightness)
            meta_contrast.append(meta_file[frame].contrast)
            meta_sharpness.append(meta_file[frame].sharpness)

            if frame in face_file:
                has_face = face_file[frame].has_detected_face()

                blink_score.append(face_file[frame].average_blink_score)
                distances_x.append(frame)
                distances_map["x"].append(frame)
                distance = face_file[frame].face_distance
                distances.append(distance)
                distances_map["y"].append(distance)
                distances_map["model"].append("blue" if face_model_name == "dlib" else "orange")

                average_pixel_x_local = []
                average_pixel_y_local = []

                for landmark in face_file[frame].landmarks:
                    average_pixel_x_local.append(landmark.landmark.image_x)
                    average_pixel_y_local.append(landmark.landmark.image_y)

                average_pixel_x.append(sum(average_pixel_x_local) / len(average_pixel_x_local))
                average_pixel_y.append(sum(average_pixel_y_local) / len(average_pixel_y_local))

            if has_face:
                score_quality = get_score_from_quality(face_labels.labels[frame])
                results_quality[face_model_name]["look" if value == 1 else "!look"].append(score_quality)

                quality_score_video.append(score_quality)
            blink_label = blink_labels.labels[frame]

            if has_face and value == 1:
                count_looks_on_screen_and_face += 1
            elif not has_face and value == 1:
                count_looks_on_screen_and_not_face += 1



            if blink_label == 0:
                gradient_labels.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0, 0.5), "#ec5252"))
            else:
                if value == 0:
                    gradient_labels.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0, 0.5), "red"))

                if value == 1:
                    gradient_labels.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0, 0.5), "green"))
                    count_looks_on_screen += 1
                if value == 2:
                    gradient_labels.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0, 0.5), "#ec5252"))

            if has_face:
                frame_has_face_count += 1
                label = face_labels.labels[frame]
                if label == 0:
                    y_axis_labels.append(VISUALIZATION_MAPPING[1])
                    labels_count[2] += 1
                    total_count_labels[face_model_name][2] += 1
                    distance_colors.append("red")
                    meta_color.append("red")
                    distances_map["color"].append("red")
                    gradient_labels.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0.5, 1), "red"))
                elif label == 1:
                    labels_count[0] += 1
                    total_count_labels[face_model_name][0] += 1
                    distance_colors.append("green")
                    meta_color.append("green")
                    distances_map["color"].append("green")
                    gradient_labels.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0.5, 1), "green"))

                elif label == 2:
                    labels_count[1] += 1
                    total_count_labels[face_model_name][1] += 1
                    meta_color.append("yellow")
                    gradient_labels.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0.5, 1), "yellow"))
                    distance_colors.append("yellow")
                    distances_map["color"].append("yellow")

                else:
                    raise Exception("Unknown Label")

            else:
                meta_color.append("black")
                blink_score.append(None)
                labels_count[3] += 1
                total_count_labels[face_model_name][3] += 1
                # gradient_labels.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0, 1), "black"))

            if looks_on_screen and not has_face:
                missed_look_on_screen_count += 1

        average_distance = sum(distances) / len(distances)
        average_quality = sum(quality_score_video) / len(quality_score_video)

        position_average[face_model_name]["x"].append(sum(average_pixel_x) / len(average_pixel_x))
        position_average[face_model_name]["y"].append(sum(average_pixel_y) / len(average_pixel_y))

        #print(f"Qualität {face_model_name} : {average_quality} ( {average_quality / 2} )")



        x_axis = []
        y_axis = []
        for i in range(len(THRESHOLD_BLINK_SCALE.list_steps)):
            current_threshold = THRESHOLD_BLINK_SCALE.list_steps[i]
            hits = predict_blinks(blink_x, blink_score, current_threshold)
            x_axis.append(current_threshold)
            y_axis.append(calculate_blink_f1_score(blink_labels, hits))

        plots.append((x_axis, y_axis, face_model_name))

        max_value = max(y_axis)

        # Den Index des Maximums in Liste y finden
        max_index = y_axis.index(max_value)

        # Den Wert in Liste x an derselben Position lesen
        results_blinks[face_model_name]["thresholds"].append(x_axis[max_index])
        results_blinks[face_model_name]["scores"].append(max_value)
        results_blinks[face_model_name]["count"] += 1

        #print(f"Average distance: {average_distance}")

        draw_scatter_plot(distances_x, distances, distance_colors, f'Distanz | Video {video_name} | {face_model_name}',
                          "Frame", "Distanz", f"./src/plots/{video_name}_{face_model_name}_distance-average.png", True,
                          (0, 1), xlim=None)

        draw_scatter_plot(meta_x, meta_contrast, meta_color, f'Kontrast | Video {video_name} | {face_model_name}',
                          "Frame", "Kontrast", f"./src/plots/{video_name}_{face_model_name}_contrast-average.png", True,
                          None, xlim=None)
        draw_scatter_plot(meta_x, meta_brightness, meta_color,
                          f'Helligkeit | Video {video_name} | {face_model_name}',
                          "Frame", "Helligkeit", f"./src/plots/{video_name}_{face_model_name}_brightness-average.png",
                          True,
                          (0,100), xlim=None)
        draw_scatter_plot(meta_x, meta_sharpness, meta_color, f'Schärfe | Video {video_name} | {face_model_name}',
                          "Frame", "Schärfe", f"./src/plots/{video_name}_{face_model_name}_sharpness-average.png", True,
                          None, xlim=None)

        count.append(labels_count)

        draw_gradient_plot(gradient_labels, f"{face_model_name} / Vergleich Gaze Labels", "Frames", "",
                           f"./src/plots/{video_name}_{face_model_name}_labels.png",
                           tick_labels=["Gaze", "Face Quality"],
                           ticks=[0.25, 0.75], divider=[0.5])


        if face_model_name == "dlib":
            missing["dlib"]["missed"] += missed_look_on_screen_count
            missing["dlib"]["not_detected"] += (total_frames - frame_has_face_count)
        if face_model_name == "mediapipe":
            missing["mediapipe"]["missed"] += missed_look_on_screen_count
            missing["mediapipe"]["not_detected"] += (total_frames - frame_has_face_count)

        no_face_frames = total_frames-frame_has_face_count
        percentage_looks_on_screen = round(count_looks_on_screen / total_frames * 100, 2)
        percentage_erkannt = round(frame_has_face_count / total_frames * 100, 2)
        score_quality = round(average_quality / 2, 2)

        no_face_frames_percentage = round((count_looks_on_screen_and_face / (count_looks_on_screen_and_not_face + count_looks_on_screen_and_face)) * 100, 2)

        results_videos[face_model_name]["face_detected"].append(percentage_erkannt)
        results_videos[face_model_name]["lambda"].append(no_face_frames_percentage)
        results_videos[face_model_name]["quality"].append(score_quality)

        color_erkennt = value_to_html_color(percentage_erkannt)
        color_looks_on_screen = value_to_html_color(percentage_looks_on_screen, max_color= (0,0,0),min_color=(1,1,1))

        colro_no_face = value_to_html_color(no_face_frames_percentage)

        color_quality = value_to_html_color(score_quality, norm=1)
        contrast_1 = get_contrast_color(f"{color_erkennt}")
        contrast_0 = get_contrast_color(f"{color_looks_on_screen}")
        contrast_2 = get_contrast_color(f"{colro_no_face}")
        contrast_3 = get_contrast_color(f"{color_quality}")

        if face_model_name == "dlib":
            table_row_face_quality += f"& \cellcolor[HTML]{{{color_looks_on_screen}}}\\textcolor{{{contrast_0}}}{{{percentage_looks_on_screen}}} ".replace(".", ",")

        string_to_print = (f"& \cellcolor[HTML]{{{color_erkennt}}}\\textcolor{{{contrast_1}}}{{{percentage_erkannt}}} "
                           f"& \cellcolor[HTML]{{{colro_no_face}}}\\textcolor{{{contrast_2}}}{{{no_face_frames_percentage} ({missed_look_on_screen_count})}} "
                           f"& \cellcolor[HTML]{{{color_quality}}}\\textcolor{{{contrast_3}}}{{{score_quality}}}").replace(".", ",")
        table_row_face_quality += string_to_print

    # use this line to print quality face table
    print(f"{video_name} {table_row_face_quality} \\\\")

    for plot in plots:
        plt.plot(plot[0], plot[1], label=plot[2])

    plt.ylim((0, 1))
    plt.xlabel("EAR Grenzwert")

    plt.ylabel("F1-Score")
    plt.title(f"Idealer EAR Grenzwert | Video {video_name}")
    plt.legend()
    plt.savefig(f"./src/plots/{video_name}_f1-score-blink.jpg")
    plt.close()

    draw_bar_chart(VISUALIZATION_MAPPING, count, "Verteilung der Qualität", "Anzahl", ["green", "orange", "red", "black"],
                   f"./src/plots/{video_name}_bar-quality.jpg")
