import os
import os
import pathlib

import matplotlib.pyplot as plt
import numpy

from .evaluate_face import read_face_file
from .label_set import LabelSet
from .visualization import draw_gradient_plot, GradientChartPoint, Scale, draw_heat_map, value_to_html_color, \
    get_contrast_color, CombinedOverviewChartItem, draw_combined_gradient_plot
from ..io.frame_gaze_data import GazeFrameData
from ..io.frame_meta_data import FrameMetaData

MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.5
STEPS = 50
STEP_SIZE = (MAX_THRESHOLD - MIN_THRESHOLD) / STEPS

THRESHOLD_SCALE = Scale(0, 0.5, 50)
THRESHOLD_UP_SCALE = Scale(-0.1, 0.5, 50)


def is_point_inside_threshold(intersection: numpy.array, threshold, position_threshold):
    x = intersection[0]
    y = intersection[1]

    min_x = -threshold / 2
    max_x = threshold / 2
    min_y = -position_threshold
    max_y = threshold - position_threshold
    return min_x <= x <= max_x and min_y <= y <= max_y


def read_gaze_file(path):
    if not path.is_file():
        return
    frames_data = {}
    with open(path, "r") as file:
        content = file.read()
        parts = content.split("\n\n")
        for index in range(1, len(parts)):
            data = GazeFrameData.from_string(parts[index])
            frames_data[data.frame_number] = data

    return frames_data


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


def get_gradient_of_face_labels(gaze_labels, face_labels):
    gradient = []
    for frame, value in gaze_labels.labels.items():
        if frame in face_labels.labels:
            label = face_labels.labels[frame]
            if label == 0:
                gradient.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0.666, 1), "red"))
            elif label == 1:
                gradient.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0.666, 1), "green"))
            elif label == 2:
                gradient.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0.666, 1), "yellow"))
            else:
                raise Exception("Unknown Label")

    return gradient


def calculate_hit(gaze_data, face_data, threshold, up_threshold):
    hits = {}
    for frame, value in gaze_data.items():

        if frame not in face_data:
            hits[frame] = False
            continue
        face = face_data[frame]

        cx, cy, cz = face.face_center
        gx, gy, gz = value.vector

        r = -cz / gz
        intersection = numpy.array([cx + r * gx, cy + r * gy, 0], dtype="float")

        hit_threshold = is_point_inside_threshold(intersection, threshold, up_threshold)
        hits[frame] = hit_threshold
    return hits


def get_gradient_of_hits(hits):
    gradient = []
    for frame, hit in hits.items():
        gradient.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0.333, .666), "green" if hit else "red"))
    return gradient


def get_gradient_of_labels(labels, blink_labels):
    gradient = []
    for frame, value in labels.labels.items():
        if blink_labels.labels[frame] == 0 or value == 2:
            gradient.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0, 0.333), "#ec5252"))
        else:
            hit = value == 1
            gradient.append(GradientChartPoint((frame - 1.5, frame + 1.5), (0, 0.333), "green" if hit else "red"))
    return gradient


def calculate_mcc(labels, predictions):
    pass


def calculate_f1_score(labels, predictions, blink_labels, face_file, use_blinks=False, results_fp_fn_hits=None, face_labels=None, face_model_name=None, gaze_model_name=None):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    for frame, value in labels.labels.items():
        guess_looks_on_screen = predictions[frame]
        blink_label = blink_labels.labels[frame]
        if frame in face_file:
            distance = face_file[frame].face_distance
            if distance > 1:
                continue

        label = None
        if results_fp_fn_hits is not None and frame in face_file:
            label = face_labels.labels[frame]


        if use_blinks:
            if blink_label == 0:
                continue
        label_looks_on_screen = value == 1 and not blink_label == 0

        if label_looks_on_screen and guess_looks_on_screen:
            true_positives += 1

        if label_looks_on_screen and not guess_looks_on_screen:
            false_negatives += 1
            if results_fp_fn_hits is not None:
                results_fp_fn_hits[face_model_name][gaze_model_name]["fn"] += 1
                if label is None:
                    results_fp_fn_hits[face_model_name][gaze_model_name]["count_none"]+=1
                if label == 0:
                    results_fp_fn_hits[face_model_name][gaze_model_name]["count_bad"]+=1
                if label == 1:
                    results_fp_fn_hits[face_model_name][gaze_model_name]["count_good"]+=1
                if label == 2:
                    results_fp_fn_hits[face_model_name][gaze_model_name]["count_ungenau"]+=1



        if not label_looks_on_screen and not guess_looks_on_screen:
            true_negatives += 1

        if not label_looks_on_screen and guess_looks_on_screen:
            false_positives += 1
            if results_fp_fn_hits is not None:
                results_fp_fn_hits[face_model_name][gaze_model_name]["fp"] += 1
                if label is None:
                    results_fp_fn_hits[face_model_name][gaze_model_name]["count_none"]+=1
                if label == 0:
                    results_fp_fn_hits[face_model_name][gaze_model_name]["count_bad"]+=1
                if label == 1:
                    results_fp_fn_hits[face_model_name][gaze_model_name]["count_good"]+=1
                if label == 2:
                    results_fp_fn_hits[face_model_name][gaze_model_name]["count_ungenau"]+=1

    return (2 * true_positives) / (2 * true_positives + false_positives + false_negatives), Result(true_positives,
                                                                                                   true_negatives,
                                                                                                   false_positives,
                                                                                                   false_negatives)


def check_diffs(video_path):
    base_labels = LabelSet.base(video_path)
    base_labels_2 = LabelSet.base_2(video_path)
    if base_labels_2 is not None and base_labels is not None:
        set_combined = base_labels.get_diffs(base_labels_2)
        if set_combined is not None:
            set_combined.export(video_path)


def get_formatted_cell(value, is_max):
    html_color = value_to_html_color(value, norm=1)
    text_color = get_contrast_color(html_color)
    text = round(value, 3)
    if is_max:
        text = f"\\textbf{{{text}}}"
    return f"\cellcolor[HTML] {{{html_color}}}\\textcolor{{{text_color}}}{{{text}}}".replace(".", ",")


class Result:

    def __init__(self, tp, tn, fp, fn):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        if tp == 0 and fp == 0:
            self.precision = 0
        else:
            self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.f1 = (2 * tp) / (2 * tp + fp + fn)

        self.precision = f"{round(self.precision, 3)}".replace(".", ",")
        self.recall = f"{round(self.recall, 3)}".replace(".", ",")
        self.f1 = f"{round(self.f1, 3)}".replace(".", ",")


def get_figure_string(path, width, label, caption):
    text = "\\begin{figure}[H]\n"
    text += f"\\includegraphics[width={width}\\linewidth]{{{path}}}\n"
    text += f"\\caption{{{caption}}}\n"
    text += "\\end{figure}\n"
    return text


def get_threshold_table(results, caption, label):
    text = "\\begin{table}[H]\n"
    text += "\\centering\n"
    text += "\\begin{tabular}{l|l|c c |c}\n"
    text += "\\toprule\n"
    text += "\\textbf{Face} & \\textbf{Gaze} & \\textbf{Seitenlänge} $\\hitthresholdsize$ & \\textbf{Verschiebung} $\\hitthresholdsize$ & \\textbf{F1-Score} \\\\ \n"
    text += "\\midrule\n"

    eth = results["dlib"]["eth-gaze"]
    mpii = results["dlib"]["mpiifacegaze"]
    gaze360 = results["dlib"]["l2cs-gaze-360"]
    head = results["dlib"]["head-rotation"]

    text += "\\multirow{4}*{dlib}"
    text += f"& ETH-XGaze & {eth.size} & {eth.position} & {eth.f1}  \\\\ \n"
    text += f"& Gaze360 & {gaze360.size} & {gaze360.position} & {gaze360.f1} \\\\ \n"
    text += f"& MPIIFaceGaze & {mpii.size} & {mpii.position} & {mpii.f1}  \\\\ \n"
    text += f"& Kopfrotation & {head.size} & {head.position} & {head.f1}  \\\\ \n"
    text += "\hline"
    eth = results["mediapipe"]["eth-gaze"]
    mpii = results["mediapipe"]["mpiifacegaze"]
    gaze360 = results["mediapipe"]["l2cs-gaze-360"]
    head = results["mediapipe"]["head-rotation"]

    text += "\\multirow{4}*{MediaPipe}"
    text += f"& ETH-XGaze & {eth.size} & {eth.position} & {eth.f1}  \\\\ \n"
    text += f"& Gaze360 & {gaze360.size} & {gaze360.position} & {gaze360.f1} \\\\ \n"
    text += f"& MPIIFaceGaze & {mpii.size} & {mpii.position} & {mpii.f1}  \\\\ \n"
    text += f"& Kopfrotation & {head.size} & {head.position} & {head.f1}  \\\\ \n"

    text += "\\bottomrule"
    text += "\\end{tabular}\n"
    text += f"\\caption{{{caption}}}\n"
    text += f"\\label{{{label}}}\n"
    text += "\\end{table}"
    return text


def get_f1_score_table(results, caption, label):
    text = "\\begin{table}[H]\n"
    text += "\\centering\n"
    text += "\\begin{tabular}{l|l|c c c c|c c|c}\n"
    text += "\\toprule\n"
    text += "\\textbf{Face} & \\textbf{Gaze} & $TP$ & $TN$& $FP$ & $FN$ & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\ \n"
    text += "\\midrule\n"
    eth = results["dlib"]["eth-gaze"]
    mpii = results["dlib"]["mpiifacegaze"]
    gaze360 = results["dlib"]["l2cs-gaze-360"]
    head = results["dlib"]["head-rotation"]

    text += "\\multirow{4}*{dlib}"
    text += f"& ETH-XGaze & {eth.tp} & {eth.tn} & {eth.fp} & {eth.fn} & {eth.precision} & {eth.recall} & {eth.f1} \\\\ \n"
    text += f"& Gaze360 & {gaze360.tp} & {gaze360.tn} & {gaze360.fp} & {gaze360.fn} & {gaze360.precision} & {gaze360.recall} & {gaze360.f1} \\\\ \n"
    text += f"& MPIIFaceGaze & {mpii.tp} & {mpii.tn} & {mpii.fp} & {mpii.fn} & {mpii.precision} & {mpii.recall} & {mpii.f1} \\\\ \n"
    text += f"& Kopfrotation & {head.tp} & {head.tn} & {head.fp} & {head.fn} & {head.precision} & {head.recall} & {head.f1} \\\\ \n"
    text += "\hline"
    eth = results["mediapipe"]["eth-gaze"]
    mpii = results["mediapipe"]["mpiifacegaze"]
    gaze360 = results["mediapipe"]["l2cs-gaze-360"]
    head = results["mediapipe"]["head-rotation"]

    text += "\\multirow{4}*{MediaPipe}"
    text += f"& ETH-XGaze & {eth.tp} & {eth.tn} & {eth.fp} & {eth.fn} & {eth.precision} & {eth.recall} & {eth.f1} \\\\ \n"
    text += f"& Gaze360 & {gaze360.tp} & {gaze360.tn} & {gaze360.fp} & {gaze360.fn} & {gaze360.precision} & {gaze360.recall} & {gaze360.f1} \\\\ \n"
    text += f"& MPIIFaceGaze & {mpii.tp} & {mpii.tn} & {mpii.fp} & {mpii.fn} & {mpii.precision} & {mpii.recall} & {mpii.f1} \\\\ \n"
    text += f"& Kopfrotation & {head.tp} & {head.tn} & {head.fp} & {head.fn} & {head.precision} & {head.recall} & {head.f1} \\\\ \n"

    text += "\\bottomrule"
    text += "\\end{tabular}\n"
    text += f"\\caption{{{caption}}}\n"
    text += f"\\label{{{label}}}\n"
    text += "\\end{table}"

    return text


class BestThreshold:

    def __init__(self, f1, size, position):
        self.f1 = f"{round(f1, 3)}".replace(".", ",")
        self.size = f"{round(size, 3)}".replace(".", ",")
        self.position = f"{round(position, 3)}".replace(".", ",")








def evaluate_gaze(video_path, results_total, f1_score_average, f1_score_average_ignore, use_blinks, BEST_CONFIG, results_fp_fn_hits):
    # logging.info(f"Evaluating gaze of {video_path}.")

    results_total_video_f1_score = {
        "dlib": {
            "eth-gaze": None,
            "head-rotation": None,
            "mpiifacegaze": None,
            "l2cs-gaze-360": None
        },
        "mediapipe": {
            "eth-gaze": None,
            "head-rotation": None,
            "mpiifacegaze": None,
            "l2cs-gaze-360": None
        }
    }

    results_local_video_f1_score_with_params = {
        "dlib": {
            "eth-gaze": None,
            "head-rotation": None,
            "mpiifacegaze": None,
            "l2cs-gaze-360": None
        },
        "mediapipe": {
            "eth-gaze": None,
            "head-rotation": None,
            "mpiifacegaze": None,
            "l2cs-gaze-360": None
        }
    }

    gaze_directory = f"{video_path}/gaze"

    directories = [file for file in pathlib.Path(gaze_directory).iterdir()]
    gaze_paths = [item for item in directories if item.is_file() and item.name.endswith(".gaze")]

    video_name = os.path.basename(video_path)

    threshold_results = {}

    best_f1_score_results = {
        "dlib": {
            "head-rotation": 0,
            "l2cs-gaze-360": 0,
            "mpiifacegaze": 0,
            "eth-gaze": 0
        },
        "mediapipe": {
            "head-rotation": 0,
            "l2cs-gaze-360": 0,
            "mpiifacegaze": 0,
            "eth-gaze": 0
        }
    }

    gaze_base_labels = LabelSet.base(video_path)
    blink_labels = LabelSet.blink(video_path)
    if gaze_base_labels is None:
        return

    plots = []
    gradien_plots = {
        "dlib": [],
        "mediapipe": []
    }
    for gaze_path in gaze_paths:
        gaze_file = read_gaze_file(gaze_path)

        name = os.path.basename(gaze_path)
        name = name.replace(".gaze", "")
        parts = name.split("_")

        face_model_name = parts[0]
        gaze_model_name = parts[1]

        if face_model_name not in threshold_results:
            threshold_results[face_model_name] = {}

        face_file = read_face_file(pathlib.Path(f"{video_path}/face/{face_model_name}/{face_model_name}.face"))

        x_axis = []
        y_axis = []

        face_labels = LabelSet.face(video_path, face_model_name)

        size = BEST_CONFIG["ignore"][face_model_name][gaze_model_name]["size"]
        position = BEST_CONFIG["ignore"][face_model_name][gaze_model_name]["position"]

        hits_of_file = calculate_hit(gaze_file, face_file, size, position)
        gradient_hits = get_gradient_of_hits(hits_of_file)
        gradient_labels = get_gradient_of_labels(gaze_base_labels, blink_labels)
        gradient_face = get_gradient_of_face_labels(gaze_base_labels, face_labels)

        score, result = calculate_f1_score(gaze_base_labels, hits_of_file,
                                           blink_labels, face_file,
                                           use_blinks, face_model_name=face_model_name, gaze_model_name=gaze_model_name,results_fp_fn_hits=results_fp_fn_hits, face_labels=face_labels)


        best_f1_score_results[face_model_name][gaze_model_name] = score
        results_total_video_f1_score[face_model_name][gaze_model_name] = result

        draw_gradient_plot(gradient_labels + gradient_hits + gradient_face,
                           f"Video {video_name} | {gaze_model_name} | {face_model_name}", "Frames",
                           "", f"./plots/{video_name}_{face_model_name}_{gaze_model_name}.jpg",
                           tick_labels=["Ground Truth Blickkontakt", "Vorhersage Blickkontakt", "Qualität Gesichtserkennung"],
                           ticks=[0.333 / 2, .333 / 2 + .333, .333 / 2 + .666], divider=[0.333, 0.666])
        gradien_plots[face_model_name].append(CombinedOverviewChartItem(f"{gaze_model_name} | {face_model_name}", gradient_labels + gradient_hits + gradient_face))

        for i in range(len(THRESHOLD_SCALE.list_steps)):
            current_thres = THRESHOLD_SCALE.list_steps[i]
            hits = calculate_hit(gaze_file, face_file, current_thres, 0)
            x_axis.append(current_thres)
            score, _ = calculate_f1_score(gaze_base_labels, hits, blink_labels, face_file, use_blinks)
            y_axis.append(score)

        # Erstelle leere Arrays für X, Y und Z
        X = numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps))
        Y = numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps))
        Z = numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps))

        # Iteriere über linspace für X und Y

        max_f1 = 0
        max_x = 0
        max_y = 0

        f1_score_average[face_model_name][gaze_model_name]["count"] += 1
        if int(video_name) not in f1_score_average_ignore["ignore"]:
            f1_score_average_ignore[face_model_name][gaze_model_name]["count"] += 1

        for i, x_value in enumerate(
                numpy.linspace(THRESHOLD_SCALE.min_value, THRESHOLD_SCALE.max_value, THRESHOLD_SCALE.steps)):
            for j, y_value in enumerate(numpy.linspace(THRESHOLD_UP_SCALE.min_value, THRESHOLD_UP_SCALE.max_value,
                                                       THRESHOLD_UP_SCALE.steps)):
                X[j, i] = x_value
                Y[j, i] = y_value
                hits = calculate_hit(gaze_file, face_file, x_value, y_value)
                f1_score, _ = calculate_f1_score(gaze_base_labels, hits, blink_labels, face_file, use_blinks)
                if f1_score > max_f1:
                    max_f1 = f1_score
                    max_x = x_value
                    max_y = y_value
                Z[j, i] = f1_score

                f1_score_average[face_model_name][gaze_model_name]["values"][j, i] += f1_score
                if int(video_name) not in f1_score_average_ignore["ignore"]:
                    f1_score_average_ignore[face_model_name][gaze_model_name]["values"][j, i] += f1_score
        results_local_video_f1_score_with_params[face_model_name][gaze_model_name] = BestThreshold(max_f1, max_x, max_y)
        output = f"./plots/{video_name}-{face_model_name}-{gaze_model_name}-contour.jpg" if not use_blinks else f"./plots/{video_name}-{face_model_name}-{gaze_model_name}-contour-without-blink.jpg"
        draw_heat_map(X, Y, Z, max_x, max_y, max_f1, f"Parameter Zone | Video {video_name} | {face_model_name} | {gaze_model_name}",
                      "Seitenlänge l", "Verschiebung p", output)

        max_value = max(y_axis)

        # Den Index des Maximums in Liste y finden
        max_index = y_axis.index(max_value)

        # Den Wert in Liste x an derselben Position lesen
        threshold_results[face_model_name][gaze_model_name] = x_axis[max_index]
        plots.append((x_axis, y_axis, name))

    for plot in plots:
        plt.plot(plot[0], plot[1], label=plot[2])

    draw_combined_gradient_plot(gradien_plots["dlib"], f"./plots/{video_name}-dlib-overview-gaze.jpg")
    draw_combined_gradient_plot(gradien_plots["mediapipe"], f"./plots/{video_name}-mediapipe-overview-gaze.jpg")

    dlib_mpii = best_f1_score_results["dlib"]["mpiifacegaze"]
    dlib_gaze360 = best_f1_score_results["dlib"]["l2cs-gaze-360"]
    dlib_head = best_f1_score_results["dlib"]["head-rotation"]
    dlib_eth = best_f1_score_results["dlib"]["eth-gaze"]

    mp_mpii = best_f1_score_results["mediapipe"]["mpiifacegaze"]
    mp_gaze360 = best_f1_score_results["mediapipe"]["l2cs-gaze-360"]
    mp_head = best_f1_score_results["mediapipe"]["head-rotation"]
    mp_eth = best_f1_score_results["mediapipe"]["eth-gaze"]

    max_value = max([dlib_gaze360, dlib_mpii, dlib_head, dlib_eth, mp_mpii, mp_gaze360, mp_head, mp_eth])

    dlib_mpii = get_formatted_cell(dlib_mpii, max_value == dlib_mpii)
    dlib_gaze360 = get_formatted_cell(dlib_gaze360, max_value == dlib_gaze360)
    dlib_head = get_formatted_cell(dlib_head, max_value == dlib_head)
    dlib_eth = get_formatted_cell(dlib_eth, max_value == dlib_eth)

    mp_mpii = get_formatted_cell(mp_mpii, max_value == mp_mpii)
    mp_gaze360 = get_formatted_cell(mp_gaze360, max_value == mp_gaze360)
    mp_head = get_formatted_cell(mp_head, max_value == mp_head)
    mp_eth = get_formatted_cell(mp_eth, max_value == mp_eth)

    dlib_string = f"dlib & {dlib_eth} & {dlib_gaze360} & {dlib_mpii} & {dlib_head}"
    mediapipe_string = f"MediaPipe & {mp_eth} & {mp_gaze360} & {mp_mpii} & {mp_head}"

    total_row = f"\multirow{{2}}{{*}}{{{video_name}}} & {dlib_string} \\\\ & {mediapipe_string} \\\\ \\hline"

    print(total_row)
    results_total["threshold"][video_name] = threshold_results

    plt.ylim((0, 1))
    plt.xlabel("Threshold")

    plt.ylabel("F1-Score")
    plt.title(f"Video {video_name}")
    plt.legend()
    plt.savefig(f"./plots/{video_name}_f1-score-gaze.jpg")
    plt.close()

    appendix_figure_f1_score_blink = get_figure_string(f"appendix/{video_name}_f1-score-blink.jpg",
                                                       0.7, f"fig:appendix:{video_name}:blinks", f"Ergebnisse der Bestimmung des idealen EAR-Grenzwerts für Video {video_name}.")
    appendix_quality_bars = get_figure_string(f"appendix/{video_name}_bar-quality.jpg",
                                              0.7, f"fig:appendix:{video_name}:quality", f"Aufteilung der Gesichtserkennungsqualität von Video {video_name}.")
    appendix_overview_dlib = get_figure_string(f"appendix/{video_name}-dlib-overview-gaze.jpg", 1, f"fig:append:{video_name}:overview:dlib", "Überblick über die Ground Truth des Blickkontakts, die Vorhersage des Blickkontakts und die Qualität der Gesichtserkennung mit \\dlib.")
    appendix_overview_mp = get_figure_string(f"appendix/{video_name}-mediapipe-overview-gaze.jpg", 1,
                                               f"fig:append:{video_name}:overview:mediapipe", "Überblick über die Ground Truth des Blickkontakts, die Vorhersage des Blickkontakts und die Qualität der Gesichtserkennung mit \\mediapipe.")

    appendix_table_f1 = get_f1_score_table(results_total_video_f1_score, "Zusammensetzung des F1-Scores unter Verwendung der in Kapitel \\ref{sec:evaluation:results:gaze:threshold} bestimmten globalen Parameter $\\hitthresholdsize$ und $\\hitthresholdposition$.", f"tab:appendix:{video_name}:f1")
    appendix_table_threshold = get_threshold_table(results_local_video_f1_score_with_params, f"Ergebnis des Hyperparameter-Tunings für die Zone speziell für Video {video_name}.",
                                                   f"tab:appendix:{video_name}:local")
    appendix_text = f"\section{{Video {video_name}}}\n{appendix_figure_f1_score_blink}\n{appendix_quality_bars}\n{appendix_table_threshold}\n{appendix_table_f1}\n{appendix_overview_dlib}\n{appendix_overview_mp}"

    #use this line to print appendix tables and figures
    #print(appendix_text)

    #print(results_fp_fn_hits)
