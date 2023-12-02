import logging
import pathlib

import matplotlib.pyplot as plt
import numpy

import lib.evaluation.evaluate_face
import lib.evaluation.evaluate_gaze
from lib.evaluation.visualization import draw_bar_chart, Scale, draw_heat_map, draw_scatter_plot

"""
Hey :D

If you are a student that continues my work let me say: I am sorry. This evaluation script + the other evaluation scripts in 
./lib/evaluation were written in the last few days of my thesis. So I sadly had no time to refactor that code.
"""

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(filename)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SCRIPT_PATH = pathlib.Path(__file__).resolve().parent
OUTPUT_DIRECTORY = SCRIPT_PATH / "output"

VISUALIZATION_MAPPING = [
    "Gut", "Ungenau", "Schlecht", "Nicht erkannt"
]


def get_row(row):
    total = row["fn"] + row["fp"]
    good = row["count_good"]
    bad = row["count_bad"]
    ungenau = row["count_ungenau"]
    none = row["count_none"]

    return f"{total} & {good} & {ungenau} & {bad} & {none}"


def print_fn_fp_table(results_fn_fp):
    text = "\\begin{table}[H]\n"
    text += "\\centering\n"
    text += "\\begin{tabular}{l|l|c|c c c c}\n"
    text += "\\toprule\n"
    text += "\\textbf{Face} & \\textbf{Gaze} & $FN+FP$ & Gut& Ungenau & Schlecht & Keins \\\\ \n"
    text += "\\midrule\n"
    eth = get_row(results_fn_fp["dlib"]["eth-gaze"])
    mpii = get_row(results_fn_fp["dlib"]["mpiifacegaze"])
    gaze360 = get_row(results_fn_fp["dlib"]["l2cs-gaze-360"])
    head = get_row(results_fn_fp["dlib"]["head-rotation"])

    text += "\\multirow{4}*{dlib}"
    text += f"& ETH-XGaze & {eth} \\\\ \n"
    text += f"& Gaze360 & {gaze360} \\\\ \n"
    text += f"& MPIIFaceGaze & {mpii} \\\\ \n"
    text += f"& Kopfrotation & {head} \\\\ \n"
    text += "\hline"
    eth = get_row(results_fn_fp["mediapipe"]["eth-gaze"])
    mpii = get_row(results_fn_fp["mediapipe"]["mpiifacegaze"])
    gaze360 = get_row(results_fn_fp["mediapipe"]["l2cs-gaze-360"])
    head = get_row(results_fn_fp["mediapipe"]["head-rotation"])

    text += "\\multirow{4}*{MediaPipe}"
    text += f"& ETH-XGaze & {eth} \\\\ \n"
    text += f"& Gaze360 & {gaze360} \\\\ \n"
    text += f"& MPIIFaceGaze & {mpii} \\\\ \n"
    text += f"& Kopfrotation & {head} \\\\ \n"
    text += "\\bottomrule"
    text += "\\end{tabular}\n"
    text += f"\\caption{{TODO}}\n"
    text += f"\\label{{tab:evaluation:fn:fp}}\n"
    text += "\\end{table}"

    return text


THRESHOLD_SCALE = Scale(0, 0.5, 50)
THRESHOLD_UP_SCALE = Scale(-0.1, 0.5, 50)

use_blinks = False

BEST_CONFIG = {
    "!ignore": {
        "dlib": {
            "eth-gaze": {
                "size": 0.224,
                "position": 0.047
            },
            "l2cs-gaze-360": {
                "size": 0.480,
                "position": 0.047
            },
            "mpiifacegaze": {
                "size": 0.092,
                "position": 0.01
            },
            "head-rotation": {
                "size": 0.286,
                "position": 0.218
            }
        },
        "mediapipe": {
            "eth-gaze": {
                "size": 0.255,
                "position": 0.022
            },
            "l2cs-gaze-360": {
                "size": 0.429,
                "position": 0.047
            },
            "mpiifacegaze": {
                "size": 0.061,
                "position": -0.002
            },
            "head-rotation": {
                "size": 0.296,
                "position": 0.206
            }
        }
    },
    "ignore": {
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

}


def main():
    directories = [file for file in OUTPUT_DIRECTORY.iterdir()]
    video_paths = [item for item in directories if item.is_dir()]

    missing = {
        "dlib": {
            "missed": 0,
            "not_detected": 0
        },
        "mediapipe": {
            "missed": 0,
            "not_detected": 0
        }
    }

    results_gaze = {
        "threshold": {}
    }

    results_fp_fn_hits = {
        "dlib": {
            "eth-gaze": {
                "fn": 0,
                "fp": 0,
                "count_bad": 0,
                "count_ungenau": 0,
                "count_good": 0,
                "count_none": 0,
            },
            "mpiifacegaze": {
                "fn": 0,
                "fp": 0,
                "count_bad": 0,
                "count_ungenau": 0,
                "count_good": 0,
                "count_none": 0,
            },
            "head-rotation": {
                "fn": 0,
                "fp": 0,
                "count_bad": 0,
                "count_ungenau": 0,
                "count_good": 0,
                "count_none": 0,
            },
            "l2cs-gaze-360": {
                "fn": 0,
                "fp": 0,
                "count_bad": 0,
                "count_ungenau": 0,
                "count_good": 0,
                "count_none": 0,
            }
        },
        "mediapipe": {
            "eth-gaze": {
                "fn": 0,
                "fp": 0,
                "count_bad": 0,
                "count_ungenau": 0,
                "count_good": 0,
                "count_none": 0,
            },
            "mpiifacegaze": {
                "fn": 0,
                "fp": 0,
                "count_bad": 0,
                "count_ungenau": 0,
                "count_good": 0,
                "count_none": 0,
            },
            "head-rotation": {
                "fn": 0,
                "fp": 0,
                "count_bad": 0,
                "count_ungenau": 0,
                "count_good": 0,
                "count_none": 0,
            },
            "l2cs-gaze-360": {
                "fn": 0,
                "fp": 0,
                "count_bad": 0,
                "count_ungenau": 0,
                "count_good": 0,
                "count_none": 0,
            }
        }
    }

    results_quality = {
        "dlib": {
            "look": [],
            "!look": []
        },
        "mediapipe": {
            "look": [],
            "!look": []
        }
    }

    results_blinks = {
        "dlib": {
            "scores": [],
            "thresholds": [],
            "count": 0
        },
        "mediapipe": {
            "scores": [],
            "thresholds": [],
            "count": 0
        }
    }

    results_videos = {
        "dlib": {
            "face_detected": [],
            "lambda": [],
            "quality": []
        },
        "mediapipe": {
            "face_detected": [],
            "lambda": [],
            "quality": []
        }
    }

    distances_map = {
        "x": [],
        "y": [],
        "color": [],
        "model": []
    }

    total_count_labels = {
        "dlib": [0, 0, 0, 0],
        "mediapipe": [0, 0, 0, 0]
    }

    f1_score_average = {
        "dlib": {
            "eth-gaze": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0,
            },
            "mpiifacegaze": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            },
            "head-rotation": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            },
            "l2cs-gaze-360": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            }
        },
        "mediapipe": {
            "eth-gaze": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0,
            },
            "mpiifacegaze": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            },
            "head-rotation": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            },
            "l2cs-gaze-360": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            }
        }
    }

    f1_score_average_ignore = {
        "ignore": [6, 7, 10, 22, 23, 24],
        "dlib": {
            "eth-gaze": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0,
            },
            "mpiifacegaze": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            },
            "head-rotation": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            },
            "l2cs-gaze-360": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            }
        },
        "mediapipe": {
            "eth-gaze": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0,
            },
            "mpiifacegaze": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            },
            "head-rotation": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            },
            "l2cs-gaze-360": {
                "values": numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps)),
                "count": 0
            }
        }
    }

    X = numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps))
    Y = numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps))
    Z = numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps))
    Z2 = numpy.zeros((THRESHOLD_UP_SCALE.steps, THRESHOLD_SCALE.steps))

    position_average = {
        "dlib": {
            "x": [],
            "y": [],
        },
        "mediapipe": {
            "x": [],
            "y": [],
        }
    }

    for video in video_paths:
        lib.evaluation.evaluate_face.evaluate_face(video, distances_map, total_count_labels, missing, results_blinks,
                                                   results_quality, position_average, results_videos)
        lib.evaluation.evaluate_gaze.evaluate_gaze(video, results_gaze, f1_score_average, f1_score_average_ignore,
                                                   use_blinks, BEST_CONFIG, results_fp_fn_hits)

    print(print_fn_fp_table(results_fp_fn_hits))
    # draw_scatter_plot(position_average["dlib"]["x"], position_average["dlib"]["y"], None,
    #                  f'Position | Total', 'x', 'y', f"./src/plots/position-dlib_model.png",
    #                  show_average=False, ylim=(0, 1920), xlim=(0, 1080))
    # draw_scatter_plot(position_average["mediapipe"]["x"], position_average["dlib"]["y"], None,
    #                  f'Position | Total', 'x', 'y', f"./src/plots/position-mp_model.png",
    #                  show_average=False, ylim=(0, 1920), xlim=(0, 1080))

    for face, value in results_videos.items():
        avg_detected = sum(value["face_detected"]) / len(value["face_detected"])
        avg_lambda = sum(value["lambda"]) / len(value["lambda"])
        quality = sum(value["quality"]) / len(value["quality"])
        print(f"{face} : {avg_detected} {avg_lambda} {quality}")

    print(results_blinks)
    for face, value in results_blinks.items():
        average = sum(value["thresholds"]) / len(value["thresholds"])
        average_f1 = sum(value["scores"]) / len(value["scores"])
        print(f"{face}: {average} | {average_f1}")
        filter_videos = [6, 7, 10, 22, 23, 24]
        filtered_average = 0
        filtered_average_f1 = 0
        count = 0
        for i in range(len(value["thresholds"])):
            if i in filter_videos:
                continue
            count += 1
            filtered_average += value["thresholds"][i]
            filtered_average_f1 += value["scores"][i]
        print(f"{face} filtert: {filtered_average / count} | {filtered_average_f1 / count}")

    for face, value in results_quality.items():
        average_score_quality_looks = sum(value["look"]) / len(value["look"])
        average_score_quality_not_looks = sum(value["!look"]) / len(value["!look"])
        print(f"{face} : {average_score_quality_looks / 3} | {average_score_quality_not_looks / 3}")

    m_dlib = missing["dlib"]["missed"]
    m_mp = missing["mediapipe"]["missed"]

    nd_dlib = missing["dlib"]["not_detected"]
    nd_mp = missing["mediapipe"]["not_detected"]

    print(f"{m_dlib} / {nd_dlib}")
    print(f"{m_mp} / {nd_mp}")

    max_f1 = 0
    max_x = 0
    max_y = 0

    for i, x_value in enumerate(
            numpy.linspace(THRESHOLD_SCALE.min_value, THRESHOLD_SCALE.max_value, THRESHOLD_SCALE.steps)):
        for j, y_value in enumerate(numpy.linspace(THRESHOLD_UP_SCALE.min_value, THRESHOLD_UP_SCALE.max_value,
                                                   THRESHOLD_UP_SCALE.steps)):
            X[j, i] = x_value
            Y[j, i] = y_value

    for face_key, face_value in f1_score_average.items():
        for gaze_key, gaze_value in face_value.items():
            for i, x_value in enumerate(
                    numpy.linspace(THRESHOLD_SCALE.min_value, THRESHOLD_SCALE.max_value, THRESHOLD_SCALE.steps)):
                for j, y_value in enumerate(numpy.linspace(THRESHOLD_UP_SCALE.min_value, THRESHOLD_UP_SCALE.max_value,
                                                           THRESHOLD_UP_SCALE.steps)):

                    score = gaze_value["values"][j, i] / gaze_value["count"]
                    Z[j, i] = score
                    if score > max_f1:
                        max_x = x_value
                        max_y = y_value
                        max_f1 = score

            output = f"./src/plots/{face_key}-{gaze_key}-total-contour.jpg" if not use_blinks else f"./src/plots/{face_key}-{gaze_key}-total-contour-without-blink.jpg"
            draw_heat_map(X, Y, Z, max_x, max_y, max_f1, f"Parameter Zone | {face_key} | {gaze_key}", "Seitenlänge l",
                          "Verschiebung p",
                          output)

            max_f1 = 0
            max_x = 0
            max_y = 0

    for face_key, face_value in f1_score_average_ignore.items():
        if face_key == "ignore":
            continue
        for gaze_key, gaze_value in face_value.items():
            for i, x_value in enumerate(
                    numpy.linspace(THRESHOLD_SCALE.min_value, THRESHOLD_SCALE.max_value, THRESHOLD_SCALE.steps)):
                for j, y_value in enumerate(numpy.linspace(THRESHOLD_UP_SCALE.min_value, THRESHOLD_UP_SCALE.max_value,
                                                           THRESHOLD_UP_SCALE.steps)):

                    score = gaze_value["values"][j, i] / gaze_value["count"]
                    Z2[j, i] = score
                    if score > max_f1:
                        max_x = x_value
                        max_y = y_value
                        max_f1 = score
            output = f"./src/plots/{face_key}-{gaze_key}-total-ignore-contour.jpg" if not use_blinks else f"./src/plots/{face_key}-{gaze_key}-total-ignore-contour-without-blink.jpg"

            draw_heat_map(X, Y, Z2, max_x, max_y, max_f1, f"Parameter Zone | Gefiltert | {face_key} | {gaze_key}",
                          "Seitenlänge l",
                          "Verschiebung p",
                          output)

            max_f1 = 0
            max_x = 0
            max_y = 0

    plt.scatter(distances_map["x"], distances_map["y"], color=distances_map["color"], marker='o', s=1)
    plt.xlabel('Frame')
    plt.ylabel('Distanz')

    # Titel hinzufügen
    plt.title(f'Distanz | Total | Relation zu Qualität')
    plt.savefig(f"./src/plots/distance-total.png")
    plt.close()

    draw_scatter_plot(distances_map["x"], distances_map["y"], distances_map["model"],
                      f'Distanz | Total | Relation zu Modell', 'Frame', 'Distanz', f"./src/plots/distance_model.png",
                      show_average=True, ylim=None, xlim=None)
    plt.scatter(distances_map["x"], distances_map["y"], color=distances_map["model"], marker='o', s=1)
    plt.xlabel('Frame')
    plt.ylabel('Distanz')

    # Titel hinzufügen
    plt.title(f'Distanz | Total | Relation zu Modell')
    plt.savefig(f"./src/plots/distance_model.png")
    plt.close()

    print(results_gaze)

    draw_bar_chart(VISUALIZATION_MAPPING, [total_count_labels["dlib"], total_count_labels["mediapipe"]],
                   "Verteilung der Qualität", "Anzahl", ["green", "orange", "red", "black"],
                   f"./src/plots/total-bar-quality.png")

    print(results_fp_fn_hits)


if __name__ == "__main__":
    main()
