import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LinearSegmentedColormap


def get_contrast_color(hex_color):
    """
    Entscheidet, ob die Schriftfarbe für einen gegebenen Hintergrund (hex_color) weiß oder schwarz sein sollte.
    """
    # Extrahiere die RGB-Werte aus dem Hex-Farbcode
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Berechne den Helligkeitswert (Luminanz) nach der Formel für sRGB
    luminance = 0.2126 * (r / 255.0) + 0.7152 * (g / 255.0) + 0.0722 * (b / 255.0)

    # Entscheide, ob die Schriftfarbe weiß oder schwarz sein sollte
    if luminance > 0.5:
        return 'black'  # Schwarz für helle Hintergründe
    else:
        return 'white'  # Weiß für dunkle Hintergründe


def create_custom_cmap(max_color, min_color):
    """
    Erstellt eine individuelle Colormap von Rot zu Grün.
    """
    colors = [min_color, max_color]  # Rot zu Grün
    cmap_name = 'red_to_green'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


def value_to_html_color(value, norm=100.0, max_color=(0, 1, 0), min_color=(1, 0, 0)):
    """
    Wandelt einen Wert von 0 bis 100 in einen HTML-Farbcode um (z.B. "#RRGGBB").
    """
    cmap = create_custom_cmap(max_color, min_color)
    norm_value = value / norm
    rgba_color = cmap(norm_value)

    # Konvertierung der RGB-Werte in Hexadezimalformat
    hex_color = "{:02X}{:02X}{:02X}".format(
        int(rgba_color[0] * 255),
        int(rgba_color[1] * 255),
        int(rgba_color[2] * 255)
    )

    return hex_color


class Scale:

    def __init__(self, min_value, max_value, steps):
        self.min_value = min_value
        self.max_value = max_value
        self.steps = steps
        self.step_size = (self.max_value - self.min_value) / self.steps
        self.list_steps = []
        for i in range(steps):
            self.list_steps.append(self.min_value + self.step_size * i)


def draw_scatter_plot(x, y, colors, title: str, xlabel: str, ylabel: str, output: str, show_average: bool, ylim, xlim):
    plt.scatter(x, y, color=colors, marker='o', s=1)
    average = sum(y) / len(y)
    if show_average:
        plt.hlines(y=average, xmin=min(x), xmax=max(x), linewidth=1, color='r',
                   label=f"{round(average), 3}")
        text = "Durchschnitt: {:.3f}".format(average)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        plt.annotate(text, xy=(max(x), average), xytext=(0.94, 0.96), **kw)

    # Achsentitel hinzufügen
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Titel hinzufügen
    plt.title(title)
    plt.savefig(output)
    plt.close()


def draw_heat_map(X, Y, Z, max_x, max_y, max_f1, title, xlabel, ylabel, output):
    fig, ax = plt.subplots(1, 1)
    text = "x={:.3f}, y={:.3f}, {:.3f}".format(max_x, max_y, max_f1)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(max_x, max_y), xytext=(0.94, 0.96), **kw)
    color = ax.pcolormesh(X, Y, Z, cmap="coolwarm", vmin=0,
                          vmax=1)  # Verwende 'shading' für eine verbesserte Darstellung
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(color)
    plt.savefig(output)
    plt.close()


# draw_gradient_plot(gradient_labels + gradient_hits + gradient_face,


#                           f"Video {video_name} | {gaze_model_name} | {face_model_name}", "Frames",
#                           "", f"./src/plots/{video_name}_{face_model_name}_{gaze_model_name}.jpg",
#

class CombinedOverviewChartItem:

    def __init__(self, title, entries):
        self.title = title
        self.entries = entries


def draw_combined_gradient_plot(plotties, output):
    tick_labels = ["Ground Truth Blickkontakt", "Vorhersage Blickkontakt", "Qualität Gesichtserkennung"]
    ticks = [0.333 / 2, .333 / 2 + .333, .333 / 2 + .666]
    divider = [0.333, 0.666]
    x_axis = "Frames"
    y_axis = ""

    num_plots = len(plotties)

    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), dpi=300, sharex="col")

    for i in range(num_plots):
        plot = plotties[i]
        values = plot.entries
        title = plot.title

        if ticks is not None:
            axs[i].set_yticks(ticks)
        if tick_labels is not None:
            axs[i].set_yticklabels(tick_labels)

        if divider is not None:
            for divide in divider:
                axs[i].axhline(y=divide, color='black', linestyle='--', linewidth=1)

        for j in range(len(values)):
            current_value = values[j]
            axs[i].axvspan(current_value.range_x[0], current_value.range_x[1],
                           ymin=current_value.range_y[0],
                           ymax=current_value.range_y[1],
                           color=current_value.color,
                           alpha=1)
        axs[i].tick_params(axis='y', rotation=45)
        axs[i].set_xlabel(x_axis)
        axs[i].set_ylabel(y_axis)
        axs[i].set_title(title)

    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def draw_contour_map(values_1, values_2, results, output, title_x, title_y, title):
    param1_grid, param2_grid = numpy.meshgrid(values_1, values_2)

    contourf = plt.contourf(param1_grid, param2_grid, results, levels=10, cmap='RdGy')
    contour = plt.contour(param1_grid, param2_grid, results, levels=contourf.levels, colors='k')

    plt.colorbar(contourf)

    # Achsenbeschriftungen
    plt.title(title)
    plt.xlabel(title_x)
    plt.ylabel(title_y)

    plt.savefig(output)
    plt.close()


def draw_bar_chart(categories, labels, title, axis, colors, output):
    # string_counts = Counter(labels)

    # unique_strings = list(string_counts.keys())
    # counts = list(string_counts.values())

    # Breite der Balken
    bar_width = 0.35

    # Positionen der Balken auf der x-Achse
    x = numpy.arange(len(categories))

    # Erstelle das Balkendiagramm
    plt.bar(x - bar_width / 2, labels[0], bar_width, label='dlib')
    plt.bar(x + bar_width / 2, labels[1], bar_width, label='mediapipe')

    #
    plt.ylabel(axis)
    plt.title(title)
    plt.xticks(x, categories)
    plt.legend()
    plt.savefig(output)
    plt.close()


def draw_point_cloud(points):
    pass


class GradientChartPoint:

    def __init__(self, range_x, range_y, color):
        self.range_x = range_x
        self.range_y = range_y
        self.color = color


class GradientChartLabel:

    def __init__(self, height, label):
        self.height = height
        self.label = label


def draw_plot(x_axis, y_axis, x_axis_label: str, y_axis_label: str, title: str, output: str):
    """
    Draw plot with connected line.
    :param x_axis:
    :param y_axis:
    :param x_axis_label: Label of x-axis to be displayed in the plot.
    :param y_axis_label: Label of y-axis to be displayed in the plot.
    :param title: Title to be displayed in the plot.
    :param output: Output path to save the plot as an image.
    :return:
    """
    plt.plot(x_axis, y_axis)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.savefig(output)
    plt.close()


def draw_gradient_plot(values, title, x_axis, y_axis, output, ticks, tick_labels, divider):
    fig, ax = plt.subplots(figsize=(12, 4))

    if ticks is not None:
        ax.set_yticks(ticks)
    if tick_labels is not None:
        ax.set_yticklabels(tick_labels)

    if divider is not None:
        for divide in divider:
            plt.axhline(y=divide, color='black', linestyle='--', linewidth=1)

    for i in range(len(values)):
        current_value = values[i]
        ax.axvspan(current_value.range_x[0], current_value.range_x[1],
                   ymin=current_value.range_y[0],
                   ymax=current_value.range_y[1],
                   color=current_value.color,
                   alpha=1)

    plt.yticks(rotation=45)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
