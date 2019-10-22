import matplotlib

# matplotlib.use("Agg")  # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

import sys, math
from matplotlib import colors
from collections.abc import Iterable


def plot_color_gradients(gradients, names, height=None):
    rc("legend", fontsize=10)

    column_width_pt = 400  # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch
    height = 0.75 * size if height is None else height

    fig, axes = plt.subplots(nrows=len(gradients),
                             sharex=True,
                             figsize=(size, height))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)

    if not isinstance(axes, Iterable):
        axes = [axes]

    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect="auto")
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3] / 2.0
        fig.text(x_text, y_text, name, va="center", ha="left", fontsize=10)

    plt.show()
    fig.savefig("gradients.pdf")


################################################################################
################################################################################

n = lambda x: max(0, min(1, x))


# a - ile osiaga maksymalnie
# b - gdzie jest szczyt
# c - tempo/ciezkosc
def gaussian(x, a, b, c, d=0):
    b += 0.00001  # FIXME: ?
    return a * math.exp(-(x - b)**2 / (2 * c**2)) + d


def isogradient(v, pallete):
    params = isopallete(pallete)

    def find_near_k(v, params, k=4):
        sort_list = []
        for p in params:
            diff = abs(v * 255 - p[1])
            sort_list.append([diff, p])
        result = sorted(sort_list)[0:k]
        return [p[1] for p in result]

    r = sum([gaussian(v * 255, *p) for p in find_near_k(v, params[0])])
    g = sum([gaussian(v * 255, *p) for p in find_near_k(v, params[1])])
    b = sum([gaussian(v * 255, *p) for p in find_near_k(v, params[2])])
    return (n(int(r) / 255), n(int(g) / 255), n(int(b) / 255))


def isopallete(pallete):
    # FIXME: output could be cached
    vec_r, vec_g, vec_b = [], [], []

    span = len(pallete.keys())
    for key, val in pallete.items():
        dynamic_param = 255 / (span * 2)
        vec_r += [[val[0], key * 255, dynamic_param]]
        vec_g += [[val[1], key * 255, dynamic_param]]
        vec_b += [[val[2], key * 255, dynamic_param]]

    return [vec_r, vec_g, vec_b]


def test_gradient(f):
    vec_x = np.arange(0, 1, 0.005)
    vec_y1, vec_y2, vec_y3 = np.vectorize(f)(vec_x)
    plt.plot(vec_x, vec_y1, color="red")
    plt.plot(vec_x, vec_y2, color="green")
    plt.plot(vec_x, vec_y3, color="blue")

    plot_color_gradients([f], ["test"], height=0.5)

    sys.exit()


################################################################################
################################################################################


def hsv2rgb(h, s, v):
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    r, g, b = {
        0: (c, x, 0),
        1: (x, c, 0),
        2: (0, c, x),
        3: (0, x, c),
        4: (x, 0, c),
        5: (c, 0, x),
    }[int(h / 60) % 6]
    return ((r + m), (g + m), (b + m))


def gradient_rgb_bw(v):
    return (v, v, v)


def gradient_rgb_gbr(v):
    pallete = {0: [0, 255, 0], 0.5: [0, 0, 255], 1: [255, 0, 0]}
    return isogradient(v, pallete)


def gradient_rgb_gbr_full(v):
    pallete = {
        0: [0, 255, 0],
        1 * (1 / 4): [0, 255, 255],
        2 * (1 / 4): [0, 0, 255],
        3 * (1 / 4): [255, 0, 255],
        1: [255, 0, 0],
    }
    return isogradient(v, pallete)


def gradient_rgb_wb_custom(v):
    pallete = {
        0: [255, 255, 255],
        1 * (1 / 7): [255, 0, 255],
        2 * (1 / 7): [0, 0, 255],
        3 * (1 / 7): [0, 255, 255],
        4 * (1 / 7): [0, 255, 0],
        5 * (1 / 7): [255, 255, 0],
        6 * (1 / 7): [255, 0, 0],
        1: [0, 0, 0],
    }
    return isogradient(v, pallete)


def interval(start, stop, value):
    return start + (stop - start) * value


def gradient_hsv_bw(v):
    return hsv2rgb(0, 0, v)


def gradient_hsv_gbr(v):
    return hsv2rgb(interval(120, 360, v), 1, 1)


def gradient_hsv_unknown(v):
    return hsv2rgb(120 - 120 * v, 0.5, 1)


def gradient_hsv_custom(v):
    return hsv2rgb(360 * (v), n(1 - v**2), 1)


if __name__ == "__main__":

    def toname(g):
        return g.__name__.replace("gradient_", "").replace("_", "-").upper()

    # XXX: test_gradient(gradient_rgb_gbr_full)

    gradients = (
        gradient_rgb_bw,
        gradient_rgb_gbr,
        gradient_rgb_gbr_full,
        gradient_rgb_wb_custom,
        gradient_hsv_bw,
        gradient_hsv_gbr,
        gradient_hsv_unknown,
        gradient_hsv_custom,
    )

    plot_color_gradients(gradients, [toname(g) for g in gradients])
