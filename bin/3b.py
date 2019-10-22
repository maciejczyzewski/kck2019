import matplotlib
from collections.abc import Iterable
"""
convert           \
   -verbose       \
   -density 150   \
   -trim          \
    map.pdf       \
   -quality 100   \
   -flatten       \
   -sharpen 0x1.0 \
    map.jpg
"""

# matplotlib.use("Agg")  # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import tqdm
import numpy as np
import math

PATH = "data/big.dem"


def read_data(path):
    data_raw = open(PATH, "r").read().split("\n")
    _, _, height = data_raw[0].split()
    data, map2d = data_raw[1:], []

    for row in data:
        vec = list(map(float, row.split()))
        if len(vec) == 0:
            continue
        map2d.append(vec)

    return float(height), np.array(map2d)


def plot_color_gradients(data, gradient, height=None):
    fig, ax = plt.subplots(nrows=1,
                           sharex=True,
                           figsize=(data.shape[0] / 100, data.shape[1] / 100))

    light_const = 130 / 3  # light = (0, 0)
    height_const = 2 * height / 100
    light_pos = (-100, -100)

    wut = np.zeros((data.shape[0], data.shape[1], 1))
    wut2 = np.zeros((data.shape[0], data.shape[1], 1))
    wut3 = np.zeros((data.shape[0], data.shape[1], 1))

    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            try:
                wx = data[i, j - 1] - data[i, j + 1]
                wy = data[i + 1, j] - data[i - 1, j]
                wz = height_const

                if wx == wy and wx == 0:
                    wy = 1

                angle = (wx * light_pos[0] + wy * light_pos[1]) / (math.hypot(
                    wx, wy) * math.hypot(light_pos[0], light_pos[1]))
                a = math.acos(angle)

                wut3[i, j] = a
                wut2[i, j] = a / 6
                wut[i, j] = a / 3
            except:
                pass

    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            try:
                a = math.atan2(1, wut3[i, j]) / math.pi
                b = 1 - abs(a)
                t3 = wut[i - 1, j - 1] + wut[i, j - 1] + wut[i - 1, j]
                d = t3 / 3 - b
                if d > 0:
                    bn = 0.1 * b  # XXX: jak spectogram
                else:
                    bn = (1 + t3) / 4
                    wut[i - 1, j - 1] = bn
                    wut[i - 1, j] = bn
                    wut[i, j - 1] = bn
                    wut2[i, j] = 2 * d**2
                wut[i, j] = bn
            except:
                pass

    for _i in tqdm(range(data.shape[0])):
        for _j in range(data.shape[1]):
            i, j = data.shape[0] - _i, data.shape[1] - _j
            try:
                a = math.atan2(1, wut3[i, j]) / math.pi
                b = 1 - abs(a)
                t3 = wut[i + 1, j + 1] + wut[i, j + 1] + wut[i + 1, j]
                d = t3 / 3 - b
                if d > 0:
                    bn = b - d**2
                else:
                    bn = (1 + t3) / 4
                    wut[i + 1, j + 1] = bn
                    wut[i + 1, j] = bn
                    wut[i, j + 1] = bn
                    wut2[i, j] = 2 * d**2
                wut[i, j] = bn
            except:
                pass

    img = np.zeros((data.shape[0], data.shape[1], 3))
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            img[i, j] = gradient(data[i, j],
                                 a=1 - wut2[i, j] * 1.3,
                                 b=wut[i, j])

    im = ax.imshow(img, aspect="auto")
    im.set_extent([0, 1, 0, 1])
    ax.yaxis.set_visible(True)
    plt.yticks(np.arange(0, 1 + 0.2, 0.2), [0, 100, 200, 300, 400, ""][::-1])
    plt.xticks(np.arange(0, 1, 0.2), np.arange(0, data.shape[1], 100))
    ax.tick_params(direction="in", top=True, right=True)

    fig.savefig("map.pdf")
    plt.show()


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


def gradient_hsv_unknown(v, a=1, b=1):
    return hsv2rgb(150 - v, a, b)


if __name__ == "__main__":
    height, data = read_data(PATH)
    plot_color_gradients(data, gradient_hsv_unknown, height=height)
