import os
import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt

PATH = "data/big.dem"


def read_data(path):
    data_raw = open(PATH, "r").read().split("\n")
    _, _, _ = data_raw[0].split()
    data, map2d = data_raw[1:], []

    for row in data:
        vec = list(map(float, row.split()))
        if len(vec) == 0:
            continue
        map2d.append(vec)

    return np.array(map2d)


def surface_intensity(terrain, azimuth=165, elevation=45):
    """@numpy"""
    az = azimuth * np.pi / 180.0  # kierunek -> radiany
    alt = elevation * np.pi / 180.0  # kat padania -> radiany

    dx, dy = np.gradient(terrain)

    slope = 0.5 * np.pi - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(dy, dx)

    intensity = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(
        slope) * np.cos(-az - aspect - 0.5 * np.pi)

    return (intensity - intensity.min()) / (intensity.max() - intensity.min())


def surface_unit_normals(terrain):
    """@numpy"""
    dr, dc = np.gradient(terrain)

    vr = np.dstack((dr, np.ones_like(dr), np.zeros_like(dr)))
    vc = np.dstack((dc, np.zeros_like(dc), np.ones_like(dc)))

    surface_normals = np.cross(vr, vc)

    normal_magnitudes = np.linalg.norm(surface_normals, axis=2)
    return surface_normals / np.expand_dims(normal_magnitudes, axis=2)


@njit
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


@njit
def gradient_hsv_unknown(v, a=1, b=1):
    return hsv2rgb(150 - v, a, b)


@jit(nopython=True)
def surface_shadow_mapping(map_data, map_shadow, map_light, map_antishadow):
    img = np.zeros((map_data.shape[0], map_data.shape[1], 3))

    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):  # hsv
            img[i, j] = gradient_hsv_unknown(
                map_data[i, j],
                a=min(
                    1.1 - np.log1p(map_light[i, j, 0]) / 2 -
                    map_antishadow[i, j]**2 / 3,
                    1,
                ),
                b=min(0.5 + np.log1p(map_shadow[i, j]), 1),
            )

    return img


def save_to_file(fig, name="map_final"):
    fig.savefig(f"{name}.pdf")
    os.system(f"""convert \
       -verbose           \
       -density 175       \
       -trim              \
        {name}.pdf        \
       -quality 100       \
       -flatten           \
       -sharpen 0x1.0     \
        {name}.jpg""")


def plot_color_gradients(data, height=None):
    fig, ax = plt.subplots(nrows=1,
                           sharex=True,
                           figsize=(data.shape[0] / 100, data.shape[1] / 100))

    map_shadow = np.zeros((data.shape[0], data.shape[1], 1))
    map_light = np.zeros((data.shape[0], data.shape[1], 1))

    import scipy.ndimage

    sigma = [0.75, 0.75]
    data = scipy.ndimage.filters.gaussian_filter(data, sigma, mode="constant")

    map_shadow = surface_intensity(data, azimuth=270, elevation=70)
    map_antishadow = surface_intensity(data, azimuth=60, elevation=80)

    map_light = surface_unit_normals(data)

    img = surface_shadow_mapping(data, map_shadow, map_light, map_antishadow)

    im = ax.imshow(img, aspect="auto")
    im.set_extent([0, 1, 0, 1])
    ax.yaxis.set_visible(True)
    plt.yticks(np.arange(0, 1 + 0.2, 0.2), [0, 100, 200, 300, 400, ""][::-1])
    plt.xticks(np.arange(0, 1, 0.2), np.arange(0, data.shape[1], 100))
    ax.tick_params(direction="in", top=True, right=True)

    save_to_file(fig)
    plt.show()


if __name__ == "__main__":
    plot_color_gradients(read_data(PATH))
