import collections
import pandas as pd
from matplotlib import rc

rc("text", usetex=True)
import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self, path, name="?", info=[]):
        self.name = name
        self.info = info
        self.data = pd.read_csv(path)
        self.vec_gen = self.data["generation"].values
        self.vec_eff = self.data["effort"].values / 1000
        self.vec_run = (self.data.drop(["generation", "effort"],
                                       axis=1).mean(axis=1).values * 100)
        self.vec_fin = (
            self.data.tail(1).drop(["generation", "effort"], axis=1).values *
            100)


class DataArray:
    dataset = collections.OrderedDict({
        "1-Evol-RS": ["blue", "o"],
        "1-Coev-RS": ["green", "v"],
        "2-Coev-RS": ["red", "D"],
        "1-Coev": ["black", "s"],
        "2-Coev": ["magenta", "d"],
    })

    def __init__(self):
        self.vec_data = []
        for name in self.dataset.keys():
            print(f"--> name={name}")
            self.vec_data.append(
                Data(f"data/{name}.csv", name=name, info=self.dataset[name]))

    def __iter__(self):
        for data in self.vec_data:
            yield data


def plot(data_array):
    plt.rcParams["axes.linewidth"] = "1.25"

    plt.subplot(1, 2, 1)

    ax = plt.gca()

    for data in data_array:
        # ax.plot(data.vec_eff, data.vec_run, color=data.info[0], linewidth=1)
        ax.plot(
            data.vec_eff,
            data.vec_run,
            color=data.info[0],
            label=data.name,
            linewidth=0.9,
            marker=data.info[1],
            markeredgecolor="k",
            markevery=25,
            markersize=6,
        )

    plt.yticks(np.arange(60, 100 + 5, 5))
    plt.xticks(np.arange(0, 500 + 100, 100))
    plt.ylim(60, 100)
    plt.xlim(0, 500)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticklabels(np.arange(0, 200 + 40, 40))

    ax.tick_params(direction="in", top=True, right=True)
    ax.set(
        xlabel="Rozegranych gier (×1000)",
        ylabel="Odsetek wygranych gier [\%]",
        title="Pokolenie",
    )
    ax.grid(linestyle="dotted", color="grey")
    ax.legend(numpoints=2)

    plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = False
    plt.rcParams["axes.linewidth"] = "0.25"

    plt.subplot(1, 2, 2)

    ax = plt.gca()
    box = ax.boxplot(
        [data.vec_fin[0] for data in data_array],
        1,  # notched
        "b+",  # points
        labels=[data.name for data in data_array],
    )

    for i, data in enumerate(data_array):
        plt.scatter(
            i + 1,
            data.vec_run[-1],
            color="b",
            edgecolors="black",
            linewidths="1.25",
        )

    for e in box["medians"]:
        e.set_color("r")
    for e in box["boxes"]:
        e.set_color("b")
    for e in box["means"]:
        e.set_color("b")
    for e in box["whiskers"]:
        e.set_color("b")
        e.set_linestyle("dashed")

    ax.yaxis.tick_right()
    plt.yticks(np.arange(60, 100 + 5, 5))
    plt.ylim(60, 100)
    ax.grid(linestyle="dotted", color="grey")
    plt.xticks(rotation=20)

    fig = plt.gcf()
    fig.savefig(
        "wykres.png",
        bbox_inches="tight",
        transparent="True",
        pad_inches=0,
        dpi=800,
    )
    plt.show()


if __name__ == "__main__":
    data_array = DataArray()
    plot(data_array)
