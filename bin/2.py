import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import Eq, Poly, symbols, solve, diff
from sympy.polys.polyroots import roots
from pprint import pprint

IDX = 0
SAVE = True


def header(s):
    print(f"\x1b[6;30;42m{s}\x1b[0m")


def plot(vec_x, vec_y):
    global IDX
    plt.plot(vec_x, vec_y)
    IDX += 1
    print(f"plot_idx={IDX}")

    if SAVE:
        fig = plt.gcf()
        fig.savefig(
            f"wykres_sympy_{IDX}.png",
            bbox_inches="tight",
            transparent="True",
            pad_inches=0,
            dpi=400,
        )
        plt.clf()
        # plt.show()


header("[1.1/1]")

f = Poly("-x**3 + 3 * x**2 + 10 * x - 24")

vec_x = np.arange(-5, 5, 0.1)
vec_y = np.vectorize(f)(vec_x)
pprint(vec_y)

r = list(roots(f).keys())
print(r)

plt.scatter(r, len(r) * [0])
plot(vec_x, vec_y)

header("[1.1/2]")

x, y = symbols("x, y")
eq_1 = Eq(x**2 + 3 * y, 10)
eq_2 = Eq(4 * x - y**2, -2)
r = solve([eq_1, eq_2], (x, y))
print("solved")

header("[1.1/3]")

pprint(r)
print(f"len(r)={len(r)}")

header("[1.1/4]")
for sol in r:
    print(f"x, y = {[sym.evalf() for sym in sol]}")

header("[1.1/5]")

import sympy
from sympy.parsing.sympy_parser import parse_expr

f = parse_expr(
    "sin(log2(x)) * cos(x**2) / x",
    local_dict={"log2": lambda x: sympy.log(x, 2)},
)
print(diff(f, x))

header("[1.2/1]")

mat = np.array([[1, 3, 1, 2], [1, 2, 5, 8], [3, 1, 2, 9], [5, 4, 2, 1]])
print(mat)

header("[1.2/2]")

mat = np.delete(mat, (0), axis=0)  # pierwszy wiersz
mat = np.delete(mat, (-1), axis=0)  # ostatni wiersz
mat = np.delete(mat, (-1), axis=1)  # ostatni kolumna

print(mat)

header("[1.2/3]")

mat2 = np.array([[2, 3, 1], [5, 1, 3]])
print(mat2)
mat2 = np.transpose(mat2)
print(mat2)

header("[1.2/4]")

print(np.dot(mat, mat2))

header("[1.2/5]")

vec_x = np.arange(-math.pi, math.pi, math.pi)
vec_y = np.vectorize(math.sin)(vec_x)

plot(vec_x, vec_y)

vec_x = np.arange(-math.pi, math.pi, 2 * math.pi / 10)
vec_y = np.vectorize(math.sin)(vec_x)

plot(vec_x, vec_y)

vec_x = np.arange(-math.pi, math.pi, 2 * math.pi / 100)
vec_y = np.vectorize(math.sin)(vec_x)

plot(vec_x, vec_y)
