"""
P1_1.py
Created by Colton Wright on 2/26/2023

First program for John Cotton's Project 1 in ME6570
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sympy
from sympy import symbols
import os
import sys
import hw3
np.set_printoptions(linewidth=sys.maxsize,threshold=sys.maxsize, precision=16)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", "ME6570", "P1_1")
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#------------------------------------------------------------------------------

class ApproxODE:

    def __init__(self):
            




q, k, x, L, c_bar = symbols("q k x L c_bar")
c = sympy.Function("c")(x)
dcdx = sympy.diff(c, x)

eq1 = sympy.Eq(q,  -k*sympy.diff(c, x))

eqs = [eq1, eq2]
solution = sympy.solvers.ode.dsolve(eq1)
sympy.pprint(solution)

