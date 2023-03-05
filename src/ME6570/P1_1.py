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
import time
np.set_printoptions(linewidth=sys.maxsize,threshold=sys.maxsize, precision=16)

start_timer = time.time()
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


    def __init__(self, A, k, Q, L, c_bar, q_bar, n_nodes):
        self.A = A
        self.k = k
        self.Q = Q
        self.L = L
        self.c_bar = c_bar
        self.q_bar = q_bar
        self.n_nodes = n_nodes


    def solve_diffusion(self):
        self.h_e = self.L/self.n_nodes
        self.K = self.get_kgl()


    def get_kgl(self):
        print()

    

k= 0.1 # Diffusion coefficient [m^2 s^-1]
q = 0.1 # Diffusion flux [kg m^-2 s^-1]
L = 2 # Length of rod
c1 = 5
x = np.linspace(0,L,1000)
c = -k*q*x+c1 # Concentration [kg m^-2]
plt.figure()
plt.plot(x, c)
plt.grid(True)


Approx1 = ApproxODE(.5, .1, .1, 1, 1, 1, 2)


print("Runtime:", time.time()-start_timer)

plt.show()