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

    def __init__(self, A, k, Q, L, c_bar, q_bar, n_nodes, n_elements, order_of_approx):
        self.A = A
        self.k = k
        self.Q = Q
        self.L = L
        self.c_bar = c_bar
        self.q_bar = q_bar
        self.n_nodes = n_nodes
        self.n_elements = n_elements
        self.order_of_approx = order_of_approx

    def solve_diffusion(self):
        self.h_e = self.L/self.n_elements
        self.K = self.get_kgl2()
        # self.K2 = self.get_kgl2()
        # self.F = self.get_load()
        # self.apply_EBC() # Apply EBC tp K and F
        # self.d = np.linalg.solve(self.K,self.F)
        # self.v = self.d
        # self.x = np.linspace(0, self.L, self.n_nodes)
        # self.g = np.zeros(self.n_nodes)
        # self.g[-1] = self.c_bar
        # self.c = self.v + self.g


    # def get_kgl(self):
        
    #     A = self.A*self.k/2/self.h_e
    #     B = -self.A*self.k/2/self.h_e

    #     Ke = np.array([[A, B], 
    #                    [B, A]])
        
    #     iconn = np.arange(1, self.n_nodes+1)
    #     iconn = np.lib.stride_tricks.sliding_window_view(iconn, 2)
        
    #     K = np.zeros((self.n_nodes, self.n_nodes)) # preallocate the stiffness matrix
    #     for e in range(self.n_elements):
    #         for i in range(2): # indices i and j local, ii and jj global
    #             for j in range(2):
    #                 ii = iconn[e, i]
    #                 jj = iconn[e, j]
    #                 K[ii-1, jj-1] = K[ii-1, jj-1] + Ke[i, j]
    #     # print("K:", K)
    #     return K
    
    def get_kgl2(self):

        N = self.get_parent_functions()
        # print("Parent element functions:", N)

        # All N_i functions are found for our problem, lets solve for reduced stiffness matrix...
        kelm = [[0 for j in range(self.order_of_approx+1)] for i in range(self.order_of_approx+1)] # Create 8x8 python list. Don't use numpy array, it is storing sympy funcs
        dxds = self.h_e/2

        for i in range(self.order_of_approx+1):
            for j in range(self.order_of_approx+1):
                
                # Calculate each element of the reduced element stiffness matrix:
                Ni_prime = np.polyder(N[i])
                Nj_prime = np.polyder(N[j])

                f = self.A*self.k*Ni_prime*Nj_prime*dxds
                f_int = np.polyint(f)
                kelm[i][j] = f_int(1) - f_int(-1)
        
        # Reduced element stiffness matrix is found, now find whole element matrix
        iconn = self.get_iconn()
        print(iconn)
        K = np.zeros((self.n_nodes, self.n_nodes))
        for e in range(self.n_elements):
            for i in range(self.n_nodes): # indices i and j local, ii and jj global
                for j in range(self.n_nodes):
                    print(i)

    def get_parent_functions(self):
        """
        Find all parent functions for our approximation. Using lagrange
        polynomials.
        """

        # Evenly spaced points on the s number line.
        s = np.linspace(-1, 1, self.order_of_approx+1)
        y = np.eye(self.order_of_approx+1)
        N = [] # Init list

        for i in range(self.order_of_approx+1):
            N.append(scipy.interpolate.lagrange(s, y[i, :]))
        return N
    
    def get_iconn(self):
        matrix = []
        start = 1
        for i in range(4):
            row = []
            for j in range(3):
                row.append(start+j*2)
            matrix.append(row)
            if i % 2 == 0:
                start = row[-1] + 2
            else:
                start = row[-1] + 3
        return matrix

    def get_load(self):
        m1 = np.zeros(self.n_nodes)
        m1[0] = 1
        f1 = self.A*self.q_bar*m1
        m2 = np.ones(self.n_nodes)
        m2[0] = .5
        m2[-1] = .5
        f2 = self.Q*m2
        m3 = np.zeros(self.n_nodes)
        m3[-1] = self.c_bar/4*(self.L*self.h_e+.5*self.L*self.h_e**2)
        f3 = -self.A*self.k*m3
        F = f1+f2+f3
        # print("F:", F)
        return F

    def apply_EBC(self):
        # print("HI")
        self.K[-1, :] = 0
        self.K[:, -1] = 0
        self.K[-1, -1] = 1
        self.F[-1] = 0
        # print("K:", self.K)
        # print("F:", self.F)

        # This will force d(I)=0, then v=sum d_i*psi_i. I is the node that 
        # the EBC is located at





k= 0.1 # Diffusion coefficient [m^2 s^-1]
q = 0.1 # Diffusion flux [kg m^-2 s^-1]
L = 2 # Length of rod
c1 = 5
x = np.linspace(0,L,1000)
c = -k*q*x+c1 # Concentration [kg m^-2]
plt.figure()
plt.plot(x, c)
plt.grid(True)


# A, k, Q, L, c_bar, q_bar, n_nodes, n_elements
Approx1 = ApproxODE(.5, .1, 0, 2, 5, .1, 6, 2, 2)
Approx1.solve_diffusion()
# Approx2 = ApproxODE(.5, .1, 0, 2, 5, .1, 4, 3, 1)
# Approx2.solve_diffusion()
# Approx3 = ApproxODE(.5, .1, 0, 2, 5, .1, 6, 5, 1)
# Approx3.solve_diffusion()
# Approx4 = ApproxODE(.5, .1, 0, 2, 5, .1, 100, 99, 1)
# Approx4.solve_diffusion()
# Approx5 = ApproxODE(.5, .1, 1, 2, 5, .1, 100, 99, 1)
# Approx5.solve_diffusion()

# approx_lists = [Approx1, Approx2, Approx3, Approx4, Approx5]

# for ap in approx_lists:
#     plt.figure()
#     plt.plot(ap.x, ap.c, 'bo')
#     plt.grid(True)

print("Runtime:", time.time()-start_timer)
# plt.show()

