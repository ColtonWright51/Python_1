"""
P1_1.py

Created by Colton Wright on 2/26/2023

First program for John Cotton's Project 1 in ME6570
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import sys
import time
np.set_printoptions(linewidth=sys.maxsize,threshold=sys.maxsize, precision=3)

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
        print("h_e:", self.h_e)
        self.K = self.get_kgl()
        self.K2 = self.get_kgl2()
        
        print("K:\n", self.K)
        print("K2:\n", self.K2)

        self.F = self.get_load()
        self.F2 = self.get_load2()
        
        print("F:\n", self.F)
        print("F2:\n", self.F2)
        # self.load_test()
        self.K2, self.F2 = self.apply_EBC(self.K2, self.F2) # Apply EBC to K and F
        print("K2:\n", self.K2); print("F2:\n", self.F2)


        self.d = np.linalg.solve(self.K2,self.F2)
        self.v = self.d
        self.x = np.linspace(0, self.L, self.n_nodes)

        print("\n\n\n")
        # self.g = np.zeros(self.n_nodes)
        # self.g[-1] = self.c_bar
        # self.c = self.v + self.g
    def get_kgl(self):

        A = self.A*self.k/2/self.h_e*self.L
        B = -self.A*self.k/2/self.h_e*self.L
        # print("K1 A:", A); print("K1 B:", B)

        Ke = np.array([[A, B], 
                       [B, A]])

        iconn = np.arange(1, self.n_nodes+1)
        iconn = np.lib.stride_tricks.sliding_window_view(iconn, 2)

        K = np.zeros((self.n_nodes, self.n_nodes)) # preallocate the stiffness matrix
        for e in range(self.n_elements):
            for i in range(2): # indices i and j local, ii and jj global
                for j in range(2):
                    ii = iconn[e, i]
                    jj = iconn[e, j]
                    K[ii-1, jj-1] = K[ii-1, jj-1] + Ke[i, j]
        # print("K:", K)
        return K

    def get_kgl2(self):

        N = self.get_parent_functions()

        # All N_i functions are found for our problem, lets solve for reduced stiffness matrix...
        # kelm = [[0 for j in range(self.order_of_approx+1)] for i in range(self.order_of_approx+1)] # Create 8x8 python list. Don't use numpy array, it is storing sympy funcs
        kelm = np.zeros((self.order_of_approx+1, self.order_of_approx+1))
        dxds = self.h_e/2 # dx/ds
        dsdx = 2/self.h_e # ds/dx

        for i in range(self.order_of_approx+1):
            for j in range(self.order_of_approx+1):
                
                # Calculate each element of the reduced element stiffness matrix:
                # print("N_" + str(i) + ":", N[i])/
                Ni_prime = np.polyder(N[i])*dsdx # dsdx must be here. the polynomial here is a function of s, need this to be a function of x.
                # print("N_" + str(i) + "':", Ni_prime)
                Nj_prime = np.polyder(N[j])*dsdx
                # print("N_" + str(j) + "':", Nj_prime)

                f = self.A*self.k*Ni_prime*Nj_prime*dxds
                f_int = np.polyint(f)
                kelm[i, j] = f_int(1) - f_int(-1)
        
        # Reduced element stiffness matrix is found, now find whole element matrix
        iconn = self.get_iconn()
        # print(iconn)
        K = np.zeros((self.n_nodes, self.n_nodes))
        for e in range(self.n_elements):
            for i in range(self.order_of_approx+1): # indices i and j local, ii and jj global\
                for j in range(self.order_of_approx+1):
                               
                    ii = iconn[e,i]
                    jj = iconn[e,j]
                    K[ii, jj] = K[ii, jj] + kelm[i, j]
                    # print(K)
        return K

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
        matrix = np.zeros((self.n_elements, self.order_of_approx+1), dtype='i')
        start = 0
        for i in range(self.n_elements):
            matrix[i, :] = np.arange(start, start+self.order_of_approx+1, dtype='i')
            start = start + self.order_of_approx
        return matrix

    def get_load(self):
        m1 = np.zeros(self.n_nodes)
        m1[0] = 1
        f1elm = self.A*self.q_bar*m1
        m2 = np.ones(self.n_nodes)
        m2[0] = .5
        m2[-1] = .5
        f2elm = self.Q*m2
        m3 = np.zeros(self.n_nodes)
        m3[-1] = self.c_bar/4*(self.L*self.h_e+.5*self.L*self.h_e**2)
        f3elm = -self.A*self.k*m3
        print("f1elm:",f1elm);print("f2elm:",f2elm);print("f3elm:",f3elm)
        F = f1elm+f2elm+f3elm
        return F

    def get_load2(self):
        F = np.zeros(self.n_nodes)
        Felm = np.zeros(self.order_of_approx+1)
        # f1elm = np.zeros(self.order_of_approx+1)
        f2elm = np.zeros(self.order_of_approx+1)
        f3elm = np.zeros(self.order_of_approx+1)
        N = self.get_parent_functions()
        dxds = self.h_e/2 # dx/ds
        dsdx = 2/self.h_e # ds/dx


        # f1elm[0] = 1
        # f1elm = self.A*self.q_bar*f1elm
        # print("N:\,",N);print("N[-1]:",N[-1])

        for i in range(self.order_of_approx+1):
            f2_toint = np.polyint(N[i]*dxds)
            f2elm[i] = self.Q*(f2_toint(1) - f2_toint(-1))

            Nn_prime = np.polyder(N[-1])*dsdx # dsdx must be here. the polynomial here is a function of s, need this to be a function of x.
            f3_toint = np.polyint(N[i]*Nn_prime*dxds)
            f3elm[i] = f3_toint(1) - f3_toint(-1)
            print("Nn_prime:",Nn_prime);print("f3_toint:",f3_toint);print("f3elm[i]",f3elm[i])
        f2elm = self.Q*f2elm
        f3elm = -self.A*self.k*self.c_bar*f3elm
        print("f2elm_2:",f2elm);print("f3elm_2:",f3elm)
        Felm = +f2elm+f3elm
        iconn = self.get_iconn()
        for e in range(self.n_elements):
            for i in range(self.order_of_approx+1): # indices i and j local, ii and jj global
                    ii = iconn[e,i]
                    F[ii] = F[ii] + Felm[i]
                    # print(F)
        f1 = np.zeros(self.n_nodes)
        f1[0] = 1
        f1 = self.A*self.q_bar*f1
        F = f1+F
        return F

    def load_test(self):
        N = self.get_parent_functions()
        for i in range(len(N)):
            f_int = np.polyint(N[i])
            f_int= f_int(1) - f_int(-1)


    def apply_EBC(self, K, F):
        K[-1, :] = 0
        K[:, -1] = 0
        K[-1, -1] = 1
        F[-1] = 0

        return K, F
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


# A, k, Q, L, c_bar, q_bar, n_nodes, n_elements, order_of_approx

a1 = ApproxODE(.5, .1, 0, 2, 5, .1, 5, 4, 1)
a1.solve_diffusion()
a2 = ApproxODE(.5, .1, 0, 2, 5, .1, 6, 5, 1)
a2.solve_diffusion()
a3 = ApproxODE(.5, .1, 0, 2, 5, .1, 7, 6, 1)
a3.solve_diffusion()
a4 = ApproxODE(.5, .1, 0, 2, 5, .1, 10, 9, 1)
a4.solve_diffusion()
a5 = ApproxODE(.5, .1, 0, 2, 5, .1, 11, 5, 2)
a5.solve_diffusion()

# approx_lists = [Approx1, Approx2, Approx3, Approx4]
approx_lists = [a1, a2, a3, a4]

for ap in approx_lists:
    plt.figure()
    plt.plot(ap.x, ap.v, 'b')
    plt.grid(True)

print("Runtime:", time.time()-start_timer)
plt.show()

