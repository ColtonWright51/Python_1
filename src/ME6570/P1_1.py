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
        # print("h_e:", self.h_e)
        self.K = self.get_kgl() # Works for 1st degree approx
        self.K2 = self.get_kgl2() # Works for n degree approx
        
        # print("K:\n", self.K)
        # print("K2:\n", self.K2)

        self.F = self.get_load()
        self.F2 = self.get_load2(self.K2)
        
        # print("F:\n", self.F)
        # print("F2:\n", self.F2)
        # self.load_test()
        self.K, self.F = self.apply_EBC(self.K, self.F) # Apply EBC to K and F
        self.K2, self.F2 = self.apply_EBC(self.K2, self.F2) # Apply EBC to K and F
        # print("K2 after EBC:\n", self.K2); print("F2 after EBC:\n", self.F2)

        # self.d = np.linalg.solve(self.K,self.F)
        # self.v = self.d
        self.x = np.linspace(0, self.L, self.n_nodes)

        self.d2 = np.linalg.solve(self.K2,self.F2)
        self.v2 = self.d2
        self.x2 = np.linspace(0, self.L, self.n_nodes)

        # print("\n\n\n")
        self.g = np.zeros(self.n_nodes)
        self.g[-1] = self.c_bar
        # self.c = self.v + self.g
        self.c2 = self.v2 + self.g

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
        # print("f1elm:",f1elm);print("f2elm:",f2elm);print("f3elm:",f3elm)
        F = f1elm+f2elm+f3elm
        return F

    def get_load2(self, K):
        F = np.zeros(self.n_nodes)
        f1 = np.zeros(self.n_nodes)
        f2 = np.zeros(self.n_nodes)
        f2elm = np.zeros(self.order_of_approx+1)
        N = self.get_parent_functions()
        dxds = self.h_e/2 # dx/ds
        dsdx = 2/self.h_e # ds/dx

        f1[0] = 1
        f1 = self.A*self.q_bar*f1

        for i in range(self.order_of_approx+1):
            f2_toint = np.polyint(N[i]*dxds)
            f2elm[i] = self.Q*(f2_toint(1) - f2_toint(-1))

        iconn = self.get_iconn()
        for e in range(self.n_elements):
            for i in range(self.order_of_approx+1): # indices i and j local, ii and jj global
                    ii = iconn[e,i]
                    f2[ii] = f2[ii] + f2elm[i]
    

        f3 = -self.c_bar*K[:, -1] # Defined in report. Last term is just last column of stiffness matrix

        F = f1+f2+f3
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
        # This will force d(I)=0, then v=sum d_i*psi_i. I is the node that 
        # the EBC is located at


A = 0.25
k = 0.1 # Diffusion coefficient [m^2 s^-1]
Q = 0.11
L = 1 # Length of rod
c_bar = 5 # Concentration [kg m^-2]
q_bar = 0.1 # Diffusion flux [kg m^-2 s^-1]


x = np.linspace(0,L,1000)
# c = q_bar/k*L + Q*L**2/2 + c_bar - q_bar/k*x - Q/2/A/k*x**2
c = -Q/2/A/k*x**2 - q_bar/k*x
raise1 = c_bar-c[-1] # Enforce EBC
print("raise:", raise1)
c = c + raise1
plt.figure()
plt.title("Exact Solution")
plt.plot(x, c, 'b')
plt.grid(True)

# def main():

# A, k, Q, L, c_bar, q_bar, n_nodes, n_elements, order_of_approx
L1 = ApproxODE(A, k, Q, L, c_bar, q_bar, 2, 1, 1)
L1.solve_diffusion()
L2 = ApproxODE(A, k, Q, L, c_bar, q_bar, 5, 4, 1)
L2.solve_diffusion()
L3 = ApproxODE(A, k, Q, L, c_bar, q_bar, 100, 99, 1)
L3.solve_diffusion()
L4 = ApproxODE(A, k, Q, L, c_bar, q_bar, 1000, 999, 1)
L4.solve_diffusion()
linear_list = [L1, L2, L3, L4]

counter = 1
for approx in linear_list:
    interpolation = scipy.interpolate.interp1d(approx.x, approx.c2, kind="linear")
    print(interpolation)
    plt.figure()
    plt.plot(x, c, 'b', label='Exact Solution')
    plt.plot(x, interpolation(x), 'r', label='Linear interpolation')
    plt.title("Linear approximation, " + str(approx.n_elements) + " elements")
    plt.grid(True)
    plt.legend()
    save_fig("Linear"+str(counter))

    residual = interpolation(x)-c
    plt.figure()
    plt.plot(x, residual)
    plt.title("Linear approximation residual, " + str(approx.n_elements) + " elements")
    plt.grid(True)
    save_fig("Linear"+str(counter)+"_residual")

    counter = counter + 1

Q1 = ApproxODE(A, k, Q, L, c_bar, q_bar, 3, 1, 2)
Q1.solve_diffusion()
Q2 = ApproxODE(A, k, Q, L, c_bar, q_bar, 9, 4, 2)
Q2.solve_diffusion()
Q3 = ApproxODE(A, k, Q, L, c_bar, q_bar, 199, 99, 2)
Q3.solve_diffusion()
Q4 = ApproxODE(A, k, Q, L, c_bar, q_bar, 1999, 999, 2)
Q4.solve_diffusion()
quadratic_list = [Q1, Q2, Q3, Q4]

counter = 1
for approx in quadratic_list:
    interpolation = scipy.interpolate.interp1d(approx.x, approx.c2, kind="quadratic")
    print(interpolation)
    plt.figure()
    plt.plot(x, c, 'b', label='Exact Solution')
    plt.plot(x, interpolation(x), 'r', label='Quadratic interpolation')
    plt.title("Quadratic approximation, " + str(approx.n_elements) + " elements")
    plt.grid(True)
    plt.legend()
    save_fig("Quadratic"+str(counter))

    residual = interpolation(x)-c
    plt.figure()
    plt.plot(x, residual)
    plt.title("Quadratic approximation residual, " + str(approx.n_elements) + " elements")
    plt.grid(True)
    save_fig("Quadratic"+str(counter)+"_residual")

    counter = counter + 1

C1 = ApproxODE(A, k, Q, L, c_bar, q_bar, 4, 1, 3)
C1.solve_diffusion()
C2 = ApproxODE(A, k, Q, L, c_bar, q_bar, 13, 4, 3)
C2.solve_diffusion()
cubic_list = [C1, C2]

counter = 1
for approx in cubic_list:
    interpolation = scipy.interpolate.interp1d(approx.x, approx.c2, kind="cubic")
    print(interpolation)
    plt.figure()
    plt.plot(x, c, 'b', label='Exact Solution')
    plt.plot(x, interpolation(x), 'r', label='Cubic interpolation')
    plt.title("Cubic approximation, " + str(approx.n_elements) + " elements")
    plt.grid(True)
    plt.legend()
    save_fig("Cubic"+str(counter))

    residual = interpolation(x)-c
    plt.figure()
    plt.plot(x, residual)
    plt.title("Cubic approximation residual, " + str(approx.n_elements) + " elements")
    plt.grid(True)
    save_fig("Cubic"+str(counter)+"_residual")

    counter = counter + 1

# Create plots of all the lagrangian parent basis functions, orders 1-20
s1 = np.linspace(-1, 1, 1000)
for i in range(1, 21):
    approx = ApproxODE(A, k, Q, L, c_bar, q_bar, i+1, 1, i)
    Ns = approx.get_parent_functions()
    plt.figure()
    for func in Ns:
        plt.plot(s1, func(s1))
    plt.grid(True)
    plt.title("Parent element, "+str(i)+" order approximation")
    save_fig("Parent"+str(i))

approx = ApproxODE(A, k, Q, L, c_bar, q_bar, 11, 1, 10)
approx.solve_diffusion()

plt.figure()
plt.plot(approx.x, approx.c2)


print("Runtime:", time.time()-start_timer)
plt.show()

# if __name__ == '__main__':
#     main()