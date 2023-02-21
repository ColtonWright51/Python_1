"""
hw4_try2.py
Created by Colton Wright on 2/20/2023

Solution to John Cotton's ME6570 HW4
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import os
import sys
import hw3
import scipy

np.set_printoptions(linewidth=sys.maxsize,threshold=sys.maxsize)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", "ME6570", "HW4")
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

#------------------------------------------------------------------------------

def solve_couette(mu, dpdx, tau, h, n_nodes, n_elements):
    h_e = h/n_elements
    K = get_kgl(mu, h_e, n_nodes, n_elements)
    F = get_load(h_e, dpdx, tau, n_nodes)

    method = 3
    K, F = apply_EBC(K, F, method)

    d = np.linalg.solve(K,F)
    u = d  # the weights are equal to the estimate at the nodes
    if method == 1:
        u[-1] = u[-1] + 20  # EBC application METHOD A
    x = np.linspace(0, h, n_nodes)
    return x, u

def get_kgl(mu, h_e, n_nodes, n_elements):
    """
    Gets the stiffness matrix
    
    Parameters:
        mu (float): Elastic modulus
        h_e (float): Element length
        n_elements (int): Number of elements
    
    Returns:
        K (ndarray): Stiffness matrix
    """

    A = 7*mu/(h_e*3)
    B = -8*mu/(h_e*3)
    C = mu/(h_e*3)
    D = 16*mu/(h_e*3)
    kelm = np.array([[A, B, C],
                     [B, D, B],
                     [C, B, A]])

    iconn = np.array([[1, 2, 3],
                      [3, 4, 5],
                      [5, 6, 7]])
    
    K = np.zeros((n_nodes, n_nodes)) # preallocate the stiffness matrix
    for e in range(n_elements):
        for i in range(3): # indices i and j local, ii and jj global
            for j in range(3):
                ii = iconn[e, i]
                jj = iconn[e, j]
                K[ii-1, jj-1] = K[ii-1, jj-1] + kelm[i, j]
    return K

def get_load(h_e, dpdx, tau, n_nodes):
    """
    Gets the load vector
    
    Parameters:
        h_e (float): Element length
        dpdx (float): Pressure gradient
        tau (float): Applied shear stress
        n_nodes (int): Number of nodes
    
    Returns:
        F (ndarray): Load vector
    """

    # set up f1
    f1 = np.zeros((n_nodes))
    f1[-1] = tau
    
    # set up f2
    f2 = -dpdx*h_e*np.array([1/6, 2/3, 2/6, 2/3, 2/6, 2/3, 1/6])
    F = f1 + f2
    return F

def apply_EBC(K, F, method):
    """
    Applies the EBC by one of 3 methods presented in class
    
    Parameters:
        K (ndarray): Stiffness matrix
        F (ndarray): Load vector
        method (int): Method for applying the EBC (1, 2, or 3)
    
    Returns:
        K (ndarray): Updated stiffness matrix with EBCs applied
        F (ndarray): Updated load vector with EBCs applied
    """

    n_nodes = len(F)
    if method == 1:
        K[n_nodes-1,:] = 0 
        K[:, n_nodes-1] = 0 
        K[n_nodes-1, n_nodes-1] = 1
        F[n_nodes-1] = 0
    elif method == 2:
        K[n_nodes-1,:] = 0 
        K[:, n_nodes-1] = 0 
        K[n_nodes-1, n_nodes-1] = 1
        F[n_nodes-1] = 0
    elif method == 3:
        BIG = K[0,0]*1e9
        K[0, 0] = K[0, 0] + BIG
#         F[n_nodes-2] = 0
        F[0] = 0*BIG

    return K, F








def interpol(xp, xn, un):
    """
    Interpolates the quadratic polynomial between the node points.

    Parameters:
    xp (array): 1-D array of x-values at which to interpolate.
    xn (array): 1-D array of x-coordinates of the nodes.
    un (array): 1-D array of y-coordinates of the nodes.

    Returns:
    up (array): 1-D array of interpolated y-values at the points in xp.
    dudxp (array): 1-D array of the first derivatives of the quadratic polynomial
                   at the points in xp.
    """

    nelm = (len(xn) - 1) // 2
    iconn = hw3.getConn(nelm)  # get connectivity matrix
    up = np.zeros(len(xp))
    dudxp = np.zeros(len(xp))

    for i in range(len(xp)):
        for ie in range(nelm):
            if xp[i] <= xn[iconn[ie, 2]]:
                ln = iconn[ie, 0]
                rn = iconn[ie, 2]
                up[i], dudxp[i] = lagrange(xp[i], xn[ln:rn+1], un[ln:rn+1])
                break

    return up, dudxp








# def main():

mu = 0.01
dpdx = -7
tau = 0.1
h = 0.02
n_nodes = 7
n_elements = 3
y_q7, u_q7 = solve_couette(mu, dpdx, tau, h, n_nodes, n_elements)
y_L7, u_L7 = hw3.solve_couette(mu, dpdx, tau, h, 7)

# Symbolic solution
y_s = np.linspace(0, h, 100)
u_s = (2*tau*y_s + dpdx*y_s**2 - 2*dpdx*h*y_s)/(2*mu)

# Interpolate the functions u & u_L7
y_q7_interp = np.linspace(0,y_q7[2],1000)
y_q7_interp2 = np.linspace(y_q7[2],y_q7[4],1000)
y_q7_interp3 = np.linspace(y_q7[4],y_q7[6],1000)

lagr1 = [1, 1, 1]
for i in range(3):
    for j in range(3):
            if i != j: # If i=j, we just skip this loop

                lagr1[i] = lagr1[i] * (y_q7_interp-y_q7[j])/ (y_q7[i]-y_q7[j])
u_q7_interp = 0
for i in range(3):
    u_q7_interp = u_q7_interp + u_q7[i]*lagr1[i]
lagr2 = scipy.interpolate.lagrange([y_q7[0], y_q7[1], y_q7[2]], [u_q7[0], u_q7[1], u_q7[2]])
lagr3 = scipy.interpolate.lagrange([y_q7[2], y_q7[3], y_q7[4]], [u_q7[2], u_q7[3], u_q7[4]])
print(lagr2)
plt.figure()
plt.plot(u_q7,y_q7, 'o', label='u_q7')
plt.plot(lagr2(y_q7_interp), y_q7_interp, 'g-')
plt.plot(lagr3(y_q7_interp2), y_q7_interp2, 'g-')
plt.legend()
plt.grid(True)
save_fig("solution_uq7")


plt.figure()
plt.plot(u_q7,y_q7, 'o', label='u_q7')
plt.plot(u_L7, y_L7, '-*', label='u_L7')
plt.plot(u_s, y_s, label='Symbolic')
plt.legend()
plt.grid(True)
save_fig("solution_uq_uL_us")

# Part 1 is done. Now for residuals:
y_s7 = np.linspace(0, h, 7)
u_s7 = (2*tau*y_s7 + dpdx*y_s7**2 - 2*dpdx*h*y_s7)/(2*mu)
E_q7 = u_s7-u_q7
E_L7 = u_s7-u_L7


plt.figure()
plt.plot(y_q7,E_q7, 'o', label='E_q7')
plt.legend()
plt.grid(True)
plt.title("Residual of Symbolic and quadratic solutions")
save_fig("residual_uq7")

plt.figure()
plt.plot(y_q7,E_L7, 'o', label='E_L7')
plt.legend()
plt.grid(True)
plt.title("Residual of Symbolic and linear solutions")
save_fig("residuals_uL7")

plt.figure()
plt.plot(y_q7,E_q7, 'o', label='E_q7')
plt.plot(y_q7,E_L7, 'o', label='E_L7')
plt.legend()
plt.grid(True)
plt.title("Residual of Symbolic and FEA approximations")
save_fig("residuals")


# plt.show()

# if __name__ == '__main__':
#     main()
