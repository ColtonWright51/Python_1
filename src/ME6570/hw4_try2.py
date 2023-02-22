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

np.set_printoptions(linewidth=sys.maxsize,threshold=sys.maxsize, precision=16)

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
y_s = np.linspace(0, h, 1000)
u_s = (2*tau*y_s + dpdx*y_s**2 - 2*dpdx*h*y_s)/(2*mu)
u_s_prime = mu*(-1/mu*(-dpdx*y_s+(dpdx*h-tau)))

y_s7 = np.linspace(0, h, 7)
u_s7 = (2*tau*y_s7 + dpdx*y_s7**2 - 2*dpdx*h*y_s7)/(2*mu)

# Interpolate for the linear fits:
y1 = np.linspace(y_s7[0], y_s7[1], 1000)
y2 = np.linspace(y_s7[1], y_s7[2], 1000)
y3 = np.linspace(y_s7[2], y_s7[3], 1000)
y4 = np.linspace(y_s7[3], y_s7[4], 1000)
y5 = np.linspace(y_s7[4], y_s7[5], 1000)
y6 = np.linspace(y_s7[5], y_s7[6], 1000)
linear_interp = scipy.interpolate.interp1d(y_L7, u_L7, kind='linear')
quad_interp = scipy.interpolate.interp1d(y_q7, u_q7, kind='quadratic')

# Symbolic solution to the problem between each set of nodes
u_s1 = (2*tau*y1 + dpdx*y1**2 - 2*dpdx*h*y1)/(2*mu)
u_s2 = (2*tau*y2 + dpdx*y2**2 - 2*dpdx*h*y2)/(2*mu)
u_s3 = (2*tau*y3 + dpdx*y3**2 - 2*dpdx*h*y3)/(2*mu)
u_s4 = (2*tau*y4 + dpdx*y4**2 - 2*dpdx*h*y4)/(2*mu)
u_s5 = (2*tau*y5 + dpdx*y5**2 - 2*dpdx*h*y5)/(2*mu)
u_s6 = (2*tau*y6 + dpdx*y6**2 - 2*dpdx*h*y6)/(2*mu)

plt.figure()
plt.plot(y_L7,u_L7, 'co', label='u_L7')
plt.plot(y1, linear_interp(y1), 'c-', y2, linear_interp(y2), 'c-', y3, linear_interp(y3), 'c-', y4, linear_interp(y4), 'c-', y5, linear_interp(y5), 'c-', y6, linear_interp(y6), 'c-')
plt.grid(True)
plt.title("Interpolated u_L7")
save_fig("interp1d_linear_test")

plt.figure()
plt.plot(y_q7,u_q7, 'ro', label='u_Q7')
plt.plot(y1, quad_interp(y1), 'r-', y2, quad_interp(y2), 'r-', y3, quad_interp(y3), 'r-', y4, quad_interp(y4), 'r-', y5, quad_interp(y5), 'r-', y6, quad_interp(y6), 'r-')
plt.grid(True)
plt.title("Interpolated u_q7")
save_fig("interp1d_quad_test")

plt.figure()
plt.plot(y_L7,u_L7, 'co', label='u_L7')
plt.plot(y1, linear_interp(y1), 'c-', y2, linear_interp(y2), 'c-', y3, linear_interp(y3), 'c-', y4, linear_interp(y4), 'c-', y5, linear_interp(y5), 'c-', y6, linear_interp(y6), 'c-')
plt.plot(y_q7,u_q7, 'ro', label='u_Q7')
plt.plot(y1, quad_interp(y1), 'r-', y2, quad_interp(y2), 'r-', y3, quad_interp(y3), 'r-', y4, quad_interp(y4), 'r-', y5, quad_interp(y5), 'r-', y6, quad_interp(y6), 'r-')
plt.plot(y_s, u_s, 'g-', label="Exact Solution")
plt.grid(True)
plt.legend()
plt.title("Solutions")
save_fig("solutions")

# Residual, exact-linear
EL1 = u_s1 - linear_interp(y1)
EL2 = u_s2 - linear_interp(y2)
EL3 = u_s3 - linear_interp(y3)
EL4 = u_s4 - linear_interp(y4)
EL5 = u_s5 - linear_interp(y5)
EL6 = u_s6 - linear_interp(y6)

# Residual, exact-quad
Eq1 = u_s1 - quad_interp(y1)
Eq2 = u_s2 - quad_interp(y2)
Eq3 = u_s3 - quad_interp(y3)
Eq4 = u_s4 - quad_interp(y4)
Eq5 = u_s5 - quad_interp(y5)
Eq6 = u_s6 - quad_interp(y6)

plt.figure()
plt.plot(y1, EL1, 'c-', y2, EL2, 'c-', y3, EL3, 'c-', y4, EL4, 'c-', y5, EL5, 'c-', y6, EL6, 'c-')
plt.grid(True)
plt.title("L7 Residual")
save_fig("residual_L7")

plt.figure()
plt.plot(y1, Eq1, 'r-', y2, Eq2, 'r-', y3, Eq3, 'r-', y4, Eq4, 'r-', y5, Eq5, 'r-', y6, Eq6, 'r-')
plt.grid(True)
plt.title("L7 Residual")
save_fig("residual_q7")

plt.figure()
plt.plot(y1, EL1, 'c-', y2, EL2, 'c-', y3, EL3, 'c-', y4, EL4, 'c-', y5, EL5, 'c-', y6, EL6, 'c-')
plt.plot(y1, Eq1, 'r-', y2, Eq2, 'r-', y3, Eq3, 'r-', y4, Eq4, 'r-', y5, Eq5, 'r-', y6, Eq6, 'r-')
plt.grid(True)
plt.title("FEA Approx. Residuals")
save_fig("residuals")

# Part 2 done. Now compare secondary variables:

u_L7_prime_1 = mu*np.gradient(linear_interp(y1), y1)
u_L7_prime_2 = mu*np.gradient(linear_interp(y2), y2)
u_L7_prime_3 = mu*np.gradient(linear_interp(y3), y3)
u_L7_prime_4 = mu*np.gradient(linear_interp(y4), y4)
u_L7_prime_5 = mu*np.gradient(linear_interp(y5), y5)
u_L7_prime_6 = mu*np.gradient(linear_interp(y6), y6)



plt.figure()
plt.plot(y1, u_L7_prime_1, 'c-', y2, u_L7_prime_2, 'c-', y3, u_L7_prime_3, 'c-', y4, u_L7_prime_4, 'c-', y5, u_L7_prime_5, 'c-', y6, u_L7_prime_6, 'c-')
plt.grid(True)
plt.title("Derivative of u_L7")
save_fig("interp1d_linear_prime_test")

"""
For the linear solution, determine the derivative for each element. Assign
these derivatives to the center of each element, plot those points and perform
a linear least squares fit. Then find the shear at 0 and 20 m from the
function generated by the least squares fit.
"""
u_L7_prime2_y_val = [y_s7[0], (y_s7[1]+y_s7[2])/2, (y_s7[2]+y_s7[3])/2, (y_s7[3]+y_s7[4])/2, (y_s7[4]+y_s7[5])/2, y_s7[6]]
u_L7_prime2_u_val = [u_L7_prime_1[0], u_L7_prime_2[0], u_L7_prime_3[0], u_L7_prime_4[0], u_L7_prime_5[0], u_L7_prime_6[0]]

linear_least_squares = scipy.interpolate.interp1d(u_L7_prime2_y_val, u_L7_prime2_u_val, kind='slinear')
plt.figure()
# plt.plot(y2,linear_least_squares(y2))
plt.plot(u_L7_prime2_y_val, u_L7_prime2_u_val, 'co')
plt.plot(y1, linear_least_squares(y1), 'c-', y2, linear_least_squares(y2), 'c-', y3, linear_least_squares(y3), 'c-', y4, linear_least_squares(y4), 'c-', y5, linear_least_squares(y5), 'c-', y6, linear_least_squares(y6), 'c-')
plt.plot(0.00, linear_least_squares(0), 'go', 0.002, linear_least_squares(0.002), 'go')
plt.grid(True)
plt.title("Derivative of u_L7, linear least squares")
save_fig("interp1d_linear_prime_lls_test")



"""
From your quadratic solution, use the derivative of the quadratic polynomial
for the element at that node.
"""
u_q7_prime_1 = mu*np.gradient(quad_interp(y1), y1)
u_q7_prime_2 = mu*np.gradient(quad_interp(y2), y2)
u_q7_prime_3 = mu*np.gradient(quad_interp(y3), y3)
u_q7_prime_4 = mu*np.gradient(quad_interp(y4), y4)
u_q7_prime_5 = mu*np.gradient(quad_interp(y5), y5)
u_q7_prime_6 = mu*np.gradient(quad_interp(y6), y6)

plt.figure()
plt.plot(y1, u_q7_prime_1, 'r-', y2, u_q7_prime_2, 'r-', y3, u_q7_prime_3, 'r-', y4, u_q7_prime_4, 'r-', y5, u_q7_prime_5, 'r-', y6, u_q7_prime_6, 'r-')
plt.grid(True)
plt.title("Derivative of u_q7")
save_fig("interp1d_quad_prime_test")

plt.figure()
# plt.plot(y1, u_L7_prime_1, 'c-', y2, u_L7_prime_2, 'c-', y3, u_L7_prime_3, 'c-', y4, u_L7_prime_4, 'c-', y5, u_L7_prime_5, 'c-', y6, u_L7_prime_6, 'c-')
plt.plot(y1, linear_least_squares(y1), 'c-', y2, linear_least_squares(y2), 'c-', y3, linear_least_squares(y3), 'c-', y4, linear_least_squares(y4), 'c-', y5, linear_least_squares(y5), 'c-', y6, linear_least_squares(y6), 'c-')
plt.plot(y1, linear_least_squares(y1), 'c-', label='Linear Least Squares')
plt.plot(y1, u_q7_prime_1, 'r-', y2, u_q7_prime_2, 'r-', y3, u_q7_prime_3, 'r-', y4, u_q7_prime_4, 'r-', y5, u_q7_prime_5, 'r-', y6, u_q7_prime_6, 'r-')
plt.plot(y6, u_q7_prime_6, 'r-', label="Quadratic")
plt.plot(y_s, u_s_prime, 'g-', label='Exact')
plt.grid(True)
plt.legend()
plt.title("Derivative of u_q7 solutions")
save_fig("shear_stress_solutions")

print("Shear stress at 0 um (linear): ", u_L7_prime_1[0], "[dynes/cm^2]")
print("Shear stress at 0 um (linear LLS): ", linear_least_squares(0), "[dynes/cm^2]")
print("Shear stress at 0 um (quadratic): ", u_q7_prime_1[0], "[dynes/cm^2]")
print("Shear stress at 0 um (exact): ", u_s_prime[0], "[dynes/cm^2]")
print("Shear stress at 20 um (linear): ", u_L7_prime_1[599], "[dynes/cm^2]")
print("Shear stress at 20 um (LLS): ", linear_least_squares(0.002), "[dynes/cm^2]")
print("Shear stress at 20 um (quadratic): ", u_q7_prime_1[599], "[dynes/cm^2]")
print("Shear stress at 20 um (exact): ", u_s_prime[99], "[dynes/cm^2]")

# plt.show()


# if __name__ == '__main__':
#     main()
