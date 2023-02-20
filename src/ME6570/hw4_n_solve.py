"""
hw4_n_solve.py
Created by Colton Wright on 2/19/2023

Solve for the N basis functions for the 7th order approximation parent element

"""

import numpy as np
import sympy as sym
import sys
np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)
def getConn(n_elements):
    """
    Gets the connectivity matrix, a fairly standard array in FE codes, for a 1-D problem
    
    Parameters:
        n_elements (int): Number of elements in the 1D problem
    
    Returns:
        iconn (ndarray): The resulting connectivity matrix, which returns the global node
                         number given element number (i) and local node number (j)
    """

    iconn = np.zeros((n_elements+1, 2), dtype=int) # initialize array for speed.
    for ielem in range(1, n_elements+1):
        iconn[ielem-1, 0] = ielem
        iconn[ielem-1, 1] = ielem + 1
    
    return iconn

def get_iconn(n_nodes=8, n_elements=3):
    iconn = np.zeros((n_elements, n_nodes), dtype=int)
    for i in range(n_elements):
        for j in range(n_nodes):
            iconn[i,j] = i+j+1
    return iconn








s_i = [-1, sym.Rational(-5,7), sym.Rational(-3,7), sym.Rational(-1,7), sym.Rational(1,7), sym.Rational(3,7), sym.Rational(5,7), 1]
s = sym.symbols('s')
n_elements = 3
h = 0.3
h_e = h/n_elements
n_nodes = 22
n_nodes_per_element = 8

N = [1, 1, 1, 1, 1, 1, 1, 1]

for i in range(0, 8): # This indexed through the N's, N1, N2, N3...
    for j in range(0, 8):

        if i != j: # If i=j, we just skip this loop

            N[i] = N[i] * (s-s_i[j])/ (s_i[i]-s_i[j])

    N[i] = sym.expand(N[i])
    # print("N_" + str(i+1) + ": ", N[i])
    # print("N_" + str(i+1) + "': ", sym.diff(N[i], s))
# sym.plotting.plot(N[0], N[1], N[2], N[3], N[4], N[5], N[6], N[7], xlim=[-1, 1], ylim=[-1,1])


# All N_i functions are found for our problem, lets solve for reduced stiffness matrix...
kelm = [[0 for j in range(n_nodes_per_element)] for i in range(n_nodes_per_element)] # Create 8x8 python list. Don't use numpy array, it is storing sympy funcs
dxds = h_e/2

for i in range(n_nodes_per_element):
    for j in range(n_nodes_per_element):
        f = N[i]*N[j]
        kelm[i][j] = dxds*sym.integrate(f, (s, -1, 1))
        # print(kelm[i][j])

kelm = np.array(kelm, dtype='f')
print("kelm rounded:\n", np.around(kelm, decimals=3))
iconn2 = np.array([[ 1,  2,  3,  4,  5,  6,  7,  8],
                   [ 8,  9, 10, 11, 12, 13, 14, 15],
                   [15, 16, 17, 18, 19, 20, 21, 22]])


K = np.zeros((n_nodes, n_nodes)) # preallocate the stiffness matrix
# print(K)
for e in range(n_elements):
    for i in range(8): # indices i and j local, ii and jj global
        # print("i: ", i)
        for j in range(8):
            ii = iconn2[e,i]-1
            jj = iconn2[e,j]-1
            # print("jj: ", jj)
            K[ii, jj] = K[ii, jj] + kelm[i, j]
        # print("ii: ", ii)

print("K rounded:\n", np.around(K, decimals=5))



