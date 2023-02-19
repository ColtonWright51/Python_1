"""
Solution to John Cotton's ME6570 HW4

Created by Colton Wright on 2/18/2023
"""

import numpy as np
import matplotlib.pyplot as plt
import os

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


def solve_couette(mu, dpdx, tau, h, n_nodes):
    n_elements = n_nodes-1
    h_e = h/n_nodes
    K = get_kgl(mu, h_e, n_elements)
    F = get_load(h_e, dpdx, tau, n_nodes)

    method = 3
    K, F = apply_EBC(K, F, method)

    d = np.linalg.solve(K,F)
    u = d  # the weights are equal to the estimate at the nodes
    if method == 1:
        u[-1] = u[-1] + 20  # EBC application METHOD A
    x = np.linspace(0, h, n_nodes)

    return x, u

def get_kgl(mu, h_e, n_elements):
    """
    Gets the stiffness matrix
    
    Parameters:
        mu (float): Elastic modulus
        h_e (float): Element length
        n_elements (int): Number of elements
    
    Returns:
        K (ndarray): Stiffness matrix
    """

    A = mu/h_e
    B = -mu/h_e
    kelm = np.array([[A, B], [B, A]])

    iconn = getConn(n_elements) # connectivity matrix (assumed to be defined elsewhere)
    
    K = np.zeros((n_elements+1, n_elements+1)) # preallocate the stiffness matrix
    for e in range(1, n_elements+1):
        for i in range(1, 3): # indices i and j local, ii and jj global
            for j in range(1, 3):
                ii = iconn[e-1,i-1]
                jj = iconn[e-1,j-1]
                K[ii-1, jj-1] = K[ii-1, jj-1] + kelm[i-1, j-1]
    
    return K

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
    f1 = np.zeros((n_nodes,1))
    f1[n_nodes-1] = tau
    
    # set up f2
    f2 = np.ones((n_nodes, 1))
    f2[0] = 1/2
    f2[n_nodes-1] = 1/2
    f2 = -dpdx*h_e*f2
    
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
n_nodes = 5
y, u = solve_couette(mu, dpdx, tau, h, n_nodes)
y2, u2 = solve_couette(mu, dpdx, tau, h, 50)

# Symbolic solution
y_s = np.linspace(0, h, 100)
u_s = (2*tau*y_s + dpdx*y_s**2 - 2*dpdx*h*y_s)/(2*mu)

plt.figure()
plt.plot(u,y, label='5 nodes')
plt.plot(u2, y2, label='50 nodes')
plt.plot(u_s, y_s, label='Symbolic')
plt.legend()
save_fig("solution")
plt.show()




# if __name__ == '__main__':
#     main()