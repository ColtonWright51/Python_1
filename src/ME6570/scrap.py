import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import os

def getConn(n_elements):
    """
    Gets the connectivity matrix, a fairly standard array in FE codes, for a 1-D problem
    
    Parameters:
        n_elements (int): Number of elements in the 1D problem
    
    Returns:
        iconn (ndarray): The resulting connectivity matrix, which returns the global node
                         number given element number (i) and local node number (j)
    """

    iconn = np.zeros((n_elements+1, n_elements+1), dtype=int) # initialize array for speed.
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
print(getConn(3))
print(get_iconn(n_nodes=3, n_elements=3))