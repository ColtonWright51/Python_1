import numpy as np
import matplotlib.pyplot as plt
from sympy import *


# Input your phi funcs here...
def get_wronskian(functions):

    wronskian_matrix = Matrix()
    j_matrix = Matrix()
    for i in range(len(functions)):

        for j in range(len(functions)): # Calculate the derivatives in row i
            # Take the ith derivative of each element & save
            j_matrix = Matrix([j_matrix, diff(functions[j], x, i)])

        j_matrix = j_matrix.transpose()

        # Append entire row of derivatives to W matrix
        wronskian_matrix = Matrix([wronskian_matrix,j_matrix])

        j_matrix = Matrix() # Clear for next j loop

    print("\nSymbolic Wronskian matrix:\n")
    pprint(wronskian_matrix)

    return(wronskian_matrix.det())






x = Symbol('x')

funcs1 = [sin(x), sin(2*x), cos(x), cos(2*x)]

funcs2 = [-3*x, 4*x+x**2, 2*x**2]

funcs3 = [-3*x, x**3, 2*x**2]


wronskian = get_wronskian(funcs3)
print("\nSymbolic Wronskian:\n")
pprint(wronskian)

# If Wronskian is nonzero ANYWHERE on the domain, the functions are linearly
# independent.
plot(wronskian, xlim=(0,2))




