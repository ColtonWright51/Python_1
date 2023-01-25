"""
hw2.py
Created on 1/24/2023 by Colton Wright

Solution to John Cotton's HW2 P2 in ME6570

"""


from sympy import *

def get_wronskian(functions, symbol, to_print=True, to_plot=True):

    # If Wronskian is nonzero ANYWHERE on the domain, the functions are linearly
    # independent. Zero everywhere, dependent.

    wronskian_matrix = Matrix()
    j_matrix = Matrix()
    for i in range(len(functions)):

        for j in range(len(functions)): # Calculate the derivatives in row i
            # Take the ith derivative of each element & save
            j_matrix = Matrix([j_matrix, diff(functions[j], symbol, i)])

        j_matrix = j_matrix.transpose()

        # Append entire row of derivatives to W matrix
        wronskian_matrix = Matrix([wronskian_matrix,j_matrix])

        j_matrix = Matrix() # Clear for next j loop

    wronskian = wronskian_matrix.det()
    if to_print:
        print("\nSymbolic Wronskian matrix:\n")
        pprint(wronskian_matrix)
        print("\nSymbolic Wronskian:\n")
        pprint(wronskian)

    if to_plot:
        plot(wronskian, xlim=(0,2))

    return(wronskian)

def main():

    x = Symbol('x')
    funcs1 = [sin(x), sin(2*x), cos(x), cos(2*x)]
    funcs2 = [-3*x, 4*x+x**2, 2*x**2]
    funcs3 = [-3*x, x**3, 2*x**2]

    wronskian = get_wronskian(funcs1, x)

if __name__ == '__main__':
    main()
