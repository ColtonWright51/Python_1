"""
hw4_n_solve.py
Created by Colton Wright on 2/19/2023

Solve for the N basis functions for the 7th order approximation parent element

"""

import numpy as np
import sympy as sym

s_i = [-1, sym.Rational(-5,7), sym.Rational(-3,7), sym.Rational(-1,7), sym.Rational(1,7), sym.Rational(3,7), sym.Rational(5,7), 1]
s = sym.symbols('s')
n_elements = 3
h = 0.3
h_e = h/n_elements
n_nodes = 22

N = [1, 1, 1, 1, 1, 1, 1, 1]

for i in range(0, 8): # This indexed through the N's, N1, N2, N3...


    for j in range(0, 8):

        if i != j: # If i=j, we just skip this loop

            N[i] = N[i] * (s-s_i[j])/ (s_i[i]-s_i[j])

    N[i] = sym.expand(N[i])
    print("N_" + str(i+1) + ": ", N[i])
    print("N_" + str(i+1) + "': ", sym.diff(N[i], s))
    # sym.pprint(N[i])

# All N_i functions are found for our problem, lets solve for reduced stiffness matrix...

