import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

a = 5
b = 1

def f(x,y):
    return np.vstack([y[1,:], np.zeros(y.shape[1])])

def bc(y0, y1):
    return np.array([y0[0] - a, y1[0] - b])

n = 100 # number of points
x = np.linspace(0,1,n)
# y0 = np.zeros((2,n))
y0 = np.random.randn(2,n)

# solve bvp
sol = solve_bvp(f, bc, x, y0)
print(sol)