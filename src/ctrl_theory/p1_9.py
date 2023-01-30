import numpy as np
import matplotlib.pyplot as plt
import modules.help1 as h1

dt = 0.01
t_start = 0
t_end = 20
n_steps = int(round(t_end-t_start)/dt)

x1 = np.zeros(n_steps+1)
x2 = np.zeros(n_steps+1)
x3 = np.zeros(n_steps+1)
x4 = np.zeros(n_steps+1)
x5 = np.zeros(n_steps+1)
x6 = np.zeros(n_steps+1)
xp = np.zeros(n_steps+1)
xs = np.zeros(n_steps+1)

m1 = 1
m2 = 1
m3 = 2
mp = 1

k1 = 1
k2 = 1
k3 = 1
k4 = 1
ks = 1

b1 = 1
b2 = 1
b3 = 1
b4 = 1

l1 = 1
l2 = 0.25

for i in range(1, n_steps+1):
    
