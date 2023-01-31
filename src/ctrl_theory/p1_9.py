import numpy as np
import matplotlib.pyplot as plt
import modules.help1 as h1

dt = 0.01
t_start = 0
t_end = 20
n_steps = int(round(t_end-t_start)/dt)

time = np.linspace(t_start, t_end, n_steps)

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

x1 = np.zeros(n_steps)
x1_t = np.zeros(n_steps)
x2 = np.zeros(n_steps)
x2_t = np.zeros(n_steps)
x3 = np.zeros(n_steps)
x4 = np.zeros(n_steps)
x5 = np.zeros(n_steps)
x6 = np.zeros(n_steps)
xp = np.zeros(n_steps)
xs = np.zeros(n_steps)

print(n_steps)

x1 = np.sin(time)
x2 = np.sin(time)



for i in range(1, n_steps):
    x1_t[i] = (x1[i]-x1[i-1])/dt
    x2_t[i] = (x2[i]-x2[i-1])/dt




plt.plot(time,x1,time,x1_t)
plt.show()