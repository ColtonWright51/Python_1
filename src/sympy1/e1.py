import sympy as sym
from sympy.solvers import pdsolve, ode
from sympy import Function, pprint



print("Attempt to solve the one-dimensional wave equation:")

x = sym.symbols('x')
t = sym.symbols('t')
a = sym.symbols('a')

y = Function('y')


yx = y(x,t).diff(x)
yxx = y(x,t).diff(x).diff(x)

yt = y(x,t).diff(t)
ytt = y(x,t).diff(t).diff(t)

wave_equation = yt-a**2*yx
pprint(wave_equation)
pprint(pdsolve(wave_equation))
# pprint(ode.classify_ode(wave_equation))

