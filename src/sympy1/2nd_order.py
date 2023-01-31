from sympy import symbols, Eq, Function, pprint, Derivative, dsolve

from sympy.solvers.ode.systems import dsolve_system

f, g = symbols("f g", cls=Function)


t = symbols("t")


x = Function("x")(t)
m = symbols("m")
b = symbols("b")
k = symbols("k")
x_t = Derivative(x, t)
x_tt = Derivative(x, t, 2)



eq = Eq(m*x_tt+b*x_t+k*x, 0)
pprint(dsolve(eq, x))


