from sympy import symbols, Eq, Function, pprint, Derivative, dsolve
from sympy.solvers.ode.systems import dsolve_system


t = symbols("t")

m1 = symbols("m1")
m2 = symbols("m")
m = symbols("m")
mp = symbols("mp")


b1 = symbols("b1")
b2 = symbols("b2")
b3 = symbols("b3")
b4 = symbols("b4")

k1 = symbols("k1")
k2 = symbols("k2")
k3 = symbols("k3")
k4 = symbols("k4")
ks = symbols("ks")

L = symbols("L")
L1 = symbols("L1")
L2 = symbols("L2")



# x1 = Function("x1")(t)
# x1_t = Derivative(x1, t)
# x1_tt = Derivative(x1, t, 2)
# x2 = Function("x2")(t)
# x2_t = Derivative(x2, t)
# x2_tt = Derivative(x2, t, 2)

x1 = symbols("x1")
x1_t = Derivative(x1, t)
x1_tt = Derivative(x1, t, 2)
x2 = symbols("x2")
x2_t = Derivative(x2, t)
x2_tt = Derivative(x2, t, 2)

x3 = Function("x3")(t)
x3_t = Derivative(x3, t)
x3_tt = Derivative(x3, t, 2)
x4 = Function("x4")(t)
x4_t = Derivative(x4, t)
x4_tt = Derivative(x4, t, 2)
x5 = Function("x5")(t)
x5_t = Derivative(x5, t)
x5_tt = Derivative(x5, t, 2)
x6 = Function("x6")(t)
x6_t = Derivative(x6, t)
x6_tt = Derivative(x6, t, 2)
xp = Function("xp")(t)
xp_t = Derivative(xp, t)
xp_tt = Derivative(xp, t, 2)
# xcg = Function("xcg")(t)
# xcg_t = Derivative(xcg, t)
# xcg_tt = Derivative(xcg, t, 2)
# xs = Function("xs")(t)
# xs_t = Derivative(xs, t)
# xs_tt = Derivative(xs, t, 2)

# J = Function("J")(t)
J = symbols("J")

# Geometric constraints
xcg = x5+L1/L*(x6-x5)
xcg_tt = Derivative(xcg, t, 2)
xs = x5+L2/L*(x6-x5)
theta = (x6-x5)/L
theta_tt = Derivative(theta, t, 2)

eq1 = Eq(m1*x3_tt, k1*(x1-x3)+b1*(x1_t-x3_t)+k3*(x5-x3)+b3*(x5_t-x3_t))
eq2 = Eq(m2*x4_tt, k2*(x2-x4)+b2*(x2_t-x4_t)+k4*(x6-x4)+b4*(x6_t-x4_t))
eq3 = Eq(m*xcg_tt, k3*(x3-x5)+k4*(x4-x6)+ks*(xp-xs)+b3*(x3_t-x5_t)+b4*(x4_t-x6_t))
eq4 = Eq(J*theta_tt, -L1*k3*(x3-x5)+(L-L1)*k4*(x4-x6)-(L1-L2)*ks*(xp-xs)-L1*b3*(x3_t-x5_t)+(L-L1)*b4*(x4_t-x6_t))
eq5 = Eq(mp*xp_tt, ks*(xs-xp))

eqs = [eq1,eq2,eq3,eq4,eq5]
funcs = [x3, x4, x5, x6, xp]
sol = dsolve_system(eqs, funcs, t)
pprint(sol)


