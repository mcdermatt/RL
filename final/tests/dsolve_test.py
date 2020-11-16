import pydy
# from sympy import symbols, simplify, trigsimp, Abs, Heaviside, integrate
from scipy.integrate import odeint
from sympy import *
from sympy.physics.vector import init_vprinting, vlatex
from sympy.printing.pretty.pretty import pretty_print

init_vprinting(use_latex='mathjax', pretty_print=True)

# ------------Solving standard DiffEq--------------

x = symbols('x')
f,g = symbols('f, g', cls = Function)

diffeq = Eq(f(x).diff(x,x) - 2*f(x).diff(x) + f(x), sin(x))
pretty_print(diffeq)

#initial conditions
ics = {f(0) : 5, f(x).diff(x, 1).subs(x,0) : 3}
# ics = {}

soln = dsolve(diffeq,f(x), ics = ics)
pretty_print(soln)