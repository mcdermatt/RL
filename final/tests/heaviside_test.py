import pydy
# from sympy import symbols, simplify, trigsimp, Abs, Heaviside, integrate
from scipy.integrate import odeint
from sympy import *
from sympy.physics.vector import init_vprinting, vlatex
from sympy.printing.pretty.pretty import pretty_print
import numpy as np

init_vprinting(use_latex='mathjax', pretty_print=True)

# Workflow in previous integration scripts:
#    describe relationships between varaibles
#    kane = KanesMethod(inertial_frame, coordinates, speeds, kinematical_differentail_equations) -> EOM
#    rhs = genearate_ode_function (from pydy codegen) - generates numerical code from symbolic description of EOM
#    y = odeint(rhs, inital conditions (array), timesteps (array), args (constants)) (from scipy integrate)
#           odeint can NOT do analytical integration like we need to deal with step functions
#           so instead of converting to numerical function and then integrating
#           we need to integrate analytically and then evaluate

# proposed strategy:
#   get EOM the same as before
#   figure out analytical integration using sympy

# -------------------------
x = symbols('x')
f,g = symbols('f, g', cls = Function)

omega, fs, fk = symbols('omega, fs, fk')

# diffeq = Eq(f(x).diff(x,x) - 2*f(x).diff(x) + f(x), sin(x))
# pretty_print(diffeq)
# soln = integrate(diffeq)
# pretty_print(soln)

a = 1-Heaviside(x,0)
soln = integrate(a,(x,-1,4)) #get definite integral with 2nd argument
# soln = integrate(a) #leave blank for indef
# pretty_print(soln)

# TODO Figure out dummy variables
# integrate(Heaviside(x)) -> x*Heaviside(x) CORRECT
# integrate(1 - Heaviside(x)) ->-x*Heaviside(x) + x ALSO CORRECT
# 
# integrate()

b = DiracDelta(x)
pretty_print(b)
soln2 = integrate(b,x)
# soln2 = integrate(b,(x,-1,2)) # if interval includes x=0 -> 1, else 0 
print(soln2)

# t = np.linspace(0,10,11)
# soln2(t)

# rhs = generate_ode_function(soln) #needs to be in kanes form to work

# -------------------------
# fs, fk, omega,x = symbols('fs, fk, omega,x')
# f = symbols('f', cls=Function)
#
# f = Eq(1-Heaviside(Abs(omega),0)* fs + Heaviside(Abs(omega),0)* fk,0)
#
# pretty_print(f)
#
# soln = dsolve(f)
#
# # df = integrate(f(omega,fs,fk))
# # print(df)

# ----------------------
#scipy integrate syntax:
# integrate(exp(-x),(x,0,oo))