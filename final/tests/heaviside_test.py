import pydy
from sympy import symbols, simplify, trigsimp, Abs, Heaviside, integrate
from scipy.integrate import odeint


fs, fk, f, omega,x = symbols('fs, fk, f, omega,x')

f = (1-Heaviside(Abs(omega),0)* fs + Heaviside(Abs(omega),0)* fk)

print(f)

df = integrate(f(omega,fs,fk))

print(df)