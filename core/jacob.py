import symengine
import sympy as sym
from sympy import *

# vars = symengine.symbols('x y')
# f = symengine.sympify(['y*x**2', '5*x + sin(y)'])
# J = symengine.zeros(len(f),len(vars))
#
# for i, fi in enumerate(f):
#     for j, s in enumerate(vars):
#         J[i, j] = symengine.diff(fi, s)
#
# print(J)
#
# print(sym.Matrix(['y*x**2', '5*x + sin(y)']).jacobian(['x', 'y']))

j_vanderpol_Mat = sym.Matrix(['y', 'y-x*x*y - x']).jacobian(['x', 'y'])
j_Mat_T = j_vanderpol_Mat.T
print(j_vanderpol_Mat, j_Mat_T)
x, y = symbols("x y")
j_sym_Mat = j_vanderpol_Mat + j_Mat_T
j_sym_Mat_eval = j_sym_Mat.evalf(subs={x: 10.0, y: 10.0})
print(j_sym_Mat_eval.eigenvals())
e_vals = list(j_sym_Mat_eval.eigenvals().keys())
f1 = float(e_vals[0])
print(f1)
f2 = float(e_vals[1])
print(f2)