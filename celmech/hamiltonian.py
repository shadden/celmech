from sympy import S, diff, lambdify, symbols, sqrt, cos, numbered_symbols, simplify
from scipy.integrate import odeint
import numpy as np

def make_Hamiltonian(H, pqpairs, params):
    class Hamiltonian(object):
        def __init__(self, **kwargs):
            self.H = H
            self.symH = H
            for param in params:
                try:
                    self.H = self.H.subs(param, kwargs[param.name])
                except KeyError:
                    raise AttributeError("Need to pass keyword {0} to Hamiltonian.".format(param.name))
            self.symderivs = {}
            self.Nderivs = []
            self.symvars = [item for pqpair in pqpairs for item in pqpair]
            for pqpair in pqpairs:
                p,q = pqpair
                self.symderivs[p.name] = -diff(self.symH, q)
                self.symderivs[q.name] = diff(self.symH, p)
                self.Nderivs.append(lambdify(self.symvars, -diff(self.H, q), 'numpy'))
                self.Nderivs.append(lambdify(self.symvars, diff(self.H, p), 'numpy'))
        def integrate(self, ic, times):
            y0 = []
            for pqpair in pqpairs:
                p,q = pqpair
                y0.append(ic[p.name])
                y0.append(ic[q.name])
            def diffeq(y, t):
                dydt = [deriv(*y) for deriv in self.Nderivs]
                #print(t, y, dydt)
                return dydt
            sol = odeint(diffeq, y0, times)
            retval = {}
            for i,pqpair in enumerate(pqpairs):
                p,q = pqpair
                retval[p.name] = sol[:,2*i]
                retval[q.name] = sol[:,2*i+1]
            return retval
    return Hamiltonian
