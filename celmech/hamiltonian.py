from sympy import S, diff, lambdify, symbols, sqrt, cos, numbered_symbols, simplify
from scipy.integrate import odeint
import numpy as np

class Hamiltonian(object):
    def __init__(self, H, pqpairs, params):
        self.pqpairs = pqpairs
        self.params = params
        self.H = H
        self.derivs = {}
        for pqpair in pqpairs:
            p,q = pqpair
            self.derivs[p] = -diff(self.H, q)
            self.derivs[q] = diff(self.H, p)
    def integrate(self, Nparams, initial_conditions, times):
        self._calculate_numerical_expressions(Nparams)

        def diffeq(y, t):
            dydt = [deriv(*y) for deriv in self.Nderivs]
            #print(t, y, dydt)
            return dydt
        sol = odeint(diffeq, initial_conditions, times)
        soldict = {}
        for i,pqpair in enumerate(self.pqpairs):
            p,q = pqpair
            soldict[p] = sol[:,2*i]
            soldict[q] = sol[:,2*i+1]
        return soldict
    def _calculate_numerical_expressions(self, Nparams):
        self.NH = self.H
        for i, param in enumerate(self.params):
            try:
                self.NH = self.NH.subs(param, Nparams[i])
            except KeyError:
                raise AttributeError("Need to pass keyword {0} to Hamiltonian.integrate".format(param))
        symvars = [item for pqpair in self.pqpairs for item in pqpair]
        self.Nderivs = []
        for pqpair in self.pqpairs:
            p,q = pqpair
            self.Nderivs.append(lambdify(symvars, -diff(self.NH, q), 'numpy'))
            self.Nderivs.append(lambdify(symvars, diff(self.NH, p), 'numpy'))
