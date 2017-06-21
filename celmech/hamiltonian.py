from sympy import S, diff, lambdify, symbols, sqrt, cos, numbered_symbols, simplify
from scipy.integrate import ode
import numpy as np
import rebound
from celmech.transformations import jacobi_masses_from_sim, poincare_vars_from_sim
from celmech.disturbing_function import laplace_coefficient

class Hamiltonian(object):
    def __init__(self, H, pqpairs, initial_conditions, params, Nparams):
        self.pqpairs = pqpairs
        self.params = params
        self.Nparams = Nparams
        self.H = H
        self.derivs = {}

        self._update(H, pqpairs, initial_conditions, params, Nparams)
    def integrate(self, time):
        if time > self.integrator.t:
            try:
                self.integrator.integrate(time)
            except:
                raise AttributeError("Need to initialize Hamiltonian")
    def _update(self, h, pqpairs, initial_conditions, params, nparams):
        for pqpair in self.pqpairs:
            p,q = pqpair
            self.derivs[p] = -diff(self.h, q)
            self.derivs[q] = diff(self.h, p)
        
        self.nh = self.h
        for i, param in enumerate(self.params):
            try:
                self.nh = self.nh.subs(param, self.nparams[i])
            except keyerror:
                raise attributeerror("need to pass keyword {0} to hamiltonian.integrate".format(param))
        symvars = [item for pqpair in self.pqpairs for item in pqpair]
        self.nderivs = []
        for pqpair in self.pqpairs:
            p,q = pqpair
            self.nderivs.append(lambdify(symvars, -diff(self.nh, q), 'numpy'))
            self.nderivs.append(lambdify(symvars, diff(self.nh, p), 'numpy'))
        
        def diffeq(t, y):
            dydt = [deriv(*y) for deriv in self.Nderivs]
            return dydt
        self.integrator = ode(diffeq).set_integrator('lsoda')
        self.integrator.set_initial_value(initial_conditions, 0)

class HamiltonianPoincare(Hamiltonian):
    def __init__(self):
        self.integrator = None 
    def initialize_from_sim(self, sim, m_res):
        Lambda1, Lambda2, lambda1, lambda2, Gamma1, Gamma2, gamma1, gamma2 = symbols('Lambda1, Lambda2, lambda1, lambda2, Gamma1, Gamma2, gamma1, gamma2')
        actionanglepairs = [(Lambda1, lambda1), (Gamma1, gamma1), (Lambda2, lambda2), (Gamma2, gamma2)]
        m1, M1, mu1, mu2, m, f27, f31 = symbols('m1, M1, mu1, mu2, m, f27, f31')
        params = [m1, M1, mu1, mu2, m, f27, f31]
        mjac, Mjac, mu = jacobi_masses_from_sim(sim)
        Nf27, Nf31 = self._calculate_fs(m_res)
        Nparams = [mjac[1], Mjac[1], mu[1], mu[1], m_res, Nf27, Nf31]
        initial_conditions = poincare_vars_from_sim(sim)[:8]
        H = -mu1/(2*Lambda1**2) - mu2/(2*Lambda2**2) - m1/M1*mu2/Lambda2**2*(f27*sqrt(Gamma1/Lambda1)*cos((m+1)*lambda2 - m*lambda1 + gamma1) + f31*sqrt(2*Gamma2/Lambda2)*cos((m+1)*lambda2 - m*lambda1 + gamma2)) 
        super(HamiltonianPoincare, self).__init__(H, actionanglepairs, initial_conditions, params, Nparams)
    def _calculate_fs(self, m):
        alpha_res = (float(m)/(m+1))**(2./3.)
        f27 = 1./2*(-2*(m+1)*laplace_coefficient(0.5, m+1, 0, alpha_res) - alpha_res*laplace_coefficient(0.5, m+1, 1, alpha_res))
        f31 = 1./2*((2*m+1)*laplace_coefficient(0.5, m, 0, alpha_res) + alpha_res*laplace_coefficient(0.5, m, 1, alpha_res))
        return f27, f31
        
