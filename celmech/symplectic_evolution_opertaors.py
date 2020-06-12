import numpy as np
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig,diff,Matrix
from celmech.hamiltonian import Hamiltonian
from celmech.disturbing_function import get_fg_coeffs, general_order_coefficient, secular_DF,laplace_B, laplace_coefficient
from celmech.disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
from celmech.transformations import masses_to_jacobi, masses_from_jacobi
from itertools import combinations
import rebound
import warnings
from abc import ABC, abstractmethod

class EvolutionOperator(ABC):
    def __init__(self,initial_state,dt):
        self._dt = dt
        self.state = initial_state
        self.particles = initial_state.particles
    @abstractmethod
    def apply(self):
        self.state.t += self._dt

    @property
    @abstractmethod
    def dt(self):
        return self._dt
    @dt.setter
    def dt(self,value):
        self._dt = dt

class KeplerianEvolutionOperator(EvolutionOperator):
    def __init__(self,initial_state,dt):
        super(KeplerianEvolutionOperator,self).__init__(initial_state,dt)
        self.G =  self.state.G
        self.m = np.array([p.m for p in self.state.particles[1:]]) 
        self.M = np.array([p.M for p in self.state.particles[1:]]) 
        self.GGMMmmm = (self.G*self.M)**2 * self.m**3
    def apply(self):
        lambda_dot = self.get_lambda_dot()
        dlambda = self.dt * lambda_dot
        ps = self.particles
        for p,dl in zip(ps[1:],dlambda):
            p.l += dl
        super().apply()
    def get_lambda_dot(self):
        ps = self.particles
        L = np.array([p.Lambda for p in ps[1:]])
        lambda_dot = self.GGMMmmm/L/L/L
        return lambda_dot

    @property
    def dt(self):
        return super().dt
    @dt.setter
    def dt(self,val):
        self._dt = val
from .poincare import LaplaceLagrangeSystem
from scipy.linalg import expm

class LinearSecularEvolutionOperator(EvolutionOperator):
    def __init__(self,initial_state,dt):
        super(LinearSecularEvolutionOperator,self).__init__(initial_state,dt)
        LL_system = LaplaceLagrangeSystem.from_Poincare(self.state)
        self.ecc_matrix = LL_system.Neccentricity_matrix
        self.inc_matrix = LL_system.Ninclination_matrix
        self.ecc_operator_matrix = expm(-1j * self.dt * self.ecc_matrix)
        self.inc_operator_matrix = expm(-1j * self.dt * self.inc_matrix)

    @property
    def dt(self):
        return super().dt
    @dt.setter
    def dt(self,val):
        self._dt = val
        self.ecc_operator_matrix = expm(-1j * self.dt * self.ecc_matrix)
        self.inc_operator_matrix = expm(-1j * self.dt * self.inc_matrix)

    def _get_x_vector(self):
        eta = np.array([p.eta for p in self.particles[1:]])
        kappa = np.array([p.kappa for p in self.particles[1:]])
        x =  (kappa - 1j * eta) / np.sqrt(2)
        return x

    def _get_y_vector(self):
        rho = np.array([p.rho for p in self.particles[1:]])
        sigma = np.array([p.sigma for p in self.particles[1:]])
        y =  (sigma - 1j * rho) / np.sqrt(2)
        return y

    def _set_x_vector(self,x):
        for p,xi in zip(self.particles[1:],x):
            p.kappa = np.sqrt(2) * np.real(xi)
            p.eta =  np.sqrt(2) * np.real(1j * xi)

    def _set_y_vector(self,y):
        for p,yi in zip(self.particles[1:],y):
            p.sigma = np.sqrt(2) * np.real(yi)
            p.rho =  np.sqrt(2) * np.real(1j * yi)

    def apply(self):
        x = self._get_x_vector()
        y = self._get_y_vector()
        xnew = self.ecc_operator_matrix @ x
        ynew = self.inc_operator_matrix @ y
        self._set_x_vector(xnew)
        self._set_y_vector(ynew)
        super().apply()

class LinearEccentricityResonancOperator(EvolutionOperator):
    r"""
    Evolution operator for linear equation of the form:
    .. math::
        dx/dt = -i e^{2i\theta} A \cdot x^* - \frac{i}{2}e^{i\theta} b
    Equations of motion derived from first- and second-order resonances,
    expanded to second order in eccentricities can be written in this form.

    An instance of `LinearEquationEvolutionOperator` provides
    a callable function that gives the evolution of h and k, where
        .. math::
            x = k - i h
    for a timestep dt and fixed angle value $\theta$.

    Parameters
    ----------
    A : ndarray, shape (M,M)
        The matrix appearing in the right-hand side
        of the equation.
    b : ndarray, shape (M)
        The vector in the r.h.s. of the equation.
    dt : float
        The timestep of the opertor.
    """
    def __init__(self,initial_state,dt):
        super(LinearResonancOperator,self).__init__(initial_state,indexIn,indexOut,res_vec ,Amtrx,bvec,dt)
        self.A = A
        self.Ainv = np.linalg.inv(A)
        self.b = b
        self.Ainv_dot_b = self.Ainv @ b
        vals,T = np.linalg.eigh(self.A)
        self.T = T
        self.Ttr = T.transpose()
        self.D = np.diag(vals)
        self.eigs = vals
        self._dt = dt
        self.cosh_lmbda_dt = np.cosh(self._dt * self.eigs)
        self.sinh_lmbda_dt = np.sinh(self._dt * self.eigs)
        self.indexIn = indexIn
        self.indexOut = indexOut
        self.particleIn = self.particles[indexIn]
        self.particleOut = self.particles[indexOut]
        self.res_vec = res_vec
        self.Lambdas_vec = np.array([-1 * res_vec[1] , res_vec[0]])
        self.Lambdas_Mtrx = np.linalg.inv([ self.Lambdas_vec , [1,1])
    @property
    def dt(self):
        return super().dt
    @dt.setter
    def dt(self,val):
        self._dt = val
        self.cosh_lmbda_dt = np.cosh(val * self.eigs)
        self.sinh_lmbda_dt = np.sinh(val * self.eigs)

    def _get_eta_vec(self):
        np.array([self.particleIn.eta,self.particleOut.eta])

    def _get_kappa_vec(self):
        np.array([self.particleIn.kappa,self.particleOut.kappa])

    def _get_theta(self):
        lvec = np.array([ self.particleIn.l, self.particle.Out.l])
        return res_vec @ lvec

    def _get_Lambdas(self):
        return np.array([self.particleIn.Lambda,self.particleOut.Lambda])
    def _get_Gammas(self):
        return np.array([self.particleIn.Gamma,self.particleOut.Gamma])

    def _set_eta_vec(self,eta_vec):
        etaIn,etaOut = eta_vec
        self.particleIn.eta = etaIn
        self.particleOut.eta = etaOut
        
    def _set_kappa_vec(self,kappa_vec):
        kappaIn,kappaOut = kappa_vec
        self.particleIn.kappa = kappaIn
        self.particleOut.kappa = kappaOut

    def _get_HK_vecs(self,s,c):
        eta = self._get_eta_vec()
        kappa = self._get_kappa_vec()
        H = self.Ttr @ (eta - 0.5 * np.sqrt(2) * self.Ainv_dot_b * s)
        K = self.Ttr @ (kappa + 0.5 * np.sqrt(2) * self.Ainv_dot_b * c)
        return H,K

    def _HK_to_etakappa(self,H,K,s,c):
        h = self.T @ H + 0.5 * np.sqrt(2) * self.Ainv_dot_b * s
        k = self.T @ K - 0.5 * np.sqrt(2) * self.Ainv_dot_b * c
        return h,k

    def _HKmap(self,Hvec,Kvec,s,c):
        s2 = 2 * s * c
        c2 = c*c - s*s
        K1vec = self.cosh_lmbda_dt * Kvec  +  self.sinh_lmbda_dt * (s2 * Kvec + c2 * Hvec)
        H1vec = self.cosh_lmbda_dt * Hvec  +  self.sinh_lmbda_dt * (c2 * Kvec - s2 * Hvec)
        return H1vec,K1vec

    def _advance_eccentricity_variables(self):
        theta = self._get_theta()   
        s,c = np.sin(theta),np.cos(theta)
        H,K = self._get_HK_vecs(s,c)
        H1,K1  = self._HKmap(H,K,s,c)
        eta1,kappa1 = self._HK_to_etakappa(H1,K1,s,c)
        self._set_eta_vec(eta1)
        self._set_kappa_vec(kappa1)
    def _set_Lambdas(self,Lambdas):
        self.particleIn = Lambdas[0]
        self.particleOut = Lambdas[0]

    def apply(self):
        Lambdas = self._get_Lambdas()
        Gammas0  = self._get_Gammas()
        C1 = self.Lambdas_vec @ Lambdas
        C2 = np.sum(Lambdas) - np.sum(Gammas0)
        self._advance_eccentricity_variables()
        Gammas1 = self._get_Gammas()
        Lambdas1 = self.LambdasMatrix @ np.array([ C1, C2 + Gammas1 ])
        self._set_Lambdas(Lambdas1)
        super().apply()
