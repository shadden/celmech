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
        self.dt = dt
        self.state = initial_state

    @abstractmethod
    def apply(self):
        self.state.t += dt


class KeplerianEvolutionOperator(EvolutionOperator):
    def __init__(self,initial_state,dt):
        super(KeplerianEvolutionOperator,self).__init__(initial_state,dt)
        self.G =  self.state.G
        self.m = np.array([p.m for p in p in self.state.particles[1:]]) 
        self.M = np.array([p.M for p in p in self.state.particles[1:]]) 
        self.GGMMmmmm = (self.G*self.M)**2 * self.m**3

    def apply(self):
        ps = self.state.particles
        L = np.array([p.Lambda for p in ps[1:]])
        lambda_dot = self.GGMMmmm/L/L/L
        dlambda = self.dt * lambda_dot
        for p,dl in zip(ps[1:],dlambda):
            p.l += dlambda


