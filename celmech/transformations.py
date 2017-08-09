from celmech import disturbing_function
import numpy as np
from collections import OrderedDict
from sympy import S
from celmech.disturbing_function import laplace_coefficient 
from celmech.disturbing_function import get_fg_coeffs
from itertools import combinations
import rebound

def jacobi_masses_from_sim(sim):
    ps = sim.particles
    mjac, Mjac, mu = np.zeros(sim.N), np.zeros(sim.N), np.zeros(sim.N)
    interior_mass = ps[0].m
    for i in range(1,sim.N):
        mjac[i] = ps[i].m*interior_mass/(ps[i].m+interior_mass) # reduced mass with interior mass
        Mjac[i] = ps[0].m*(ps[i].m+interior_mass)/interior_mass # mjac[i]*Mjac[i] always = ps[i].m*ps[0].m
        mu[i] = sim.G**2*Mjac[i]**2*mjac[i]**3 # Deck (2013) notation
    mjac[0] = interior_mass
    return list(mjac), list(Mjac), list(mu)

def jacobi_masses(masses):
    N = len(masses)
    mjac, Mjac, mu = np.zeros(N), np.zeros(N), np.zeros(N)
    interior_mass = masses[0]
    for i in range(1,N):
        mjac[i] = masses[i]*interior_mass/(masses[i]+interior_mass) # reduced mass with interior mass
        Mjac[i] = masses[0]*(masses[i]+interior_mass)/interior_mass # mjac[i]*Mjac[i] always = m[i]*m[0]
    mjac[0] = interior_mass
    return list(mjac), list(Mjac)

def synodic_Lambda_correction(sim, i1, i2, Lambda1, Lambda2):
    """
    Do a canonical transformation to correct the Lambdas for the fact that we have implicitly
    averaged over all the synodic terms we do not include in the Hamiltonian.
    """
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    ps = sim.particles
    s=0
    alpha = ps[i1].a/ps[i2].a
    deltan = ps[i1].n-ps[i2].n
    prefac = mu[i2]/Lambda2**2*mjac[i1]/Mjac[i1]/deltan
    deltalambda = ps[i1].l - ps[i2].l
    for j in range(1,150):
        s += laplace_coefficient(0.5, j, 0, alpha)*np.cos(j*deltalambda)
    s -= alpha*np.cos(deltalambda)
    s *= prefac
    return [Lambda1-s, Lambda2+s]

def ActionAngleToXY(Action,angle):
        return np.sqrt(2*Action)*np.cos(angle),np.sqrt(2*Action)*np.sin(angle)

def XYToActionAngle(X,Y):
        return 0.5 * (X*X+Y*Y), np.arctan2(Y,X)
