from celmech import disturbing_function
import numpy as np
from collections import OrderedDict
from sympy import S
from celmech.disturbing_function import laplace_coefficient 
from celmech.disturbing_function import get_fg_coeffs
from itertools import combinations
import rebound

def masses_to_jacobi(masses):
    N = len(masses)
    mjac, Mjac = np.zeros(N), np.zeros(N)
    interior_mass = masses[0]
    for i in range(1,N):
        mjac[i] = masses[i]*interior_mass/(masses[i]+interior_mass) # reduced mass with interior mass
        Mjac[i] = masses[0]*(masses[i]+interior_mass)/interior_mass # mjac[i]*Mjac[i] always = m[i]*m[0]
        interior_mass += masses[i]
    mjac[0] = interior_mass
    return list(mjac), list(Mjac)

def masses_from_jacobi(mjac, Mjac):
    plus = Mjac[1] # m0+m1
    product = mjac[1]*Mjac[1] # m0m1
    minus = np.sqrt(plus**2 - 4*product) # m0-m1
    m0 = (plus + minus)/2.
    masses = [m0] + [Mjac[i]*mjac[i]/m0 for i in range(1,len(mjac))] # mjac[i]*Mjac[i] = m0mi
    return masses

def ActionAngleToXY(Action,angle):
        return np.sqrt(2*Action)*np.cos(angle),np.sqrt(2*Action)*np.sin(angle)

def XYToActionAngle(X,Y):
        return 0.5 * (X*X+Y*Y), np.arctan2(Y,X)

def pol_to_cart(R, phi):
    X = R*np.cos(phi)
    Y = R*np.sin(phi)
    return X, Y

def cart_to_pol(X, Y):
    R = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y,X)
    return R, phi

