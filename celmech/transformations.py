from celmech import disturbing_function
import numpy as np
from collections import OrderedDict
from sympy import S
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

def masses_to_heliocentric(masses):
    r"""
    Convert list of masses to mass parameters 
    used in the definition of canonical 
    heliocentric Delauney actions.

    Arguments
    ---------
    masses : ndarray
        Masses of particles in the system with stellar mass
        listed first.

    Returns
    -------
    mu : list
        List of reduced masses, 

        .. math::

            \mu_i = \frac{M_* m_i}{M_* + m_i},

        for :math:`i\ge 1`. 
        mu[0] is the stellar mass.

    M : list
        List of
        
        .. math::

            M_i = M_* + m_i

        values for :math:`i\ge 1`. 
        M[0] is set to 0.
    """
    N=len(masses)
    masses_arr = np.array(masses)
    mstar = masses_arr[0]
    mplanet =masses_arr[1:]
    M = mplanet + mstar
    mu = mplanet * mstar / (mstar + mplanet)
    return [mstar] + mu.tolist(), [0] + M.tolist() 

def masses_from_heliocentric(mu,M):
    r"""
    Convert list of mass parameters 
    used in the definition of canonical 
    heliocentric Delauney actions to 
    star and planet masses. 

    Arguments
    ---------
    mu : array-like
        List of reduced masses, 

        .. math::

            \mu_i = \frac{M_* m_i}{M_* + m_i},

        for :math:`i\ge 1`. 

    M : array-like
        List of
        
        .. math::

            M_i = M_* + m_i

        values for :math:`i\ge 1`. 

    Arguments
    ---------
    masses : list 
        Masses of particles in the system with stellar mass
        listed first.

    """
    mu_arr = np.array(mu)[1:]
    M_arr = np.array(M)[1:]
    X = np.sqrt(M_arr**2 - 4 * M_arr * mu_arr)
    mstar_arr = 0.5 * (M_arr + X)
    m_arr = 0.5 * (M_arr - X)
    assert np.alltrue(np.isclose(mstar_arr,mstar_arr[0],rtol=1e-10))
    mstar = np.mean(mstar_arr)
    return [mstar] + m_arr.tolist()

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

