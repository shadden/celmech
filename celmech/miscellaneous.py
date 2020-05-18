import numpy as np
from scipy.special import k0,k1,p_roots
import warnings


def sk(k,y,tol=1.49e-08,rtol=1.49e-08,maxiter=50,miniter=1):
    """
    Approximate disturibing function coefficient described in 
    Hadden & Lithwick (2018) [#]_

    .. [#] `ADS link <https://ui.adsabs.harvard.edu/abs/2018AJ....156...95H/abstract>`
    
    Quadrature routine based on scipy.quadrature.

    Arguments
    ---------
    k : int
        Order of resonance
    y : float
        e / ecross; must be y<1
    tol, rtol: float, optional
        Control absolute and relative tolerance of integration.
        Iteration stops when error between last two iterates 
        is less than `tol` OR the relative change is less than 
        `rtol`.
    maxiter : int, optional
        Maximum order of Gaussian quadrature.
    miniter : int, optional
        Minimum order of Gaussian quadrature

    Returns
    -------
    val : float
        Gaussian quadrature approximation of s_k(y)
    """
    if y>1:
        raise ValueError("sk(k,y) called with y={:f}."
        "Value of y must be less than 1.")
    maxiter=max(miniter+1,maxiter)
    val = np.inf
    err = np.inf
    for n in xrange(miniter,maxiter+1):
        newval = _sk_integral_fixed_quad(k,y,n)
        err = abs(newval-val)
        val = newval
        if err<tol or err< rtol*abs(val):
            break
    else:
        warnings.warn("maxiter (%d) exceeded. Latest difference = %e" % (maxiter, err))
    return val
def _sk_integral_fixed_quad(k,y,Nquad):

    # Get numerical quadrature nodes and weight
    nodes,weights = p_roots(Nquad)
    
    # Rescale for integration interval from [-1,1] to [-pi,pi]
    nodes = nodes * np.pi
    weights = weights * 0.5
    arg1 = 2 * k * (1 + y * np.cos(nodes)) / 3
    arg2 = k * nodes + 4 * k * y * np.sin(nodes) / 3
    integrand = k0(arg1) * np.cos(arg2)
    return  (2/np.pi) * integrand @ weights

def Dsk(k,y,tol=1.49e-08,rtol=1.49e-08,maxiter=50,miniter=1):
    """
    Derivative of disturibing function coefficient s_k
    with respect to argument y. Coefficients are described 
    in Hadden & Lithwick (2018) [#]_

    .. [#] `ADS link <https://ui.adsabs.harvard.edu/abs/2018AJ....156...95H/abstract>`
    
    Quadrature routine based on scipy.quadrature.

    Arguments
    ---------
    k : int
        Order of resonance
    y : float
        e / ecross; must be y<1
    tol, rtol: float, optional
        Control absolute and relative tolerance of integration.
        Iteration stops when error between last two iterates 
        is less than `tol` OR the relative change is less than 
        `rtol`.
    maxiter : int, optional
        Maximum order of Gaussian quadrature.
    miniter : int, optional
        Minimum order of Gaussian quadrature

    Returns
    -------
    val : float
        Gaussian quadrature approximation of s_k(y)
    """
    if y>1:
        raise ValueError("sk(k,y) called with y={:f}."
        "Value of y must be less than 1.")
    maxiter=max(miniter+1,maxiter)
    val = np.inf
    err = np.inf
    for n in xrange(miniter,maxiter+1):
        newval = _Dsk_integral_fixed_quad(k,y,n)
        err = abs(newval-val)
        val = newval
        if err<tol or err< rtol*abs(val):
            break
    else:
        warnings.warn("maxiter (%d) exceeded. Latest difference = %e" % (maxiter, err))
    return val
def _Dsk_integral_fixed_quad(k,y,Nquad):

    # Get numerical quadrature nodes and weight
    nodes,weights = p_roots(Nquad)
    
    # Rescale for integration interval from [-1,1] to [-pi,pi]
    nodes = nodes * np.pi
    weights = weights * 0.5
    arg1 = 2 * k * (1 + y * np.cos(nodes)) / 3
    arg2 = k * nodes + 4 * k * y * np.sin(nodes) / 3
    integrand = -2 * k * k1(arg1) * np.cos(nodes) * np.cos(arg2) / 3 - 4 * k * k0(arg1) * np.sin(nodes) * np.sin(arg2) / 3
    return (2/np.pi) * integrand @ weights

def getOmegaMatrix(n):
    """
    Get the 2n x 2n skew-symmetric block matrix:
          [0 , I_n]
          [-I_n, 0 ]
    that appears in Hamilton's equations.

    Arguments
    ---------
    n : int
        Determines matrix dimension

    Returns
    -------
    numpy.array
    """
    return np.vstack(
        (
         np.concatenate([np.zeros((n,n)),np.eye(n)]).T,
         np.concatenate([-np.eye(n),np.zeros((n,n))]).T
        )
    )
