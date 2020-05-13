import numpy as np
from scipy.special import k0,k1

def sk(k,y,Nquad = 40):
    """
    Approximate disturibing function
    coefficient described in 
    Hadden & Lithwick (2018) [#]_
    
    .. [#] `ADS link <https://ui.adsabs.harvard.edu/abs/2018AJ....156...95H/abstract>`
    
    Arguments
    ---------
    k : int
        Order of resonance
    y : float
        e / ecross; must be y<1
    Nquad : int, optional
        Number of quadrature points to use to 
        determine integral.
    Returns
    -------
    float
    """
    assert y<1
    # Get numerical quadrature nodes and weight
    nodes,weights = np.polynomial.legendre.leggauss(N_QUAD_PTS)
    
    # Rescale for integration interval from [-1,1] to [-pi,pi]
    nodes = nodes * np.pi
    weights = weights * 0.5
    arg1 = 2 * k * (1 + y * np.cos(nodes)) / 3
    arg2 = k * nodes + 4 * k * y * np.sin(nodes) / 3
    integrand = k0(arg1) * np.cos(arg2)
    return integrand @ weights
