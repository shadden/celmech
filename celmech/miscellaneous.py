import numpy as np
from sympy import symbols, series
from scipy.special import k0,k1,p_roots
import warnings
from . import clibcelmech
from .nbody_simulation_utilities import get_canonical_heliocentric_orbits
from ctypes import POINTER,c_int,c_double,c_long

_machine_eps = np.finfo(np.float64).eps

def get_symbol(latex, subscript=None, **kwargs): # i=None, kwargs
    """
    Get a sympy sympy based on an input LaTeX string.
    Valid keyword arguments for the function ``sympy.symbols``
    can also be passed. 

    Arguments
    ---------
    latex : string
        LaTeX expression to render as a sympy symbol
    subscript : string or int, optional
        A subscript for the sympy symbol

    Returns
    -------
    sympy symbol
    """
    if subscript:
        return symbols(r"{0}_{{{1}}}".format(latex, subscript), **kwargs)
    else:
        return symbols(r"{0}".format(latex), **kwargs)

def get_symbol0(latex, subscript=None, **kwargs): # i=None, kwargs
    """
    Same as :func:`get_symbol`, but appends a "0" to the subscript.
    """
    if subscript:
        return symbols(r"{0}_{{{1}\,0}}".format(latex, subscript), **kwargs)
    else:
        return symbols(r"{0}_0".format(latex), **kwargs)

def sk(k,y,tol=1.49e-08,rtol=1.49e-08,maxiter=50,miniter=1):
    """
    Approximate disturibing function coefficient described in
    `Hadden & Lithwick (2018)`_

    .. _Hadden & Lithwick (2018): https://ui.adsabs.harvard.edu/abs/2018AJ....156...95H/abstract

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
    in `Hadden & Lithwick (2018)`_

    .. _Hadden & Lithwick (2018): https://ui.adsabs.harvard.edu/abs/2018AJ....156...95H/abstract
    
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
    zeros = np.zeros((n,n),dtype=int)
    I = np.eye(n,dtype=int)
    return np.vstack(
        (
         np.concatenate([zeros,I]).T,
         np.concatenate([-I,zeros]).T
        )
    )
######################################################
################ AMD Calculation #####################
######################################################
from scipy.optimize import brenth
def _F(e,alpha,gamma):
    """Equation 35 of Laskar & Petit (2017)"""
    denom = np.sqrt(alpha*(1-e*e)+gamma*gamma*e*e)
    return alpha*e -1 + alpha + gamma*e / denom

def _F_for_res_overlap(e,alpha,gamma,mutot):
    """Equation 35 of Laskar & Petit (2017)"""
    fByg = alpha**(0.825)
    ecross = 1/alpha - 1
    daBya= 1 - alpha
    ecrit = ecross * np.exp(-2.2 * mutot**(1/3) * daBya**(-4/3)) 
    denom = np.sqrt( (1 - e*e) * fByg**2 + e*e*alpha*gamma*gamma )
    e1 = gamma * np.sqrt(alpha) * e / denom 

    return fByg * e + e1 - fByg  * ecrit 

def critical_relative_AMD(alpha,gamma):
    r"""
    The critical value of 'relative AMD', :math:`{\cal C} = C/\Lambda_\mathrm{out}`,
    of a planet pair above which intersecting orbits are allowed.

    See Equation 29 of 
    `Laskar & Petit (2017) <https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..72L/abstract>`_

    Arguments
    ---------
    alpha : float
        The semi-major axis ratio, :math:`\alpha=a_\mathrm{in}/a_\mathrm{out}` of the planet pair.
    gamma : float
        The mass ratio of the planet pair, :math:`\gamma = m_\mathrm{in}/m_\mathrm{out}`.

    Returns
    -------
    Ccrit : float
        The value of the the critical AMD
        (:math:`C_c(\alpha,\gamma)` in the notation of 
        `Laskar & Petit (2017) 
        <https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..72L/abstract>`_
    """
    e0 = np.min((1,1/alpha-1))
    ec = brenth(_F,0,e0,args=(alpha,gamma))
    e1c = np.sin(np.arctan(gamma*ec / np.sqrt(alpha*(1-ec*ec))))
    curlyC = gamma*np.sqrt(alpha) * (1-np.sqrt(1-ec*ec)) + (1 - np.sqrt(1-e1c*e1c))
    return curlyC

def critical_relative_AMD_resonance_overlap(alpha,gamma,mutot):
    r"""
    The critical value of 'relative AMD', :math:`{\cal C} = C/\Lambda_\mathrm{out}`,
    of a planet pair above which resonance overlap can occur based
    on the resonance overlap criterion of Hadden & Lithwick (2018)

    Arguments
    ---------
    alpha : float
        The semi-major axis ratio, :math:`\alpha=a_\mathrm{in}/a_\mathrm{out}` of the planet pair.
    gamma : float
        The mass ratio of the planet pair, :math:`\gamma = m_\mathrm{in}/m_\mathrm{out}`.
    mutot : float
        The total mass of the planet pair relative to the star, i.e., 
        :math:`(\mu_\mathrm{in} + \mu_\mathrm{out}) / M_*`

    Returns
    -------
    Ccrit : float
        The value of the the critical AMD
        (:math:`C_c(\alpha,\gamma)` in the notation of 
        `Laskar & Petit (2017) 
        <https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..72L/abstract>`_
    """
    e0 = np.min((1,1/alpha-1))
    ec = brenth(_F_for_res_overlap,0,e0,args=(alpha,gamma,mutot))
    fByg = alpha**(0.825)
    denom = np.sqrt( (1 - ec*ec) * fByg**2 + ec*ec*alpha*gamma*gamma )
    e1c = gamma * np.sqrt(alpha) * ec / denom 
    curlyC = gamma*np.sqrt(alpha) * (1-np.sqrt(1-ec*ec)) + (1 - np.sqrt(1-e1c*e1c))
    return curlyC

def compute_AMD(sim):
    """
    Compute total AMD of a planetary system.
    
    The angular momentum deficit (AMD) of a 
    planetary system is the difference between
    the angular momentum of a hypothetical system
    with the same masses and semi-major axes but with
    circular, coplanar orbits and the actual 
    angular momentum of a planetary system.
    It is a conserved quantity of the purely
    secular dynamics of a system.

    Arguments
    ---------
    sim : :class:`rebound.Simulation`
        A REBOUND simulation of a planetary system.

    Returns
    -------
    AMD : float
        The value of the systems angular momentum 
        deficit.
    """

    pstar = sim.particles[0]
    Mstar = pstar.m
    Ltot = pstar.m * np.cross(pstar.xyz,pstar.vxyz)
    ps = sim.particles[1:]
    Lmbda=np.zeros(len(ps))
    G = np.zeros(len(ps))
    Lhat = np.zeros((len(ps),3))
    ch_orbits = get_canonical_heliocentric_orbits(sim)
    for k,p in enumerate(sim.particles[1:]):
        orb = ch_orbits[k]
        GMi = sim.G * (p.m + Mstar)
        mu = p.m*Mstar/(p.m + Mstar)
        Lmbda[k] = mu * np.sqrt(GMi * orb.a)
        G[k] = Lmbda[k] * np.sqrt(1-orb.e*orb.e)
        hvec = np.cross(p.xyz,p.vxyz)
        Lhat[k] = hvec / np.linalg.norm(hvec)
        Ltot = Ltot + p.m * hvec
    cosi = np.array([Lh.dot(Ltot) for Lh in Lhat]) / np.linalg.norm(Ltot)
    return np.sum(Lmbda) - np.sum(G * cosi)

def AMD_stable_Q(sim):
    """
    Test the AMD-stability of a planetary system.
    Returns :code:`True` if a planetary system is AMD-stable
    and :code:`False` if not. 

    Arguments
    ---------
    sim : :class:`rebound.Simulation`
     Simulation object to copmute stability criterion for.
     
    Returns
    -------
    bool : 
     :code:`True` if the sytem is AMD-stable, otherwise :code:`False`.
    """
    AMD = compute_AMD(sim)
    pstar = sim.particles[0]
    ps = sim.particles[1:]
    for i in range(len(ps)-1):
        pIn = ps[i]
        pOut = ps[i+1]
        orbIn = pIn.calculate_orbit(pstar)
        orbOut = pOut.calculate_orbit(pstar)
        alpha = orbIn.a / orbOut.a
        gamma = pIn.m / pOut.m
        LmbdaOut = pOut.m * np.sqrt(orbOut.a)
        Ccrit = critical_relative_AMD(alpha,gamma)
        C = AMD / LmbdaOut
        if C>Ccrit:
            return False
    return True

def AMD_stability_coefficients(sim,overlap=False):
    r"""
    Compute AMD stability coefficients 
    of the successive adjacent planet pairs 
    of a planetary system.

    A planet pair's AMD stability coefficicent 
    is defined as the total planetary system's
    AMD divided by the critical AMD required
    for the pair's orbits to cross.
    (Equation 58 of `Laskar & Petit (2017) 
    <https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..72L/abstract>`_)

    
    Arguments
    ---------
    sim : rebound.Simulation
      Simulation object to copmute AMD coefficients for.
    overlap : bool, optional
      If True, planet pairs' critical AMD values are computed
      as the critical AMD value for resonance overlap. By default 
      the critical values are computed as the value required
      for orbit crossing. 
     
    Returns
    -------
    ndarray : 
      Values of :math:`\beta = \frac{C}{\Lambda'C_c}`
      for planet pairs.
    """
    AMD = compute_AMD(sim)
    pstar = sim.particles[0]
    ps = sim.particles[1:]
    coeffs = np.zeros(len(ps)-1)
    for i in range(len(ps)-1):
        pIn = ps[i]
        pOut = ps[i+1]
        orbIn = pIn.calculate_orbit(pstar)
        orbOut = pOut.calculate_orbit(pstar)
        alpha = orbIn.a / orbOut.a
        gamma = pIn.m / pOut.m
        LmbdaOut = pOut.m * np.sqrt(orbOut.a)
        if overlap:
            mutot = (pIn.m + pOut.m) / pstar.m
            Ccrit = critical_relative_AMD_resonance_overlap(alpha,gamma,mutot)
        else:
            Ccrit = critical_relative_AMD(alpha,gamma)
        C = AMD / LmbdaOut
        coeffs[i] = C / Ccrit
    return coeffs

######################################################
######################## FMFT ########################
######################################################
p2d = np.ctypeslib.ndpointer(dtype = np.float,ndim = 2,flags = 'C')
_fmft = clibcelmech.fmft_wrapper
_fmft.argtypes =[p2d, c_int, c_double, c_double, c_int, p2d, c_long]
_fmft.restype = c_int
def _check_errors(ret, func, args):
    if ret<=0:
        raise RuntimeError("FMFT returned error code %d for the given arguments"%ret)
    return ret
_fmft.errcheck = _check_errors
def _nearest_pow2(x):
	return int(2**np.floor(np.log2(x)))
def frequency_modified_fourier_transform(time, z, Nfreq, method_flag = 3, min_freq = None, max_freq = None):
    """
    Apply the frequency-modified Fourier transfrorm algorithm of `Šidlichovský & Nesvorný (1996)`_
    to a time series to determine the series' principle Fourier modes. This function simply
    proivdes a wrapper to to C implementation written by D. Nesvorný available
    at `www-n.oca.eu/nesvorny/programs.html`_.


    .. _Šidlichovský & Nesvorný (1996): https://ui.adsabs.harvard.edu/abs/1996CeMDA..65..137S/abstract>
    .. _www-n.oca.eu/nesvorny/programs.html: https://www-n.oca.eu/nesvorny/programs.html

    Arguments
    ---------
    time : ndarray, shape (N,)
        Times of input data values.
    z : complex ndarray, shape (N,) 
      Input data time series in the form.
    Nfreq : int
        Number of Fourier modes to determine.
    method_flag : int
        The FMFT algorithm 
		Basic Fourier Transform algorithm           if   flag = 0;   not implemented   
		Modified Fourier Transform                  if   flag = 1;
		Frequency Modified Fourier Transform        if   flag = 2;
		FMFT with additional non-linear correction  if   flag = 3
    """
    output_arr = np.empty((Nfreq,3),order='C',dtype=np.float64)
    input_arr = np.array(np.vstack((time,np.real(z),np.imag(z))).T,order='C',dtype=np.float64)
    Ndata = _nearest_pow2(len(input_arr))
    dt = time[1]-time[0]
    _Nyq =  np.pi / dt
    if not min_freq:
        min_freq = -1 * _Nyq
    if not max_freq:
        max_freq = _Nyq
    _fmft( output_arr,
            Nfreq,
            min_freq,
            max_freq,
            c_int(method_flag),
            input_arr,
            c_long(Ndata)
    )
    return {x[0]:x[1]*np.exp(1j*x[2]) for x in output_arr}

def holman_weigert_stability_boundary(mu,e,Ptype=True):
    r"""
    Compute the critical semi-major axis represnting an approximate
    stability boundary for circumbinary planets in P- or S-type orbits.
    Formulas for critical semi-major axes are taken from `Holman & Wiegert (1999)`_

    .. _Holman & Wiegert (1999): https://ui.adsabs.harvard.edu/abs/1999AJ....117..621H/abstract

    Arguments
    ---------
    mu : float
      The mass-ratio of the binary,

      .. math::
        \mu = \frac{m_B}{m_A+m_B}

      where math:`m_A` and math:`m_B` are the component masses of the binary.
    e : float
      The eccentricity of the binary.
    Ptype : bool, optional
      If ``True`` (default) orbit is assumed to be a P-type circumbinary orbit.
      If ``False``, a S-type circum-primary/secondary orbit is considered.

    Returns
    -------
    aC : float
      The critical semi-major axis marking the stability boundary
    """
    if Ptype:
        if mu<0.1 or mu>0.5:
            warnings.warn("Input 'mu'={:.2g} is outside range [0.1,0.5] for which the stability boundary has been computed".format(mu))
        if e<0.0 or e>0.7:
            warnings.warn("Input 'e'={:.2g} is outside range [0.0,0.7] for which the stability boundary has been computed".format(e))
        aC =  1.6
        aC += 5.1 * e
        aC += -2.22 * e * e
        aC += 4.12 * mu
        aC += -4.27 * mu * e
        aC += -5.09 * mu * mu
        aC += 4.61 * mu * mu * e * e
    else:
        if mu<0.1 or mu>0.9:
            warnings.warn("Input 'mu'={:.2g} is outside range [0.1,0.5] for which the stability boundary has been computed".format(mu))
        if e<0.0 or e>0.8:
            warnings.warn("Input 'e'={:.2g} is outside range [0.0,0.7] for which the stability boundary has been computed".format(e))
        aC =  0.464
        aC += -0.38 * mu
        aC += -0.631 * e
        aC += 0.586 * mu * e
        aC += 0.150 * e * e
        aC += -0.198 * mu * e * e
    return aC
#######################
from sympy import diff, Matrix
def poisson_bracket(f,g,re_varslist,complex_varslist):
    r"""
    Calculate the Poisson bracket

    .. math::
        [f,g] = \sum_{i=1}^N
        \frac{\partial f}{\partial q_i}
        \frac{\partial g}{\partial p_i}
        -
        \frac{\partial f}{\partial p_i}
        \frac{\partial g}{\partial q_i}
        -
        i \sum_{j=1}^{M}
        \frac{\partial f}{\partial z_j}
        \frac{\partial g}{\partial \bar{z}_j}
        -
        \frac{\partial f}{\partial \bar{z}_j}
        \frac{\partial g}{\partial {z}_i}

    where :code:`re_varslist` is :math:`=(q_1,...,q_N,p_1,...,p_N)`
    and :code:`complex_varslist` is :math:`=(x_1,...,x_M,\bar{x}_1,...,\bar{x}_M)`.

    Arguments
    ---------
    f : sympy expression
        Function appearing in Poisson bracket.
    g : sympy expression
        Other function appearing in Poisson bracket.
    re_varslist : list of sympy symbols
        List of real canonical variables in the form 
        :math:`(q_1,...,q_N,p_1,...,p_N)`
    complex_varslist : list of sympy symbols
        List of complex canonical variables in the form 
        :math:`(x_1,...,x_M,\bar{x}_1,...,\bar{x}_M)`

    Returns
    -------
    sympy expression
    """
    br = 0
    if len(complex_varslist)>0:
        Omega_c =Matrix(-1j * getOmegaMatrix(len(complex_varslist)//2))
        gradf_c = Matrix([diff(f,v) for v in complex_varslist])
        gradg_c = Matrix([diff(g,v) for v in complex_varslist])
        br +=  gradf_c.dot(Omega_c * gradg_c)
    if len(re_varslist)>0:
        Omega_re=Matrix(getOmegaMatrix(len(re_varslist)//2))
        gradf_re = Matrix([diff(f,v) for v in re_varslist])
        gradg_re = Matrix([diff(g,v) for v in re_varslist])
        br+= gradf_re.dot(Omega_re * gradg_re)
    return br

def truncated_expansion(exprn,order_rules,max_order):
    r"""
    Expand a sympy expression up to a maximum order in a
    small book-keeping parameter after assigning variables 
    appearing in the expression a given order using the 
    `order_rules` argument.

    Arguments
    ---------
    exprn : sympy expression
        The original expression from which to calculate
        expansion.
    order_rules : dict
        A dictionary specifying what order various variables
        should be assumed to have in the book-keeping parameter.
        Each key-value pair ``{n:[x_1,x_2,..,x_m]}`` in ``order_rules``
        specifies that a set of variables 

        .. math::
            (x_1,...,x_m) \sim \mathcal{O}(\epsilon^n)
        
        where :math:`\epsilon` is the book-keeping parameter.
    max_order : int
        The order at which the resulting series expansion in 
        the book-keeping parameter :math:`\epsilon` should
        be truncated.

    Returns
    -------
    sympy expression
    """
    eps = symbols("epsilon")
    assert eps not in exprn.free_symbols, "Epsilon appears as a free symbols in 'exprn'."
    rule = dict()
    for n,variables in order_rules.items():
        rule.update({v:eps**n * v for v in variables})
    sexprn = series(exprn.subs(rule),eps,0,max_order+1)
    result = sexprn.removeO().subs({eps:1})
    return result
