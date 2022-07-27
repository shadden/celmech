from sympy import S, diff, lambdify, symbols, sqrt, cos,sin, numbered_symbols, simplify,binomial, hyper, hyperexpand, Function, factorial,elliptic_k,elliptic_e, expand_trig, Function,bell
from sympy import I,exp,series
from . import clibcelmech
from ctypes import Structure, c_double, POINTER, c_float, c_int, c_uint, c_uint32, c_int64, c_long, c_ulong, c_ulonglong, c_void_p, c_char_p, CFUNCTYPE, byref, create_string_buffer, addressof, pointer, cast
from scipy.integrate import quad
import math
import numpy as np
from scipy.optimize import leastsq
from scipy.special import poch,factorial2,binom,factorial,gamma,hyp2f1
from collections import defaultdict
import warnings

def get_df_term_latex(k1,k2,k3,k4,k5,k6,nu1,nu2,nu3,nu4,l1,l2,indexIn,indexOut):
    r"""
    Get the latex expression for the disturbing function coefficient

    .. math::
        C_{\pmb{k}}^{\pmb{\nu},\pmb{l}}(\alpha_{i,j})

    Returns
    -------
    latex
    """
    assert l1 + l2 < 2, "I'm not going to print something so ridiculuous!"
    C  = r"\tilde{{C}}_{{({0},{1},{2},{3},{4},{5})}}".format(k1,k2,k3,k4,k5,k6)
    if l1==1:
        C = r"\alpha\frac{d}{d\alpha}" + C
    if l2==1:
        C = r"\alpha\frac{d}{d\alpha}" + C
    C += r"^{{({0},{1},{2},{3})}}".format(nu1,nu2,nu3,nu4)
    C += r"(\alpha_{{{0},{1}}})".format(indexIn,indexOut)
    term = r"{0}\frac{{Gm_{1}m_{2}}}{{a_{{{2},0}}}}".format(["-","+"][l2],indexIn,indexOut)
    term += C
    e1exp = abs(k3) + 2*nu3
    e2exp = abs(k4) + 2*nu4
    s1exp = abs(k5) + 2*nu1
    s2exp = abs(k6) + 2*nu2
    if l1==1:
        term+=r"\left(\frac{{\delta a_{0}}}{{a_{{{0},0}}}}\right)".format(indexIn)
    if l2==1:
        term+=r"\left(\frac{{\delta a_{0}}}{{a_{{{0},0}}}}\right)".format(indexOut)
    if e1exp > 0:
        if e1exp == 1:
            term += r"e_{0}".format(indexIn)
        else:
            term += r"e_{0}^{1}".format(indexIn, e1exp)
    if e2exp > 0:
        if e2exp == 1:
            term += r"e_{0}".format(indexOut)
        else:
            term += r"e_{0}^{1}".format(indexOut, e2exp)
    if s1exp > 0:
        if s1exp == 1:
            term += r"s_{0}".format(indexIn)
        else:
            term += r"s_{0}^{1}".format(indexIn, s1exp)
    if s2exp > 0:
        if s2exp == 1:
            term += r"s_{0}".format(indexOut)
        else:
            term += r"s_{0}^{1}".format(indexOut, s2exp)
    arg = r""
    if k1 == 0 and k2 == 0 and k3 == 0 and k4 == 0 and k5==0 and k6 == 0:
        pass # don't write cosine
    else:
        if k1 < 0:
            arg += r"-"
        if abs(k1) > 1:
            arg += r"{0}".format(abs(k1))
        if k1 != 0:
            arg += r"\lambda_{0}".format(indexOut)
        if k2 != 0:
            arg += r"+" if k2 > 0 else r"-"
            if abs(k2) > 1:
                arg += r"{0}".format(abs(k2))
            arg += r"\lambda_{0}".format(indexIn)
        if k3 != 0:
            arg += r"+" if k3 > 0 else r"-"
            if abs(k3) > 1:
                arg += r"{0}".format(abs(k3))
            arg += r"\varpi_{0}".format(indexIn)  
        if k4 != 0:
            arg += r"+" if k4 > 0 else r"-"
            if abs(k4) > 1:
                arg += r"{0}".format(abs(k4))
            arg += r"\varpi_{0}".format(indexOut)
        if k5 != 0:
            arg += r"+" if k5 > 0 else r"-"
            if abs(k5) > 1:
                arg += r"{0}".format(abs(k5))
            arg += r"\Omega_{0}".format(indexIn)
        if k6 != 0:
            arg += r"+" if k6 > 0 else r"-"
            if abs(k6) > 1:
                arg += r"{0}".format(abs(k6))
            arg += r"\Omega_{0}".format(indexOut)
        term += r"\cos({0})".format(arg)
        
    return term

def get_df_coefficient_symbol(k1,k2,k3,k4,k5,k6,nu1,nu2,nu3,nu4,l1,l2,indexIn,indexOut):
    r"""
    Get a sympy symbol for the disturbing function coefficient

    .. math::
        C_{\pmb{k}}^{\pmb{\nu},\pmb{l}}(\alpha_{i,j})

    Returns
    -------
    sympy symbol
    """
    symbol_str  = r"C_{{({0}\,{1}\,{2}\,{3}\,{4}\,{5})}}".format(k1,k2,k3,k4,k5,k6)
    symbol_str += r"^{{({0}\,{1}\,{2}\,{3})\,({4}\,{5})}}".format(nu1,nu2,nu3,nu4,l1,l2)
    symbol_str += r"(\alpha_{{{0}\,{1}}})".format(indexIn,indexOut)
    return symbols(symbol_str)

def _delta(*args):
    """
    Return 0 if all arguments are 0, otherwise return 1.
    """
    intarr  = np.array([args],dtype=np.int64)
    if np.alltrue(intarr == 0):
        return 0
    else:
        return 1
    
def df_arguments_dictionary(Nmax):
    r"""
    Get the arguments appearing in the disturbing function
    up to order Nmax. Arguments are returned as a nested dictionary.
    The outer level keys are orders. 
    The inner level keys are 
    :math:`\Delta k = |k_1-k_2|`, 
    denoting the order of MMR for which the argument appears. 
    The values of the inner level dictionaries are lists of
    tuples containing :math:`(k_3,k_4,k_5,k_6)`.
    Specifically, the dictionary is returned in the form:
    
    | {
    |     0:{0:[(0,0,0,0)],
    |     1:{1:[(-1,0,0,0),(0,-1,0,0)]},
    |     ...
    |     Nmax:{
    |         0:[(k3,k4,k5,k6)],
    |         ...
    |         Nmax:[(k3,k4,k5,k6),...]
    |     }
    | }
    
    Arguments
    ---------
    Nmax: int
        Maximum order of dictionary arguments.

    Returns
    -------
    arguments : defaultdict
        Nested defaultdicts containing the cosine
        arguments appearing in the disurbing function.
    """
    args_dict = defaultdict(lambda: defaultdict(list))
    Nmax_by_2 = Nmax // 2
    for h in range(Nmax_by_2 + 1):
        khi = 2 * Nmax_by_2
        klo = -khi * _delta(h)
        for k in range(klo,khi + 1):
            hplusk = (h+k)
            hminusk = (h-k)
            s_hi = Nmax - abs(hplusk) - abs(hminusk)
            s_lo = -s_hi * _delta(h,k)
            for s in range(s_lo,s_hi + 1):
                s1_hi = Nmax - abs(hplusk) - abs(hminusk) - abs(s)
                s1_lo = -1 * s1_hi * _delta(h,k,s)
                for s1 in range(s1_lo,s1_hi + 1):
                    dj = 2 * h + s + s1
                    sgn = 1 if dj == 0 else np.sign(dj)
                    k3=-sgn*s
                    k4=-sgn*s1
                    k5=-sgn*(hplusk)
                    k6=-sgn*(hminusk)
                    N = abs(k3) + abs(k4) + abs(k5) + abs(k6)
                    args_dict[N][dj*sgn].append((k3,k4,k5,k6))
    return args_dict

def _nucombos(nutot):
    nucombos = []
    for nu1 in range(nutot+1):
        for nu2 in range(nutot+1-nu1):
            for nu3 in range(nutot+1-nu1-nu2):
                nu4 = nutot - nu1 - nu2 - nu3
                nucombos.append((nu1, nu2, nu3, nu4))
    return nucombos

def _lcombos(ltot):
    lcombos = []
    for l1 in range(ltot+1):
        l2 = ltot - l1
        lcombos.append((l1, l2))
    return lcombos
def k_nu_depend_on_inclinations(k,nu):
    _,_,_,_,k5,k6 = k
    nu1,nu2,_,_ = nu
    arr=np.array([k5,k6,nu1,nu2])
    return np.any(arr!=0)
def k_nu_depend_on_eccentricities(k,nu):
    _,_,k3,k4,_,_ = k
    _,_,nu3,nu4 = nu
    arr=np.array([k3,k4,nu3,nu4])
    return np.any(arr!=0)
def list_resonance_terms(p,q,min_order=None,max_order=None,eccentricities=True,inclinations=True):
    """
    Generate the list of disturbing function terms for a
    p:p-q resonance with eccentricity/inclination order
    between min_order and max_order

    Arguments
    ---------
    p : int
        Determines resonance
    q : int
        Order of the resonance
    min_order : int, optional
        Minimum order in eccentricities and inclinations to include. Defaults to order of the resonance q (leading order)
    max_order: int
        Maximum order in eccentricities and inclinations to include. Defaults to order of the resonance q (leading order)
    eccentricities: bool, optional
        By default, includes all eccentricity terms.
        Can set to False to exclude any eccentricity terms (e.g., fixed circular orbits).
    inclinations: bool, optional
        By default, includes all inclination terms.
        Can set to False to exclude any inclination terms (e.g., co-planar systems).

    Returns
    -------
    term : list
        A list of disturbing function terms.
        Each entry in the list is of the form
        (k_vec, nu_vec) (see PoincareHamiltonian.add_cosine_term in poincare.py)
    """
    if not min_order:
        min_order = q
    if not max_order:
        max_order = q
    args_dict = df_arguments_dictionary(max_order)
    args = []
    for N in range(min_order,max_order+1):
        for q1 in range(q,N+1,q):
            if (N-q1) % 2:
                continue
            p1 = (q1//q) * p
            for N1 in range(q1,N+1,2):
                nutot = (N-N1)//2
                for arg in args_dict[N1][q1]:
                    for nu_vec in _nucombos(nutot):
                        k_vec = (p1,q1 - p1,*arg)
                        if inclinations == False and k_nu_depend_on_inclinations(k_vec, nu_vec) == True:
                            continue
                        if eccentricities == False and k_nu_depend_on_eccentricities(k_vec, nu_vec) == True:
                            continue
                        args.append((k_vec,nu_vec))
    return args

def list_secular_terms(min_order,max_order,eccentricities=True,inclinations=True):
    """
    Generate the list of secular disturbing function terms 
    with eccentricity/inclination order between min_order and max_order.

    Arguments
    ---------
    min_order : int
     Minimum order in eccentricities and inclinations to include.
    max_order : int
     Maximum order in eccentricities and inclinations to include.
    eccentricities: bool, optional
        By default, includes all eccentricity terms.
        Can set to False to exclude any eccentricity terms (e.g., fixed circular orbits).
    inclinations: bool, optional
        By default, includes all inclination terms.
        Can set to False to exclude any inclination terms (e.g., co-planar systems).

    Returns
    -------
    term : list
        A list of disturbing function terms. 
        Each entry in the list is of the form
        (k_vec, nu_vec) (see PoincareHamiltonian.add_cosine_term in poincare.py)
    """
    args_dict = df_arguments_dictionary(max_order)
    args = []
    Nmax1 = (max_order//2) * 2 
    Nmin1 = (min_order//2) * 2 
    for N in range(0,Nmax1 + 1,2):
        argsN = args_dict[N][0]
        nutot_min = max( (Nmin1 - N)//2 , 0)
        nutot_max = (Nmax1 - N)//2 
        for nutot in range(nutot_min,nutot_max + 1):
            for nu_vec in _nucombos(nutot):
                for arg in argsN:
                    k_vec = (0,0,*arg)
                    if inclinations == False and k_nu_depend_on_inclinations(k_vec, nu_vec) == True:
                        continue
                    if eccentricities == False and k_nu_depend_on_eccentricities(k_vec, nu_vec) == True:
                        continue
                    args.append((k_vec,nu_vec))
    return args

def laplace_b(s,j,n,alpha):
    r"""
    Calculates :math:`n`th derivative with respect to :math:`\alpha` of Laplace coefficient :math:`b_s^j(\alpha)`.
    Uses recursion and scipy special functions. 

    Arguments
    ---------
    s : float 
        half-integer parameter of Laplace coefficient. 
    j : int 
        integer parameter of Laplace coefficient. 
    n : int 
        Specify the :math:`n`th derivative with respect to :math:`alpha`. 
    alpha : float
        Semi-major axis ratio, :math:`\alpha=a_{in}/a_{out}`.

    Returns
    -------
    float
        The value of the coefficient.
    """
    assert alpha>=0 and alpha<1, "alpha not in range [0,1): alpha={}".format(alpha)
    if j<0:
        return laplace_b(s,-j,n,alpha)
    if n >= 2:
        return s * (
            laplace_b(s+1,j-1,n-1,alpha) 
            -  2 * alpha * laplace_b(s+1,j,n-1,alpha)
            + laplace_b(s+1,j+1,n-1,alpha)
            - 2 * (n-1) * laplace_b(s+1,j,n-2,alpha)
        )
    if n==1:
        return s * (
            laplace_b(s+1,j-1,0,alpha) 
            - 2 * alpha * laplace_b(s+1,j,0,alpha) 
            + laplace_b(s+1,j+1,0,alpha)
        )
    return 2 * poch(s,j) * alpha**j * hyp2f1(s,s+j,j+1,alpha**2)/ factorial(j)

def evaluate_df_coefficient_dict(coeff_dict,alpha):
    r"""
    Evaluate a dictionary representing a sum
    of Laplace coefficient terms like those returned
    by df_coefficient_C and df_coefficient_Ctilde evaluated at a 
    specific value of semi-major axis ratio, alpha.

    Arguments
    ---------
    coeff_dict : dictionary
        Dictionary with entries {(p,(s,j,n)) : coeff}
        representing a sum of Laplace coefficients:

        .. math::
         \mathrm{coeff} \times \alpha^p \frac{ d^n b_s^{(j)}(\alpha)} { d\alpha^n}
    alpha : float
        Value of semi-major axis ratio, :math:`\alpha=a_i/a_j`, appearing
        as an argument of Laplace coefficients.

    Returns
    -------
    float :
        The sum of Laplace coefficeint terms represented
        by dictionary entries.
    """
    tot = 0
    for key,val in coeff_dict.items():
        if key[0] == 'indirect':
            pwer = key[1]
            rt_alpha_inv = 1 / np.sqrt(alpha) 
            tot += val * rt_alpha_inv**pwer
        else:
            p,arg = key
            tot += val * alpha**p * laplace_b(*arg,alpha)
    return tot


def calB(n,k,p):
    arglist = [negative_binom(p,l) * factorial(l) for l in range(1,n-k+3)]
    return bell(n,k,arglist)

def falling_factorial(x,n):
    return poch(-x,n) * (-1)**n

def _Psi_coeff(l1,l2,p1,p2,m1,m2,r1,r2):
    return binom(l1,m1) * binom(l2,m2) *\
    falling_factorial(-2*r1 - p2/2,m2) * falling_factorial(-p1/2,m1) *\
    calB(l1-m1,r1,2) * calB(l2-m2,r2,-2)

def _calc_df_coefficient_C_l1_l2_Taylor_coeff(l1,l2,p1,p2,derivs_list):
    tot = 0
    l1fact_inv = 1/factorial(l1)
    l2fact_inv = 1/factorial(l2)
    prefactor = l1fact_inv * l2fact_inv
    for m1 in range(l1+1):
        for r1 in range(l1-m1+1):
            for m2 in range(l2+1):
                for r2 in range(l2-m2+1):
                    val = prefactor * derivs_list[r1+r2] * _Psi_coeff(l1,l2,p1,p2,m1,m2,r1,r2)
                    tot=tot+val        
    return tot

def _p1_p2_from_k_nu(kvec,nuvec):
    r"""
    Convenience function for calculate integers p1 and p2 
    defined as 
        p1 = abs(k3) + abs(k5) + 2 * nu1 + 2 * nu3
        p2 = 4 + abs(k4) + abs(k6) + 2 * nu2 + 2 * nu4

    These definitions appear in the formula for the expansion
    of DF coefficients in powers delta
    """
    k1,k2,k3,k4,k5,k6 = kvec
    nu1,nu2,nu3,nu4 = nuvec
    p1 = abs(k3) + abs(k5) + 2 * nu1 + 2 * nu3
    p2 = 4 + abs(k4) + abs(k6) + 2 * nu2 + 2 * nu4
    return p1,p2

def evaluate_df_coefficient_delta_expansion(Coeff_dict,p1,p2,lmax,alpha):
    r"""
    Calculate coefficients of the Taylor series of `Coeff_dict` in 
    powers math:`\delta_i` and math::`\delta_j` evaluated at semi-major
    axis ratio math:`\alpha=a_1/a_2`.

    Arguments
    ---------
    Coeff_dict : dict
        Dictionary representation of DF coefficient
    p1 : int
        Integer specific to the DF term's (k,nu) values
        given by
            p1 = abs(k3) + abs(k5) + 2 * nu1 + 2 * nu3
    p2 : int
        Integer specific to the DF term's (k,nu) values
        given by
            p2 = 4 + abs(k4) + abs(k6) + 2 * nu2 + 2 * nu4
    lmax : int
        Maximum power of Taylor expansion.
    alpha : float
        Semi-major axis ratio at which to evaluate coefficients.
    """
    answer = dict()
    C = Coeff_dict
    # Derivatives of C w.r.t. alpha
    NC_derivs = [evaluate_df_coefficient_dict(C,alpha)]
    for ltot in range(lmax+1):
        C=deriv_df_coefficient(C)
        NC_derivs.append(alpha**(ltot+1) * evaluate_df_coefficient_dict(C,alpha))
        for lIn in range(ltot+1):
            lOut = ltot - lIn
            answer[(lIn,lOut)] = _calc_df_coefficient_C_l1_l2_Taylor_coeff(lIn,lOut,p1,p2,NC_derivs)
    return answer

# Vector of resonance coefficients
def get_res_coefficient_vector(j,k,include_indirect_terms=True):
    r"""
    Get a vector comprised of all sub-resonance coefficients for the j:j-k mean motion resonance.

    Arguments
    ---------
    j : int
        Specify the j:j-k resonance
    k : int
        Order of the resonance
    include_indirect_terms :  boole, optional
        whether the contribution of indirect terms should be
        accounted for when computing the coefficients. default
        is true.

    Returns
    -------
    vals : ndarray
        Array of coefficient values from l=0 to l=k
    """ 
    res_pratio = float(j - k) /float(j)
    alpha = res_pratio**(2./3.)
    res_ecc_arg = lambda j,k,l: (j,k-j,-l,l-k,0,0,0,0,0,0) # k and nu values
    Cjkl = lambda j,k,l,alpha: evaluate_df_coefficient_dict(df_coefficient_C(*res_ecc_arg(j,k,l),include_indirect_terms=include_indirect_terms),alpha)
    vals = np.array([Cjkl(j,k,l,alpha) for l in range(k+1)],dtype=np.float64)
    return vals

def get_fg_coefficients(res_j,res_k):
    """Get 'f' and 'g' coefficients for approximating the disturbing function coefficients associated with an MMR."""
    res_pratio = float(res_j - res_k) /float(res_j)
    alpha = res_pratio**(2./3.)
    vec = get_res_coefficient_vector(res_j,res_k)
    resids_vec_fn = lambda fg: vec - np.array([binomial(res_k,l) * fg[0]**(l) * fg[1]**(res_k-l) for l in range(res_k+1)],dtype=np.float64)
    ex = (1-alpha)
    f0 = -1 / ex
    g0 = 1 / ex
    f,g = leastsq(resids_vec_fn,(f0,g0))[0]
    return f,g


def NCOd0(a,b,c):
    """
    Value of of Newcomb operator
        X^{a,b}_{c,0}
        
    Arguments
    ---------
    a : int
    b : int
    c : int
    
    Returns
    -------
    X^{a,b}_{c,0} : float
    """
    if c==0: return 1
    if c==1: return b - a /2
    nc1 = NCOd0(a,b+1,c-1)
    nc2 = NCOd0(a,b+2,c-2)
    return  (2 * (2*b-a) * nc1 + (b-a) * nc2 )/ c / 4

def NewcombOperator(a,b,c,d):
    """
    Value of of Newcomb operator
        X^{a,b}_{c,d}
        
    Arguments
    ---------
    a : int
    b : int
    c : int
    d : int
    
    Returns
    -------
    X^{a,b}_{c,d} : float
    """
    if c<0 or d<0: return 0
    if d==0: return NCOd0(a,b,c)
    tot = -2 * (2 * b + a) * NewcombOperator(a,b-1,c,d-1)
    tot += -1 * (b + a) * NewcombOperator(a,b-2,c,d-2)
    tot += -1 * (c - 5 * d + 4 + 4 * b + a ) * NewcombOperator(a,b,c-1,d-1)    
    for j in range(2,d+1):
        tot += 2 * (c-d+b) * (-1)**j * binom(3/2,j) * NewcombOperator(a,b,c-j,d-j)
    return tot / 4 / d

def HansenCoefficient_term(a,b,c,sigma):
    r"""
    Series coefficient in Taylor series
    of the Hansen coefficient X^{a,b}_c(e).
    The Hansen coefficient is given by:

    .. math::

        X^{a,b}_c(e) = e^{|c-b|} \times
        \sum_{\sigma=0}^\infty \mathrm{HansenCoefficient\_term(a,b,c,sigma)}e^{2\sigma}

    Arguments
    ---------
    a : int

    b : int
    
    c : int
    
    sigma : int

    Returns
    -------
    float
    """
    alpha = max(c-b,0)
    beta  = max(b-c,0)
    return NewcombOperator(a,b,alpha+sigma,beta+sigma)

def calX_term(a,b,c,d):
    r"""
    Series coefficient in Taylor series of the expession 
    calX^{a,b}_c(e) = (1-e^2)^{-1/2} * X^{a,b}_c(e) where
    X^{a,b}_c is a traditional Hansen coefficient 
    math::
        calX^{a,b}_c(e) = e^{|c-b|} \times
        \sum_{\sigma=0}^\infty calX_term(a,b,c,sigma)e^{2\sigma}

    Arguments
    ---------
    a : int
    
    b : int
    
    c : int
    
    d : int

    Returns
    -------
    float
    """
    tot = 0
    for n in xrange(d+1):
        tot += binom(-0.5,n) * (-1)**n * HansenCoefficient_term(a,b,c,d - n)
    return tot

def threeFtwo(a,b):
    """
    Hypergerometric 3_F_2([a1,a2,a3],[b1,b2],1)
    
    Used in calcluations of KaulaF function
    
    Arguments
    ---------
    a : list of ints
    
    b :  list of ints
    
    Returns
    -------
    float
    """
    a1,a2,a3 = a
    b1,b2 = b
    kmax = min(1-a1,1-a2,1-a3)
    tot = 0
    for k in range(0,kmax):
        tot += poch(a1,k) * poch(a2,k) *poch(a3,k) / poch(b1,k) / poch(b2,k) / factorial(k)
    return tot

def KaulaF(n,q,p,j):
    """
    Series coefficient in the Taylor expansion of the Kaula inclination
    function F_{nqp}(I).  See, e.g., `Kaula (1962,1966)`_ or `Ellis & Murray
    (2000)`_.

    The function returns the jth term of the Taylor
    expansion in the variable s = sin(I/2). I.e.
        KaulaF(n,q,p,j) = (1/j!) d^j F{nqp}/ ds^j

    This implementation is based on the Mathematica
    package by Fabio Zugno available at:
        https://library.wolfram.com/infocenter/MathSource/4256/

    .. _Kaula (1962,1966): https://ui.adsabs.harvard.edu/abs/1962AJ.....67..300K/abstract
    .. _Ellis & Murray (2000): https://ui.adsabs.harvard.edu/abs/2000Icar..147..129E/abstract
    Arguments
    ---------
    n : int

    q : int
    
    p : int
    
    j : int

    Returns
    -------
    float
    """
    if q==0 and 2 * p == n:
        return (-1)**(j+n) * binom(n,j) * binom(n+j,j) * factorial2(n-1) / factorial2(n)
    if n - 2*p - q < 0: 
        if n-q<0: return 0
        return (-1)**(n-q) * factorial(n+q) * KaulaF(n,-q,n-p,j) / factorial(n-q)
    numerator =  (-1)**j * factorial2(2*n-2*p-1)     
    numerator *= binom(n/2+p+q/2,j) 
    numerator *= threeFtwo([-j,-2*p,-n-q],[1+n-2*p-q,-(n/2)-p-q/2]) 
    if p<0: return 0.
    denom =  factorial(n -2 * p - q) * factorial2(2 * p)
    return numerator / denom 

def KK(i,n,m):
    numerator = (-1)**(i-n) * (1 + 2 * n) * gamma(1/2 + i) * gamma(3/2 + i)
    denom = 4 * gamma((2 + i-m-n)/2) * gamma((2 + i + m-n)/2) * gamma((3 + i - m + n)/2) * gamma((3 + i + m + n)/2)
    return numerator / denom

def getrange(lim1,lim2,n=1):
    assert n>0, "Negative interval n={} passed to getrange".format(n)
    if lim1 < lim2:
        return range(lim1,lim2 + n,n)
    else:
        return range(lim1,lim2-n,-n)
    
def FX(h,k,i,p,u,v1,v2,v3,v4,z1,z2,z3,z4):
    
    nlim1 = int(np.ceil(max(h,(h + abs(k)) / 2)))
    nlim2 = i
    hplusk_mod2 = (h+k) % 2
    delta = 1 - np.alltrue(np.array([h,k,v1-v2,v4-v3]) == 0)
    inc_total= 0
    for n in getrange(nlim1,nlim2,1):
        mlim1 = max(n-i,p-n+h,p-k-n+hplusk_mod2)
        mlim2 = min(i-n,p+n-h,n+p-k-hplusk_mod2)
        for m in getrange(mlim1,mlim2,2):
            term = (-1)**abs(k+m+n-p) * KK(i,n,m)
            term *= KaulaF(n,-k-m+p,(-h + m + n - p)//2,z1)
            term *= KaulaF(n, k+m-p,(-h - m + n + p)//2,z2)
            inc_total += term
    
    ecc_total = 0
    for t in range(0,u+1):
        term = (-1)**(u + t) / factorial(u-t) / factorial(t)
        term *= HansenCoefficient_term(i+t,v1,v2,z3)
        term *= HansenCoefficient_term(-1-i-t,v3,v4,z4)
        ecc_total += term
        
    return (1 + delta) * inc_total * ecc_total

    

def df_coefficient_Ctilde(k1,k2,k3,k4,k5,k6,nu1,nu2,nu3,nu4,include_indirect = True):
    r"""
    Get the coefficient of the disturbing function term:

    .. math::
        s_i^{|k_5|+2\nu_1} s_j^{|k_6|+2\nu_2}
        e_i^{|k_3|+2\nu_4} e_j^{|k_4|+2\nu_3} 
        \times
        \cos[
            k_1\lambda_j + k_2\lambda_i +
            k_3 \varpi_i + k_4 \varpi_j +
            k_5 \Omega_i + k_6 \Omega_j
        ]

    where the indices :math:`i` and :math:`j` corresponds to the inner and
    outer planets, respectively.  The result is returned as a dictionary of
    Laplace coefficient arguemnts and their numerical coefficents.

    Arguments:
    ----------
    k1 : int
        Coefficient of outer planet's mean longitude in cosine argument
    k2 : int
        Coefficient of inner planet's mean longitude in cosine argument
    k3 : int
        Coefficient of inner planet's mean longitude in cosine argument
    k4 : int
        Coefficient of outer planet's mean longitude in cosine argument
    k5 : int
        Coefficient of inner planet's longitude of ascending node in cosine argument
    k6 : int
        Coefficient of outer planet's longitude of ascending node in cosine argument
    nu1 : int
        Select specific term where the exponent of :math:`s_i` is :math:`|k_5|+2\nu_1`
    nu2 : int
        Select specific term where the exponent of :math:`s_j` is :math:`|k_6|+2\nu_2`
    nu3 : int
        Select specific term where the exponent of :math:`e_i` is :math:`|k_3|+2\nu_3`
    nu4 : int
        Select specific term where the exponent of :math:`e_j` is :math:`|k_4|+2\nu_4`
    include_indirect : booole, optional
        Whether to include the indirect contribution to the disturibing function
        coefficient.

    Returns
    -------
    dictionary
        The coefficient is given by the sum over laplace coefficients
        contained in the dictionary entries: 

        ..math::
            \sum A \times \alpha^p \frac{d^{n}}{d\alpha^{n}} b_{s}^{j}(\alpha)

        where the dictionary entries are in the form { (p,(s,j,n)) : A }
    """
    total = defaultdict(float)
    # must be even power in inclination
    if abs(k5 + k6) % 2:
        warnings.warn(
                "\n df_coefficient called with an argument not symmetric w.r.t. planet inclinations:\n" + 
                "\t (k1,k2,k3,k4,k5,k6)=({},{},{},{},{},{})".format(k1,k2,k3,k4,k5,k6)
                )
        return dict(total)
    # Sum of integer coefficients must be 0
    if k1 + k2 + k3 + k4 + k5 + k6:
        warnings.warn(
                "\n df_coefficient called with an argument that does not satisfy D'Alembert relation:\n" + 
                "\t (k1,k2,k3,k4,k5,k6)=({},{},{},{},{},{})".format(k1,k2,k3,k4,k5,k6)
                )
        return dict(total)
    kvec = np.array([k1,k2,k3,k4,k5,k6])
    if np.alltrue(kvec==0):
        for i in getrange(0,nu1+nu2,1):
            for p in getrange(i%2,i,2):
                for u in getrange(0,2*nu3+2*nu4):
                    cf = FX(0,0,i,p,u,0,0,0,0,nu1,nu2,nu3,nu4) * (1 + (p != 0))
                    if not np.isclose(cf,0):
                        total[(i+u,(i+1/2,abs(p),u))]+=cf
    elif np.alltrue(kvec[2:]==0):
        j = abs(k1)
        for i in getrange(0,nu1+nu2,1):
            for p in getrange(i%2,i,2):
                for u in getrange(0,2*nu3+2*nu4):
                    cf = FX(0,0,i,p,u,j,j,j,j,nu1,nu2,nu3,nu4) 
                    if not np.isclose(cf,0):
                        total[(i+u,(i+1/2,abs(j+p),u))]+=cf
                        if p != 0:
                            total[(i+u,(i+1/2,abs(j-p),u))]+=cf
    else:
        h = (-k5-k6)//2
        k = (k6-k5)//2
        n0 = int(np.ceil(max(h,-k5/2,-k6/2))) 
        for i in getrange(n0,nu1 + nu2 + (abs(k5)+abs(k6)//2),1):
            for p in getrange(h-i,i-h,2):
                for u in getrange(0,2*nu3 + 2*nu4 + abs(k3) + abs(k4),1):
                    cf = FX(h, k, i, p, u, k2 + k3, k2, k1 + k4, k1, nu1, nu2, nu3, nu4)
                    if not np.isclose(cf,0):
                        total[(i+u,(i+1/2,abs(k1+k4-h+p),u))]+=cf

    # add indirect term
    if include_indirect:
        coeff = df_coefficient_Ctilde_indirect_piece(k1,k2,k3,k4,k5,k6,nu1,nu2,nu3,nu4)
        total[('indirect',1)] = coeff
    return dict(total)

def _df_coeff_mult_by_alpha_power(coeffdict,r):
    """
    For a DF coefficient 'coeff' represented by 'coeffdict', 
    return the new coefficient dictionary  alpha^r * coeff
    """
    new = dict()
    for key,val in coeffdict.items():
        if key[0]=='indirect':
            newkey = (key[0],key[1]-2*r)
        else:
            p,sjn = key
            newkey = (p+r,sjn)
        new[newkey]=val
    return new

def _df_coefficient_scalar_mult(coeffdict,s):
    """
    Return dictionary for s * coeffdict for scalar value 's'
    """
    return {key:s*val for key,val in coeffdict.items()}

def _get_alpha_times_derivs_list(coeffdict, ltot):
    derivs_list=[coeffdict]
    dcoeff = coeffdict.copy()
    for l in range(ltot):
        dcoeff = deriv_df_coefficient(dcoeff)
        derivs_list.append(_df_coeff_mult_by_alpha_power(dcoeff,l+1))
    return derivs_list



def df_coefficient_C(k1,k2,k3,k4,k5,k6,nu1,nu2,nu3,nu4,l1=0,l2=0,include_indirect_terms = True):
    r"""
    Get the coefficient of the disturbing function term

    .. math ::

        |Y_i|^{|k_5|+2\nu_1}
        |Y_j|^{|k_6|+2\nu_2}
        |X_i|^{|k_3|+2\nu_3}
        |X_j|^{|k_4|+2\nu_4}
        \delta_i^{l_1}
        \delta_j^{l_2}
        \times \cos[
            k_1 \lambda_j + k_2 \lambda_i +
            k_3 \varpi_i + k_4 \varpi_j +
            k_5 \Omega_i + k_6 \Omega_j
        ]~,

    where the indices :math:`i` and :math:`j` corresponds to the
    inner and outer planets, respectively.
    The result it returned as a dictionary of Laplace coefficient
    arguemnts and their numerical coefficents.

    Arguments:
    ----------
    k1 : int
        Coefficient of outer planet's mean longitude in cosine argument
    k2 : int
        Coefficient of inner planet's mean longitude in cosine argument
    k3 : int
        Coefficient of inner planet's mean longitude in cosine argument
    k4 : int
        Coefficient of outer planet's mean longitude in cosine argument
    k5 : int
        Coefficient of inner planet's longitude of ascending node in cosine argument
    k6 : int
        Coefficient of outer planet's longitude of ascending node in cosine argument
    nu1 : int
        Select specific term where the exponent of :math:`|Y_i|` is :math:`|k_5|+2\nu_1`
    nu2 : int
        Select specific term where the exponent of :math:`|Y_j|` is :math:`|k_6|+2\nu_2`
    nu3 : int
        Select specific term where the exponent of :math:`|X_i|` is :math:`|k_3|+2\nu_3`
    nu4 : int
        Select specific term where the exponent of :math:`|X_j|` is :math:`|k_4|+2\nu_4`
    l1 : int, optional
        Select specifc term where exponent of 
        :math:`\delta_1 = (\Lambda_1 - \Lambda_{1,0})/\Lambda_{1,0})`
        is l1.
        Default value is l1 = 0
    l2 : int, optional
        Select specifc term where exponent of 
        :math:`\delta_2 = (\Lambda_2 - \Lambda_{2,0})/\Lambda_{2,0})`
        is l2.
        Default value is l2 = 0
    include_indirect_terms :  boole, optional
        whether the contribution of indirect terms should be
        accounted for when computing the coefficients. default
        is true.

    Returns
    -------
    dictionary
        The coefficient is given by the sum over laplace coefficients
        contained in the dictionary entries:

        .. math::
            \sum C \times \alpha^p \frac{d^{n}}{d\alpha^{n}} b_{s}^{j}(\alpha)

        where the dictionary entries are in the form { (p,(s,j,n)) : C }
    """
    Chat = defaultdict(float)
    msg = "Integer arguments nu_i and l_i must be non-negative."
    assert np.alltrue(np.array([nu1,nu2,nu3,nu4,l1,l2])>=0), msg
    for n3 in range(nu3+1):
        for n4 in range(nu4+1):
            term_dict = df_coefficient_Ctilde(k1,k2,k3,k4,k5,k6,nu1,nu2,n3,n4,include_indirect_terms)
            prefactor = Xi(nu3-n3,n3+abs(k3)/2,nu1+abs(k5)/2) * Xi(nu4-n4,n4+abs(k4)/2,nu2+abs(k6)/2)
            if prefactor != 0.:
                for key,val in term_dict.items():
                    Chat[key] += prefactor * val
    ltot = l1 + l2
    if ltot==0:
        result = Chat
    else:
        result = defaultdict(float)
        derivs_list = _get_alpha_times_derivs_list(Chat,ltot)
        l1fact_inv = 1/factorial(l1)
        l2fact_inv = 1/factorial(l2)
        prefactor = l1fact_inv * l2fact_inv
        p1,p2= _p1_p2_from_k_nu([k1,k2,k3,k4,k5,k6],[nu1,nu2,nu3,nu4])
        for m1 in range(l1+1):
            for r1 in range(l1-m1+1):
                for m2 in range(l2+1):
                    for r2 in range(l2-m2+1):
                        to_add = _df_coefficient_scalar_mult(derivs_list[r1+r2],prefactor * _Psi_coeff(l1,l2,p1,p2,m1,m2,r1,r2))
                        result = _add_dicts(result,to_add)      
    return result

def has_indirect_component(k1,k2,k3,k4,k5,k6):
    two_p = k2 + k4 + 1 
    two_p1 = -1 * (k1 + k3 - 1)
    m = k5 - two_p1 + 1
    m_is_zero_or_one = m == 0 or m == 1
    return is_zero_or_two(two_p) and is_zero_or_two(two_p1) and m_is_zero_or_one

def is_zero_or_two(n):
    return n == 0 or n == 2
    
def deriv_df_coefficient(coeff):
    r"""
    Derivative of a disturbing function coefficient with repsect to 
    alpha.

    Argument
    --------
    coeff : dict
        The coefficient is given by the sum over laplace coefficients
        contained in the dictionary entries:

            .. math::
            \sum C \times \alpha^p \frac{d^{n}}{d\alpha^{n}} b_{s}^{j}(\alpha)
        where the dictionary entries are in the form { (p,(s,j,n)) : C }
    
    Returns
    -------
    dict 
        A new dictionary in the same form as the input 'coeff' representing
        the derivative of coeff with respect to alpha.
    """
    dcoeff = defaultdict(float)
    for key,val in coeff.items():
        if key[0]=='indirect':
            # term \propto \alpha^{-p/2}
            pwer = key[1]
            dcoeff[('indirect',pwer + 2)] = -0.5 * pwer * val
        else:
            p,sjn= key
            s,j,n = sjn
            if p>0:
                dcoeff[(p-1,sjn)] += p * val
            dcoeff[(p,(s,j,n+1))] += val
    return dict(dcoeff)

def Xi(N,p,q):
    tot = 0
    for l in range(0,N+1):
        tot += binom(p,l) * negative_binom(-q,N-l) * (1/2)**l
    tot *= (-1/2)**N
    return tot

def negative_binom(minus_q,l):
    # scipy.special.binom returns a NaN when called at a
    # negative integer so I use this alternate formulation
    # when the argument is potenially a negative integer
    return (-1)**l * poch(-1 * minus_q,l) / factorial(l)


def _Cindirect_type1(k1,k2,m,z1,z2,z3,z4):
    r"""
    Indirect term cosine coefficient for argument:
         \theta = k_1\lambda_j + k_2\lambda_i
                    - (k_1-1)\varpi_j - (k_2+1) \varpi_i
                    + m(\Omega_i-\Omega_j)
    """
    if m > 2 or m < 0:
        return 0
    base_case = -1 * calX_term(0,1,-k1,z4) * calX_term(0,1,k2,z3)
    if m == 0:
        return binom(1,z1) * binom(1,z2) * (-1)**(z1) * (-1)**(z2) * base_case
    if m == 1:
         return 2 * binom(0.5 ,z1) * binom(0.5, z2 ) * (-1)**(z1) * (-1)**(z2) *\
        base_case
    if m == 2:
        return (1 - _delta(z1)) * (1 - _delta(z2)) * base_case

def _Cindirect_type2(k1,k2,m,z1,z2,z3,z4):
    r"""
    Indirect term cosine coefficient for argument:

        .. math::
        \theta = k_1\lambda_j + k_2\lambda_i
                    - (k_1-1)\varpi_j - (k_2-1) \varpi_i
                    - \Omega_i - \Omega_j + m(\Omega_j-\Omega_i)
    """
    if m > 2 or m <0:
        return 0
    base_case = -1 * calX_term(0,1,k1,z4) * calX_term(0,1,k2,z3)
    if m == 0:
        return binom(1 ,z1) * (1 - _delta(z2)) * (-1)**(z1) * base_case
    if m == 1:
        return -2 * binom(0.5 ,z1) * binom(0.5, z2 ) * (-1)**(z1) * (-1)**(z2) * base_case
    if m == 2:
        return binom(1 ,z2) * (1 - _delta(z1)) * (-1)**(z2) * base_case


def df_coefficient_Ctilde_indirect_piece(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4):
    r"""
    Get the indirect contribution to disturbing function coefficient

    .. math::
        \bar{C}_{\pmb{k}}^{(z_1,z_2,z_3,z_4)}
    """
    if np.sum([k1,k2,k3,k4,k5,k6]) !=0:
        warnings.warn(
        "\n df_coefficient called with an argument that does not satisfy D'Alembert relation:\n" +
        "\t (k1,k2,k3,k4,k5,k6)=({},{},{},{},{},{})".format(k1,k2,k3,k4,k5,k6)
        )
        return 0
    if k1 == 0 or k2 == 0:
        return 0
    s1 = k1 + k4
    s2 = k2 + k3
    if abs(s1) == 1 and abs(s2)==1:
        if s1 == -1*s2:
            return _Cindirect_type1(s2 * k1,s2 * k2,s2 * k6,z1,z2,z3,z4)
        else:
            return _Cindirect_type2(s1 * k1,s1 * k2,-1 * s1 * k5,z1,z2,z3,z4)
    else:
        return 0

# SH --- Needs modified to accomodate expansion in delta.
def terms_list_to_HamiltonianCoefficients_dict(terms_list,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out,include_alpha_derivs=False):
    """
    Retrieve the a dictionary with the coefficient values of terms appearing in the Hamiltonian.
    
    Arguments
    ---------
    terms_list : list
      List of terms to copmute coefficients for. List entries
      should be in the form (kvec, zvec) where kvec and zvec
      are tuples of integers.
    G : float
      Value of gravitational constant.
    mIn : float
      Inner planet mass parameter
    mOut : float
      Outer planet mass parameter
    MIn : float
      Inner planet stellar mass parameter
    MOut : float
      Outer planet stellar mass parameter
    Lambda0In : float
      Lambda of inner planet for evaluating coefficient.
    Lambda0Out : float
      Lambda of outer planet for evaluating coefficient.
    include_alpha_derivs : bool, optional
      If True, dictionary values are tuple containing
      both the coefficient values and their derivatives
      with respect to alpha = aIn/aOut.
    
    Returns
    -------
    coeff_dictionary : dict
      Coefficients given in dictionary where entries are given in the form
        {(kvec,zvec):Coeff}
      or, if include_alpha_derivs=True, in the form
        {(kvec,zvec):(Coeff,dCoeff/dalpha}
      where 'Coeff' represents the coefficient the term
        (1/2) * Coeff * \exp[i * (k[0] *\lambda_{out} + k[1] *\lambda_{in})] * 
          X_{in}^{|kvec[2]| + zvec[2]} * 
          X_{out}^{|kvec[3]| + zvec[3]} * 
          Y_{in}^{|kvec[4]| + zvec[0]} * 
          Y_{out}^{|kvec[5]| + zvec[1]}
          +
          complex conjugate
      appearing in the interaction Hamiltonian between a pair of planets.
      
    """
    muIn = mIn * (MIn - mIn) / MIn
    muOut = mOut * (MOut - mOut) / MOut
    aIn0 = (Lambda0In / muIn)**2 / MIn / G
    aOut0 = (Lambda0Out / muOut)**2 / MOut / G
    alpha = aIn0 / aOut0
    assert alpha < 1, "Particles are not in order by semi-major axis."
    aOut_inv = G*MOut*muOut*muOut / Lambda0Out / Lambda0Out  
    prefactor = -G * mIn * mOut * aOut_inv
    result = dict()
    for kvec,zvec in terms_list:
        C = df_coefficient_C(*kvec,*zvec)
        coeff = prefactor * evaluate_df_coefficient_dict(C,alpha)
        if include_alpha_derivs:
            ind = C.pop(('indirect',1))
            dC = deriv_df_coefficient(C)
            dcoeff = prefactor * ( evaluate_df_coefficient_dict(dC,alpha) - 0.5 * ind / np.sqrt(alpha*alpha*alpha))
            result[(kvec,zvec)] = coeff,dcoeff
        else:
            result[(kvec,zvec)] = coeff
    return result

def kz_to_xx1yy1_powers(kz):
    """
    Convert (kvec,zvec) Poisson series term designation 
    to the integer powers of variables x_{in},x_{out},y{in}, and y_{out}
    and their complex conjugates.

    Arguments
    ---------
    kz : tuple
      tuple (kvec,zvec) designating the Poisson series term.

    Returns
    -------
    pows : array
      Poisson series powers
      x_{in}^{pows[0]} * 
      x_{out}^{pows[1]} *
      y_{in}^{pows[2]} * 
      y_{out}^{pows[3]} 
      xbar_{in}^{pows[4]} * 
      xbar_{out}^{pows[5]} *
      ybar_{in}^{pows[6]} * 
      ybar_{out}^{pows[7]} 
    """
    k,z = kz
    zpow = lambda x: max(x,0)
    zbarpow = lambda x: max(-x,0)
    xx1yy1_pows = [
        z[2] + zpow(k[2]),
        z[3] + zpow(k[3]),
        z[0] + zpow(k[4]),
        z[1] + zpow(k[5])
    ]
    xx1yy1_bar_pows = [
        z[2] + zbarpow(k[2]),
        z[3] + zbarpow(k[3]),
        z[0] + zbarpow(k[4]),
        z[1] + zbarpow(k[5])
    ]
    return np.array(xx1yy1_pows + xx1yy1_bar_pows,dtype=np.int64)

def xx1yy1_powers_to_kz(xx1yy1_powers,k1=0,k2=0):
    """
    Inverse of 'kz_to_xx1yy1_powers'
    """
    pows = xx1yy1_powers[:4]
    bar_pows = xx1yy1_powers[4:]
    zpows = np.min(np.vstack((pows,bar_pows)),axis=0)
    kvec=np.concatenate(([k1,k2],pows - bar_pows))
    zvec = np.concatenate((zpows[2:],zpows[:2]))
    return tuple(kvec),tuple(zvec)

def xx1yy1_powers_conj(xx1yy1_powers):
    return np.concatenate([xx1yy1_powers[-4:],xx1yy1_powers[:4]])

def _add_ppbar_bracket_terms(coeff1,k1z1,coeff2,k2z2,results,LambdaIn,LambdaOut):
    pows1 = kz_to_xx1yy1_powers(k1z1)
    pows2 = kz_to_xx1yy1_powers(k2z2)
    pows1conj = xx1yy1_powers_conj(pows1)
    pows2conj = xx1yy1_powers_conj(pows2)
    LmbdaInv_factors = np.array((2/LambdaIn,2/LambdaOut,0.5/LambdaIn,0.5/LambdaOut))
    p1conjp2 = pows1conj + pows2
    for i,p1bar,p2 in zip(range(4),pows1conj[:4],pows2[4:]):
        if p1bar > 0 and p2 > 0:
            v = p1conjp2.copy()
            v[i]-=1
            v[4+i]-=1
            coeff = p1bar * p2 * coeff1 * coeff2 * LmbdaInv_factors[i]
            kznew = xx1yy1_powers_to_kz(v)
            results[kznew] -= coeff
    p1p2conj = pows1 + pows2conj
    for i,p1,p2bar in zip(range(4),pows1[:4],pows2conj[4:]):
        if p1 > 0 and p2bar > 0:
            v = p1p2conj.copy()
            v[i]-=1
            v[4+i]-=1
            coeff = p1 * p2bar * coeff1 * coeff2 * LmbdaInv_factors[i]
            kznew = xx1yy1_powers_to_kz(v)
            results[kznew] += coeff

def _consolidate_dictionary_terms(d):
    """
    Combine all terms (kvec,zvec) that are conjugates of one another
    into single terms with a positive coefficient for 
    pomega_in (i.e., k[2]) or, if k[2]=0, Omega_in (k[4]), or, if k[4]=0,
    pomega_out k[3]
    """
    dnew = defaultdict(int)
    for kz, val in d.items():
        k,z = kz
        k = np.array(k,dtype=np.int64)
        if k[2] != 0:
           k *= np.sign(k[2])
        elif k[4] !=0:
           k *= np.sign(k[4]) 
        elif k[3] !=0:
           k *= np.sign(k[3]) 
        else: 
            assert np.alltrue(k==0) and not np.alltrue(np.array(z)==0), "Suspected bad k,z={},{} pair in term dictionary!".format(k,z)
        dnew[(tuple(k),z)] += val
    return dict(dnew)

def resonant_terms_list_to_secular_contribution_dictionary(terms_list,j,k,Nmin,Nmax,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    """
    Generate a dictionary containing the secular terms generated by a list of resonant terms.
    The secular contributions arise at second order in planet masses.

    Arguments
    ---------
    terms_list : list
      List of resonance terms in (kvec,zvec) form.
    j : int
      Specifies resonance as the j:j-k MMR.
    k : int
      Order of MMR.
    G : float
      Value of gravitational constant.
    mIn : float
      Inner planet mass parameter
    mOut : float
      Outer planet mass parameter
    MIn : float
      Inner planet stellar mass parameter
    MOut : float
      Outer planet stellar mass parameter
    Lambda0In : float
      Lambda of inner planet for evaluating coefficient.
    Lambda0Out : float
      Lambda of outer planet for evaluating coefficient.
    Nmin : int
      Minimum power (in inclination/eccentricity) of terms to include in result.
    Nmax : int
      Maximum power (in inclination/eccentricity) of terms to include in result.

    Returns
    -------
    result : dict
      A dictionary containing the secular terms with 
      entries in the form
       {(kvec,zvec):Coeff}
    """
    muIn = mIn * (MIn - mIn) / MIn
    muOut = mOut * (MOut - mOut) / MOut
    omega_vec = G * G * np.array([
        MIn * MIn * muIn * muIn * muIn / Lambda0In / Lambda0In / Lambda0In,
        MOut * MOut * muOut * muOut * muOut / Lambda0Out / Lambda0Out / Lambda0Out
    ])
    Domega = np.diag(-3 * omega_vec / np.array([Lambda0In,Lambda0Out]))
    res_kvec = np.array([k-j,j])
    res_omega = res_kvec @ omega_vec
    res_factor = 0.25 * (res_kvec @ Domega @ res_kvec) / res_omega / res_omega
    p = terms_list_to_HamiltonianCoefficients_dict(terms_list,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out,include_alpha_derivs=True)
    aIn0 = (Lambda0In / muIn)**2 / MIn / G
    aOut0 = (Lambda0Out / muOut)**2 / MOut / G
    alpha = aIn0 / aOut0
    dalpha_dLambdaIn = 2 * alpha / Lambda0In
    dalpha_dLambdaOut = -2 * alpha / Lambda0Out
    result = defaultdict(int)
    for kz1,coeffs1 in p.items():
        coeff1,dcoeff1 = coeffs1
        for kz2, coeffs2 in p.items():
            coeff2,dcoeff2 = coeffs2
            pows1 = kz_to_xx1yy1_powers(kz1)
            pows2 = kz_to_xx1yy1_powers(kz2)
            pows2conj = xx1yy1_powers_conj(pows2)
            powsnew = pows1 + pows2conj
            tot_pow = np.sum(powsnew)
            if Nmin<=tot_pow<=Nmax:
                rtLambda_inv_pow_In = powsnew[0] + powsnew[2]
                rtLambda_inv_pow_Out = powsnew[1] + powsnew[3]
                newk,newz = xx1yy1_powers_to_kz(powsnew)
                # term 1 part
                val = res_factor*coeff1*coeff2
                # term 2 part
                d_ppbar_dLambdaIn =  (dcoeff1 * coeff2 + coeff1 * dcoeff2) * dalpha_dLambdaIn  - 0.5 * rtLambda_inv_pow_In  * coeff1 * coeff2 / Lambda0In 
                d_ppbar_dLambdaOut = (dcoeff1 * coeff2 + coeff1 * dcoeff2) * dalpha_dLambdaOut - 0.5 * rtLambda_inv_pow_Out * coeff1 * coeff2 / Lambda0Out
                val -= 0.25 * res_kvec @ np.array([d_ppbar_dLambdaIn,d_ppbar_dLambdaOut]) / res_omega
                result[(newk,newz)] += val
            if Nmin<=tot_pow-2<=Nmax:
                _add_ppbar_bracket_terms((+0.25/res_omega) * coeff1,kz1,coeff2,kz2,result,Lambda0In,Lambda0Out)
    return _consolidate_dictionary_terms(result)

def _add_dicts(*dicts):
    keys = set().union(*[d.keys() for d in dicts])
    dsum = {k:np.sum([d.get(k,0) for d in dicts]) for k in keys}
    return dsum


def _resonance_arguments_of_fixed_order(j,k,N):
    args_dict = df_arguments_dictionary(N)
    args = []
    for N1 in range(k,N+1,2):
        if (N-N1) % 2:
            continue
        nutot = (N-N1)//2
        for arg in args_dict[N1][k]:
            for nuc in _nucombos(nutot):
                js = (j,k - j,*arg)
                args.append((js,nuc))
    return args

def resonant_secular_contribution_dictionary(j,k,Nmin,Nmax,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    """
    Generate a dictionary containing the secular terms with order between Nmin and 
    Nmax generated by a j:j-k MMR (including its harmonics).

    The secular contributions arise at second order in planet masses.

    Arguments
    ---------
    j : int
      Specifies resonance as the j:j-k MMR.
    k : int
      Order of MMR.
    G : float
      Value of gravitational constant.
    mIn : float
      Inner planet mass parameter
    mOut : float
      Outer planet mass parameter
    MIn : float
      Inner planet stellar mass parameter
    MOut : float
      Outer planet stellar mass parameter
    Lambda0In : float
      Lambda of inner planet for evaluating coefficient.
    Lambda0Out : float
      Lambda of outer planet for evaluating coefficient.
    Nmin : int
      Minimum power (in inclination/eccentricity) of terms to include in result.
    Nmax : int
      Maximum power (in inclination/eccentricity) of terms to include in result.

    Returns
    -------
    result : dict
      A dictionary containing the secular terms with 
      entries in the form
       {(kvec,zvec):Coeff}
    """

    extra_args = (G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out)
    # ensure even orders
    Nmin = 2 * (Nmin//2)
    Nmax = 2 * (Nmax//2)
    all_dicts = []
    
    # highest harmonic of j:j-k resonance to include
    nmax = (Nmax + 2) // (2*k)  # Complete secular contribution at desired order
    # loop over harmonics
    for n in range(1,nmax+1):
        j1 = n * j
        k1 = n * k
        res_args = []
    
        # With resonant coefficient p(z,zbar)
        # expanded to order e^(k1 + 2* Mmax + 2), 
        # maximum order secular terms are: 
        #  D_e[e^k1] * D_e [e^(k1 + 2 * Mmax + 2)] = e^Nmax
        # These come from the  ~{pbar,p} / omega_res
        # term in the transformed Hamiltonian so they're
        # subdominant but we'll add them for completeness.
        Mmax = 1 + (Nmax - 2 * k1)//2
        # Mmax = (Nmax - 2 * k1)//2
        for M in range(0,Mmax+1):
            # 
            res_args += _resonance_arguments_of_fixed_order(j1,k1,k1 + 2 * M)
        dres = resonant_terms_list_to_secular_contribution_dictionary(
            res_args,
            j1,
            k1,Nmin,Nmax,
            *extra_args
        )
        all_dicts.append(dres)
    
    return _add_dicts(*all_dicts)
