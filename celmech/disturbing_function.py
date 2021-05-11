from sympy import S, diff, lambdify, symbols, sqrt, cos,sin, numbered_symbols, simplify,binomial, hyper, hyperexpand, Function, factorial,elliptic_k,elliptic_e, expand_trig
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

def get_DFCoeff_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4,indexIn,indexOut):
    return symbols("C_{0}\,{1}\,{2}\,{3}\,{4}\,{5}^{6}\,{7}\,{8}\,{9};({10}\,{11})".format(
        k1,k2,k3,k4,k5,k6,z1,z2,z3,z4,indexIn,indexOut)
    )

def _delta(*args):
    intarr  = np.array([args],dtype=np.int64)
    if np.alltrue(intarr == 0):
        return 0
    else:
        return 1
    
def DFArguments_dictionary(Nmax):
    """
    Get the arguments appearing in the disturbing function
    up to order Nmax.  
    
    Arguments are returned as a nested dictionary. 
    The outer level keys are orders. The inner level
    keys are dj = |j1-j2|, denoting the order of MMR
    for which the argument appears. The values of the 
    inner level dictionaries are lists of 
    tuples containing (j3,j4,j5,j6). 
    
    {
        0:{0:[(0,0,0,0)],
        1:{1:[(-1,0,0,0),(0,-1,0,0)]},
        ...
        Nmax:{
            0:[(j3,j4,j5,j6)],
            ...
            Nmax:[(j3,j4,j5,j6),...]
        }
    }
    
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
                    sgn = 1 if dj is 0 else np.sign(dj)
                    j3=-sgn*s
                    j4=-sgn*s1
                    j5=-sgn*(hplusk)
                    j6=-sgn*(hminusk)
                    N = abs(j3) + abs(j4) + abs(j5) + abs(j6)
                    args_dict[N][dj*sgn].append((j3,j4,j5,j6))
    return args_dict

def _zcombos_iter(ztot):
    for z1 in range(ztot+1):
        for z2 in range(ztot+1-z1):
            for z3 in range(ztot+1-z1-z2):
                z4 = ztot - z1 - z2 - z3
                yield (z1,z2,z3,z4)
                
def ResonanceTermsList(j,k,Nmin,Nmax):
    """
    Generate the list of disturbing function terms for a 
    j:j-k resonance with eccentricity/inclination order
    between Nmin and Nmax.

    Arguments
    ---------
    j : int
     Determines resonance
    k : int
     Order of the resonance
    Nmin : int
     Minimum order of terms to include
    Nmax : int
     Maximum order of terms to include

    Returns
    -------
    term : list
        A list of disturbing function terms. 
        Each entry in the list is of the form
        (kvec, zvc)
    """
    args_dict = DFArguments_dictionary(Nmax)
    args = []
    for N in range(Nmin,Nmax+1):
        for k1 in range(k,N+1,k):
            if (N-k1) % 2:
                continue
            j1 = (k1//k) * j
            for N1 in range(k1,N+1,2):
                ztot = (N-N1)//2
                for arg in args_dict[N1][k1]:
                    for zc in _zcombos_iter(ztot):
                        js = (j1,k1 - j1,*arg)
                        args.append((js,zc))
    return args

def SecularTermsList(Nmin,Nmax):
    """
    Generate the list of secular disturbing function terms 
    with eccentricity/inclination order between Nmin and Nmax.

    Arguments
    ---------
    j : int
     Determines resonance
    k : int
     Order of the resonance
    Nmin : int
     Minimum order of terms to include
    Nmax : int
     Maximum order of terms to include

    Returns
    -------
    term : list
        A list of disturbing function terms. 
        Each entry in the list is of the form
        (kvec, zvc)
    """
    args_dict = DFArguments_dictionary(Nmax)
    args = []
    Nmax1 = (Nmax//2) * 2 
    Nmin1 = (Nmin//2) * 2 
    for N in range(0,Nmax1 + 1,2):
        argsN = args_dict[N][0]
        ztot_min = max( (Nmin1 - N)//2 , 0)
        ztot_max = (Nmax1 - N)//2 
        for ztot in range(ztot_min,ztot_max + 1):
            for zc in _zcombos_iter(ztot):
                for arg in argsN:
                    js = (0,0,*arg)
                    args.append((js,zc))
    return args

def laplace_b(s,j,n,alpha):
    """
    Calculates nth derivative with respect to a (alpha) of Laplace coefficient b_s^j(a).
    Uses recursion and scipy special functions. 
    
    Arguments
    ---------
    s : float 
        half-integer parameter of Laplace coefficient. 
    j : int 
        integer parameter of Laplace coefficient. 
    n : int 
        return nth derivative with respect to a of b_s^j(a)
    a : float
        semimajor axis ratio a1/a2 (alpha)
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

def eval_DFCoeff_dict(Coeff_dict,alpha):
    r"""
    Evaluate a dictionary representing a sum
    of Laplace coefficient terms like those returned
    by DFCoeff_C and DFCoeff_Cbar evaluated at a 
    specific value of semi-major axis ratio, alpha.

    Arguments
    ---------
    Coeff_dict : dictionary
        Dictionary with entries {(p,(s,j,n)) : coeff}
        representing a sum of Laplace coefficients:

        .. math::
         \mathrm{coeff}  \alpha^p \frac{ d^n b_s^{(j)}(\alpha)} { d\alpha^n}
    alpha : float
        Value of semi-major axis ratio a1/a2 appearing
        as an argument of Laplace coefficients.

    Returns
    -------
    float : 
        The sum of Laplace coefficeint terms represented
        by dictionary entries.
    """
    tot = 0
    for key,val in Coeff_dict.items():
        if key is 'indirect':
            tot += val / np.sqrt(alpha)
        else:
            p,arg = key
            tot += val * alpha**p * laplace_b(*arg,alpha)
    return tot

def eccentricity_type_resonance_coefficient(j,k,l,alpha):
    r"""
    Get the coefficient of the distrubing function term:
        e_1^{l}e_2^{k-l}\cos[j \lambda_2 - (j-k) \lambda_1 - l\varpi_1 - (k-l)\varpi_2]
    that appears as the leading-order term of a kth order eccentricity resonance.

    Arguments
    ---------
    j : int
        Specifies the resonance term
    k : int
        Order of the resonance
    l : int
        Specify the e_1^{l}e_2^{k-l} sub-resonance
    alpha : float
        Semi-major axis ratio a_1/a_2

    Returns
    -------
    val : float
        Coefficient's numerical value
    """
    if l >  k:
        raise ValueError("Integer arguemnt l={} cannot be greater than the resonance order k={}".format(l,k))
    j1 = j
    j2 = k - j
    j3 = -l
    j4 = l-k
    j5 = 0
    j6 = 0
    z1=z2=z3=z4=0
    coeff = DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
    ncoeff = eval_DFCoeff_dict(coeff,alpha)
    return ncoeff

# Vector of resonance coefficients
def get_res_coeff_vector(j,k):
    r"""
    Get a vector comprised of all sub-resonance coefficients for the j:j-k mean motion resonance.

    Arguments
    ---------
    j : int
        Specify the j:j-k resonance
    k : int
        Order of the resonance
    include_indirect_terms :  boole, optional
        Whether the contribution of indirect terms should be
        accounted for when computing the coefficients. Default
        is True.

    Returns
    -------
    vals : ndarray
        Array of coefficient values from l=0 to l=k
    """ 
    res_pratio = float(j - k) /float(j)
    alpha = res_pratio**(2./3.)
    Cjkl = eccentricity_type_resonance_coefficient
    vals = np.array([Cjkl(j,k,l,alpha) for l in range(k+1)],dtype=np.float64)
    return vals

def get_fg_coeffs(res_j,res_k):
    """Get 'f' and 'g' coefficients for approximating the disturbing function coefficients associated with an MMR."""
    res_pratio = float(res_j - res_k) /float(res_j)
    alpha = res_pratio**(2./3.)
    vec = get_res_coeff_vector(res_j,res_k)
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
    Series coefficient in the Taylor expansion of the
    Kaula inclination function F_{nqp}(I).
    See, e.g., Kaula (1962,1966) or Ellis & Murray (2000).

    The function returns the jth term of the Taylor
    expansion in the variable s = sin(I/2). I.e.
        KaulaF(n,q,p,j) = (1/j!) d^j F{nqp}/ ds^j

    This implementation is based on the Mathematica
    package by Fabio Zugno available at:
        https://library.wolfram.com/infocenter/MathSource/4256/

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

    

def DFCoeff_Cbar(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4,include_indirect = True):
    r"""
    Get the coefficient of the disturbing function term:
    
      s1^{|j5|+2z1} s2^{|j6|+2z2} * e2^{|j4|+2*z4} * e1^{|j3|+2*z3} \times 
          cos[j1*L2 + j2*L1 + j3 * pomega1 + j4 * w2 + j5 * Omega1 + j6 * Omega2)

    where s1 = sin(I1/2) and s2 = sin(I2/2) as a dictionary of Laplace coefficient 
    arguemnts and their numerical coefficents.
    
    Arguments:
    ----------
    j1 : int
        Coefficient of outer planet's mean longitude in cosine argument
    j2 : int
        Coefficient of inner planet's mean longitude in cosine argument
    j3 : int
        Coefficient of inner planet's mean longitude in cosine argument
    j4 : int
        Coefficient of outer planet's mean longitude in cosine argument
    j5 : int
        Coefficient of inner planet's longitude of ascending node in cosine argument
    j6 : int
        Coefficient of outer planet's longitude of ascending node in cosine argument
    z1 : int
        Select specific term where the exponent of s1 is |j5|+2*z1
    z2 : int
        Select specific term where the exponent of s2 is |j6|+2*z3
    z3 : int
        Select specific term where the exponent of e1 is |j3|+2*z3
    z4 : int
        Select specific term where the exponent of e1 is |j4|+2*z4
    include_indirect : booole, optional
        Whether to include the indirect contribution to the disturibing function
        coefficient.
        
    Returns
    -------
    dictionary 
        The coefficient is given by the sum over laplace coefficients
        contained in the dictionary entries: 

        ..math::
            \sum C \times \alpha^p \frac{d^{n}}{d\alpha^{n}} b_{s}^{j}(\alpha)
        where the dictionary entries are in the form { (p,(s,j,n)) : C }
    """
    total = defaultdict(float)
    # must be even power in inclination
    if abs(j5 + j6) % 2:
        warnings.warn(
                "\n DFCoeff called with an argument not symmetric w.r.t. planet inclinations:\n" + 
                "\t (j1,j2,j3,j4,j5,j6)=({},{},{},{},{},{})".format(j1,j2,j3,j4,j5,j6)
                )
        return dict(total)
    # Sum of integer coefficients must be 0
    if j1 + j2 + j3 + j4 + j5 + j6:
        warnings.warn(
                "\n DFCoeff called with an argument that does not satisfy D'Alembert relation:\n" + 
                "\t (j1,j2,j3,j4,j5,j6)=({},{},{},{},{},{})".format(j1,j2,j3,j4,j5,j6)
                )
        return dict(total)
    jvec = np.array([j1,j2,j3,j4,j5,j6])
    if np.alltrue(jvec==0):
        for i in getrange(0,z1+z2,1):
            for p in getrange(i%2,i,2):
                for u in getrange(0,2*z3+2*z4):
                    cf = FX(0,0,i,p,u,0,0,0,0,z1,z2,z3,z4) * (1 + (p != 0))
                    if not np.isclose(cf,0):
                        total[(i+u,(i+1/2,abs(p),u))]+=cf
    elif np.alltrue(jvec[2:]==0):
        j = abs(j1)
        for i in getrange(0,z1+z2,1):
            for p in getrange(i%2,i,2):
                for u in getrange(0,2*z3+2*z4):
                    cf = FX(0,0,i,p,u,j,j,j,j,z1,z2,z3,z4) 
                    if not np.isclose(cf,0):
                        total[(i+u,(i+1/2,abs(j+p),u))]+=cf
                        if p != 0:
                            total[(i+u,(i+1/2,abs(j-p),u))]+=cf
    else:
        h = (-j5-j6)//2
        k = (j6-j5)//2
        n0 = int(np.ceil(max(h,-j5/2,-j6/2))) 
        for i in getrange(n0,z1 + z2 + (abs(j5)+abs(j6)//2),1):
            for p in getrange(h-i,i-h,2):
                for u in getrange(0,2*z3 + 2*z4 + abs(j3) + abs(j4),1):
                    cf = FX(h, k, i, p, u, j2 + j3, j2, j1 + j4, j1, z1, z2, z3, z4)
                    if not np.isclose(cf,0):
                        total[(i+u,(i+1/2,abs(j1+j4-h+p),u))]+=cf

    # add indirect term
    if include_indirect:
        total['indirect'] = DFCoeff_Cbar_indirect_piece(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4)
    return dict(total)

def DFCoeff_C(j1,j2,j3,j4,j5,j6,N1,N2,N3,N4):
    r"""
    Get the coefficient of the disturbing function term:

    Y1^{|j5|+2*N1} * Y2^{|j6|+2*N2} * X1^{|j3|+2*N3} * X2^{|j4|+2*N4}
     *cos[j1*L2 + j2*L1 + j3 * pomega1 + j4 * w2 + j5 * Omega1 + j6 * Omega2)

    as a dictionary of Laplace coefficient
    arguemnts and their numerical coefficents.

    Arguments:
    ----------
    j1 : int
        Coefficient of outer planet's mean longitude in cosine argument
    j2 : int
        Coefficient of inner planet's mean longitude in cosine argument
    j3 : int
        Coefficient of inner planet's mean longitude in cosine argument
    j4 : int
        Coefficient of outer planet's mean longitude in cosine argument
    j5 : int
        Coefficient of inner planet's longitude of ascending node in cosine argument
    j6 : int
        Coefficient of outer planet's longitude of ascending node in cosine argument
    N1 : int
        Select specific term where the exponent of Y1 is |j5|+2*N1
    N2 : int
        Select specific term where the exponent of Y2 is |j6|+2*N3
    N3 : int
        Select specific term where the exponent of X1 is |j3|+2*N3
    N4 : int
        Select specific term where the exponent of X2 is |j4|+2*N4

    Returns
    -------
    dictionary
        The coefficient is given by the sum over laplace coefficients
        contained in the dictionary entries:

        .. math::
            \sum C \times \alpha^p \frac{d^{n}}{d\alpha^{n}} b_{s}^{j}(\alpha)

        where the dictionary entries are in the form { (p,(s,j,n)) : C }
    """
    terms_total = defaultdict(float)
    for n3 in range(N3+1):
        for n4 in range(N4+1):
            term_dict = DFCoeff_Cbar(j1,j2,j3,j4,j5,j6,N1,N2,n3,n4)
            prefactor = Xi(N3-n3,n3+abs(j3)/2,N1+abs(j5)/2) * Xi(N4-n4,n4+abs(j4)/2,N2+abs(j6)/2)
            if prefactor != 0.:
                for key,val in term_dict.items():
                    terms_total[key] += prefactor * val
    return dict(terms_total)

def has_indirect_component(j1,j2,j3,j4,j5,j6):
    two_p = j2 + j4 + 1 
    two_p1 = -1 * (j1 + j3 - 1)
    m = j5 - two_p1 + 1
    m_is_zero_or_one = m == 0 or m == 1
    return is_zero_or_two(two_p) and is_zero_or_two(two_p1) and m_is_zero_or_one

def is_zero_or_two(n):
    return n == 0 or n == 2
    
def deriv_DFCoeff(coeff):
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
        if key=='indirect':
            # Fix!
            continue
        p,sjn= key
        s,j,n = sjn
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
    """
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


def DFCoeff_Cbar_indirect_piece(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4):
    r"""
    Get the indirect contribution to disturbing function coefficient

    .. math::
        \bar{C}_{\pmb{k}}^{(z_1,z_2,z_3,z_4)}
    """
    if np.sum([k1,k2,k3,k4,k5,k6]) !=0:
        warnings.warn(
        "\n DFCoeff called with an argument that does not satisfy D'Alembert relation:\n" +
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
        C = DFCoeff_C(*kvec,*zvec)
        coeff = prefactor * eval_DFCoeff_dict(C,alpha)
        if include_alpha_derivs:
            ind = C.pop('indirect')
            dC = deriv_DFCoeff(C)
            dcoeff = prefactor * ( eval_DFCoeff_dict(dC,alpha) - 0.5 * ind / np.sqrt(alpha*alpha*alpha))
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
    args_dict = DFArguments_dictionary(N)
    args = []
    for N1 in range(k,N+1,2):
        if (N-N1) % 2:
            continue
        ztot = (N-N1)//2
        for arg in args_dict[N1][k]:
            for zc in _zcombos_iter(ztot):
                js = (j,k - j,*arg)
                args.append((js,zc))
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
