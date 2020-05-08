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



def laplace_coefficient(s,j,n,a):
    """
    Calculates alpha^n times the nth derivative with respect to a (alpha) of 
    Laplace coefficient b_s^j(a). Code due to Jack Wisdom. Changed notation 
    to match Murray & Dermott (Eq. 6.67)
    
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
    clibcelmech.laplace.restype = c_double
    return clibcelmech.laplace(c_double(s), c_int(j), c_int(n), c_double(a))

def laplace_b(s,j,n,alpha):
    """
    Calculates nth derivative with respect to a (alpha) of Laplace coefficient b_s^j(a).
    Uses recursion and scipy special functions instead of C code unlike
    'laplace_coefficient' function.
    
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
         coeff * \alpha^p * d^n b_s^{(j)}(\alpha) / d\alpha^n
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
        p,arg = key
        tot += val * alpha**p * laplace_b(*arg,alpha)
    return tot

def general_order_coefficient(res_j, order, epower, a):
    clibcelmech.GeneralOrderCoefficient.restype = c_double
    return clibcelmech.GeneralOrderCoefficient(c_int(res_j), c_int(order), c_int(epower), c_double(a))

# Vector of resonance coefficients
def get_res_coeff_vector(j,k,include_indirect_terms = True):
    """Returns a vector comprised of all sub-resonance coefficients for the j:j-k mean motion resonance""" 
    res_pratio = float(j - k) /float(j)
    alpha = res_pratio**(2./3.)
    Cjkl = general_order_coefficient
    vals = np.array([Cjkl(j,k,l,alpha) for l in range(k+1)],dtype=np.float64)
    if j==k + 1 and include_indirect_terms:
        correction = Nto1_indirect_term_correction(j)
        vals[0] += correction
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

def Xlm0(l,m,e):
    """ 
        Get the closed-form expression for the Hansen coefficient X^(l,m)_0
        which appears in the secular component of the disturbing function (e.g., Mardling 2013) 
    """
    a =  (m-l-1) / S(2)
    b = a + 1/ S(2)
    c = m + 1
    fn = hyperexpand(hyper([a,b],[c],e*e))
    return (-e / 2)**m * binomial(l+m+1,m) * fn

def Nto1_indirect_term_correction(N):
    r"""
    Get the correction to the DF coefficient of an
    N:1 resonance that comes from the indirect term
    to leading order in eccentricity.

    The corrections applies to the coefficient associated
    with the argument
        $N\lambda_2 - \lambda_1 - (N-1)\varpi_2$
    and is computed by means of expanding the expression
    for the Hansen coefficient.

    Arguemnts
    ---------
    N : int
        Integer denoting the specific N:1 MMR to compute
        the correction for.

    Returns
    -------
    correction_term : float
        Correction term to add to Cjkl(N,N-1,0,alpha)
    """
    hard_coded_coeffs = [2,27/8,16/3,3125/284] 
    assert N>1,"Indirect terms not implemented for {}:1 resonance!".format(N)
    if N<len(hard_coded_coeffs) + 2:
        coeff = hard_coded_coeffs[N-2]
    else:
        u,e,x=symbols('u,e,x')
        expif = cos(u)-e + I * sqrt(1-e*e) * sin(u)
        M = u - e * sin(u)
        exp_iNpl1M = exp(-I * N * M)
        r_by_a = 1 - e * cos(u)
        integrand = expif * exp_iNpl1M / (r_by_a)**2
        s = series(integrand,e,0,N)
        subdict={
            sin(u):(x - 1/x) / 2 / I,
            exp(I*u): x,
            exp(-I*u): 1/x,
            cos(u):(x + 1/x) / 2}
        term = s.coeff(e,N-1)
        term = term.subs(subdict).expand()
        coeff = term.coeff(x,0).evalf()
    alpha  = N**(-2/3)
    return -1 * coeff * alpha


def secular_eps_l_Df_m(l,m,e,e1):
    """
        Get the secularly-averaged value of 
            eps^l * exp[ i*m*(f'-f) ] 
        where l and m are integers, f' and f are true anamolies, and
            eps =  (r/a) / (r'/a')  - 1
        where r and a are radial distance and semi-major axis.
    """
    s = 0
    for i in range(l+1):
        s = s + binomial(l,i)*(-1)**(l-i) * Xlm0(i,m,e) * Xlm0(-i-1,m,e1)
    return s
def secular_DF_harmonic_term(e,e1,dw,m,order):
    s = 0
    b = Function('b') 
    alpha = symbols('alpha')
    if order%2==0:
        maxk = order + 1
    else:
        maxk=order
    for i in range(maxk):
     s = s + alpha**i *  b(1/S(2),m,i,alpha) / factorial(i) * secular_eps_l_Df_m(i,m,e,e1)
    if m==0:
        s=s / S(2)
    return s * cos(m*dw)
def secular_DF(e,e1,w,w1,order):
    """ 
    Return the secular component of the disturbing function for coplanet planets up to
    the specified order in the planets' eccentricities for an inner plaent with eccentricity 
    and longitude of periapse (e,w) and outer planet eccentricity/longitude of periape (e1,w1)
    """
    s = 0
    dw = w1 - w
    # Introduce `eps' as order parameter multiplying eccentricities. 
    # 'eps' is set to 1 at teh end of calculation.
    eps = S('epsilon')
    maxM = int( np.floor( order / 2.) )
    for i in range(maxM+1):
        term = secular_DF_harmonic_term(eps*e,eps*e1,dw,i,order) 
        term = term.series(eps,0,order+1).removeO()
        s = s + expand_trig(term) 
    return s.subs(eps,1)
def secular_DF_full(e,e1,w,w1,order):
    """ 
    Same as 'secular_DF' but uses the full expression for the Hansen coefficients that enter
    the disturbing function. While these expressions are typically more complicated when written
    in terms of eccentricity, they may be simpler when written in terms of canoncial momenta.
    """
    s = 0
    dw = w1 - w
    maxM = int( np.floor( order / 2.) )
    for i in range(maxM+1):
        s = s + secular_DF_harmonic_term(e,e1,dw,i,order) 
    return s

class laplace_B(Function):
    nargs=4
    @classmethod
    def eval(cls, s,j,n,alpha):
    # laplace coeffcients don't evaluate well with the C code
    # for large n, therefore I've substituted the exact expression
    # in terms of the elliptic K function when j=0
        if j is S.Zero:
            x = S('x')
            exprn = 4 / np.pi * diff(elliptic_k(x*x),x,n)
            return exprn.subs(x,alpha)
        elif j is S.One:
            x = S('x')
            exprn0 =  4 / np.pi *  (elliptic_k(x*x) - elliptic_e(x*x) ) / x 
            exprn = diff(exprn0, x ,n)
            return exprn.subs(x,alpha)
        else:
            return laplace_coefficient(s,j,n,alpha)/ alpha**n


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
        X^{a,b}_{c,0}
        
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

        X^{a,b}_c(e) = e^{|c-b|} \times
        \sum_{\sigma=0}^\infty HansenCoefficient_term(a,b,c)e^{2\sigma}

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
            term = (-1)**(k+m+n-p) * KK(i,n,m)
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

def Xi(N,n,k,m):
    return (-1)**(N-n) * binom(n, N-n-m) * binom(abs(k)/2,m)


def DFCoeff_C(j1,j2,j3,j4,j5,j6,z1,z2,z3,z4):
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
        
    Returns
    -------
    dictionary 
        The coefficient is given by the sum over laplace coefficients
        contained in the dictionary entries: 
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
    return dict(total)

def DFCoeff_Cbar(j1,j2,j3,j4,j5,j6,N1,N2,N3,N4):
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
            \sum C \times \alpha^p \frac{d^{n}}{d\alpha^{n}} b_{s}^{j}(\alpha)
        where the dictionary entries are in the form { (p,(s,j,n)) : C }
    """
    terms_total = defaultdict(float)
    for n1 in range(N1+1):
        for m1 in range(N1-n1+1):
            for n2 in range(N2+1):
                for m2 in range(N2-n2+1):
                    for n3 in range(N3+1):
                        for m3 in range(N3-n3+1):
                            for n4 in range(N4+1):
                                for m4 in range(N4-n4+1):
                                    term_dict = DFCoeff_C(j1,j2,j3,j4,j5,j6,n1,n2,n3,n4)
                                    prefactor = Xi(N1,n1,j5,m1) * Xi(N2,n2,j6,m2) * Xi(N3,n3,j3,m3) * Xi(N4,n4,j4,m4)
                                    if prefactor != 0.:
                                        for key,val in term_dict.items():
                                            terms_total[key] += prefactor * val
    return dict(terms_total)
