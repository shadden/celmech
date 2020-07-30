import numpy as np
from scipy.special import k0,k1,p_roots
import warnings
from . import clibcelmech
from ctypes import POINTER,c_int,c_double,c_long


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
def frequency_modified_fourier_transform(inpt, Nfreq, method_flag = 3, min_freq = None, max_freq = None):
    """
    Apply the frequency-modified Fourier transfrorm algorithm (Šidlichovský & Nesvorný 1996) [#]_
    to a time series to determine the series' principle Fourier modes. This function simply
    proivdes a wrapper to to C implementation written by D. Nesvorný available at 
    https://www-n.oca.eu/nesvorny/programs.html.

    .. [#] `ADS link <https://ui.adsabs.harvard.edu/abs/1996CeMDA..65..137S/abstract>`

    Arguments
    ---------
    inpt : ndarray, shape (N,3)
      Input data time series in the form 
        [
         [time[0],Re(z[0]),Im(z[0])],
         ...,
         [time[i],Re(z[i]),Im(z[i])],
         ...,
         [time[N-1],Re(z[N-1]),Im(z[N-1])]
        ]
    
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
    input_arr = np.array(inpt,order='C',dtype=np.float64)
    Ndata = _nearest_pow2(len(inpt))
    _Nyq = 2 * np.pi * 0.5
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
