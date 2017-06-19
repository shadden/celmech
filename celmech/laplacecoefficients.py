from . import clibcelmech
from ctypes import Structure, c_double, POINTER, c_float, c_int, c_uint, c_uint32, c_int64, c_long, c_ulong, c_ulonglong, c_void_p, c_char_p, CFUNCTYPE, byref, create_string_buffer, addressof, pointer, cast
from scipy.integrate import quad
import math
import numpy as np

def laplace_coefficient(s,j,n,a):
    """
    Calculates nth derivative with respect to a (alpha) of Laplace coefficient b_s^j(a).
    Code due to Jack Wisdom. Changed notation to match Murray & Dermott (Eq. 6.67)
    
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

def general_order_coefficient(res_j, order, epower, a):
    clibcelmech.GeneralOrderCoefficient.restype = c_double
    return clibcelmech.GeneralOrderCoefficient(c_int(res_j), c_int(order), c_int(epower), c_double(a))
