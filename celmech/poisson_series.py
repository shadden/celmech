import numpy as np
from ctypes import *
from .disturbing_function import  df_coefficient_C,evaluate_df_coefficient_dict,list_resonance_terms
#from . import clibcelmech
from celmech.disturbing_function import df_arguments_dictionary
#libname = "/Users/shadden/Projects/celmech/src/libcelmech.so"
#clibcelmech = CDLL(libname)
from . import clibcelmech
_rt2 = np.sqrt(2)
_rt2_inv = 1  / _rt2

class SeriesTerm(Structure):
    pass
SeriesTerm._fields_ = [
            ("k",c_int * 6),
            ("z",c_int * 4),
            ("coeff",c_double),
            ("next",POINTER(SeriesTerm))
            ]

_evaluate_series = clibcelmech.evaluate_series
_evaluate_series.argtypes = [
    np.ctypeslib.ndpointer(dtype = np.complex128,ndim=1),
    np.ctypeslib.ndpointer(dtype = np.complex128,ndim=1),
    POINTER(SeriesTerm),
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double)
]
_evaluate_series.restype = None

_evaluate_series_and_derivs = clibcelmech.evaluate_series_and_derivs
_evaluate_series_and_derivs.argtypes = [
    np.ctypeslib.ndpointer(dtype = np.complex128,ndim=1),
    np.ctypeslib.ndpointer(dtype = np.complex128,ndim=1),
    POINTER(SeriesTerm),
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
] + [(4 * c_double) for _ in range(4)]
_evaluate_series_and_derivs.restype = None

_evaluate_series_and_jacobian = clibcelmech.evaluate_series_and_jacobian
_evaluate_series_and_jacobian.argtypes = [
    np.ctypeslib.ndpointer(dtype = np.complex128,ndim=1),
    np.ctypeslib.ndpointer(dtype = np.complex128,ndim=1),
    POINTER(SeriesTerm),
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
] + [(4 * c_double) for _ in range(4)] + [(64 * c_double) for _ in range(2)]
_evaluate_series_and_jacobian.restype = None

def get_generic_df_coefficient_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4):
    return symbols("C_{0}\,{1}\,{2}\,{3}\,{4}\,{5}^{6}\,{7}\,{8}\,{9}".format(
        k1,k2,k3,k4,k5,k6,z1,z2,z3,z4)
    )
from sympy import symbols,exp,I
def get_term_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4):
    X,Xprime,Y,Yprime = symbols('X,Xprime,Y,Yprime')
    Xbar,Xprimebar,Ybar,Yprimebar = symbols('Xbar,Xbarprime,Ybar,Ybarprime')
    l,lprime = symbols('lambda,lambdaprime')
    coeff = get_generic_df_coefficient_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4)

    term = coeff * exp(I * (k1 * lprime + k2 * l))
    term *= X**k3 if k3>0 else Xbar**(-k3)
    term *= Xprime**k4 if k4>0 else Xprimebar**(-k4)
    term *= Y**k5 if k5>0 else Ybar**(-k5)
    term *= Yprime**k6 if k6>0 else Yprimebar**(-k6)
    for z,prod in zip((z1,z2,z3,z4), (Y*Ybar,Yprime*Yprimebar,X*Xbar,Xprime*Xprimebar)):
        if z>0:
            term *= prod**z
    return term

class DFTermSeries(object):
    def __init__(self,resterm_dictionary,Lambda0In,Lambda0Out):
        r"""
        An object representing a Poisson series in variables
         exp[i\lambda_out],exp[i\lambda_in],Xin,Xout,Yin, and Yout.

        An instance can be used to calculate the value of the 
        Poisson series as well as its first and second derivatives
        with respect to Xin,Xout,Yin,and Yout.

        Arguments
        ---------
        resterm_dictionary : dict
          Supply the terms of the Poisson series and their coefficients.
          Dictionary entries are in the form {(kvec,zvec):Coeff}          
        Lambda0In : float
          The value of the inner planet's canonical momentum, Lambda,
          which is trated as consant in the Poisson series. 
          The value is neccessary for computing the value of derivatives
          of the canonical variables \eta,\kappa,\rho,\sigma.
        Lambda0Out : float
          The value of the outer planet's canonical momentum, Lambda,
          which is trated as consant in the Poisson series. 
          The value is neccessary for computing the value of derivatives
          of the canonical variables \eta,\kappa,\rho,\sigma.
        """
        self.slast_pointer = None
        self.s_dXbar_last_pointer = None
        kmax = 0
        Nmax = 0
        for kzpair,coeff_value in resterm_dictionary.items():
            ks,zs = kzpair
            kmax = max(kmax,abs(ks[0]),abs(ks[1]))
            Nmax = max(
                      Nmax,
                      abs(ks[2]) + zs[2],
                      abs(ks[3]) + zs[3],
                      abs(ks[4]) + zs[0],
                      abs(ks[5]) + zs[1]
                    )
            s = SeriesTerm()
            s.k = (6 * c_int)(*ks)
            s.z = (4 * c_int)(*zs)
            s.coeff = coeff_value
            s.next = self.slast_pointer
            self.slast_pointer = pointer(s)
        self.s0 = s
        self.Nmax = Nmax
        self.kmax = kmax
        rtLmbdaInv = 1 / np.sqrt([Lambda0In,Lambda0Out])
        mtrx = np.diag( np.concatenate((rtLmbdaInv, 0.5 * rtLmbdaInv )) )
        self.dXY_dQP  = np.block([
            [-1j * mtrx, +1j * mtrx],
            [mtrx, mtrx]
            ])
        Zeros = np.zeros((4,4))
        Id = np.eye(4)
        self.Omega = np.block([[Zeros,Id],[-Id,Zeros]])

    @classmethod
    def from_resonance_list(cls,resterm_list,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
        muIn = mIn * (MIn - mIn) / MIn
        muOut = mOut * (MOut - mOut) / MOut
        aIn0 = (Lambda0In / muIn)**2 / MIn / G
        aOut0 = (Lambda0Out / muOut)**2 / MOut / G
        alpha0 = aIn0 / aOut0
        aOut_inv = G*MOut*muOut*muOut / Lambda0Out / Lambda0Out  
        prefactor = -G * mIn * mOut * aOut_inv
        assert alpha0 < 1, "Particles are not in order by semi-major axis."
        resterm_dictionary  = {
                (ks,zs):prefactor * evaluate_df_coefficient_dict(df_coefficient_C(*ks,*zs),alpha0)
                for ks,zs in resterm_list
                }
        return cls(resterm_dictionary,Lambda0In,Lambda0Out)
    @classmethod
    def from_resonance_range(cls,j,k,Nmin,Nmax,G,mIn,mOut,MIn,MOut,Lambda0In,Lamda0Out):
        terms = list_resonance_terms(j,k,Nmin,Nmax)
        return cls.from_resonance_list(terms,G,mIn,mOut,MIn,MOut,Lambda0In,Lamda0Out)

    def _evaluate(self,lambda_arr, xy_arr):
        expIL = np.exp( 1j * lambda_arr)
        sum_re,sum_im  = c_double(),c_double()
        _evaluate_series(
                expIL,
                xy_arr,
                self.s0,
                self.kmax,
                self.Nmax,
                pointer(sum_re),
                pointer(sum_im)
            )
        return np.float64(sum_re)

    def _evaluate_with_derivs(self,lambda_arr,xy_arr):
        expIL = np.exp( 1j * lambda_arr)
        sum_re,sum_im  = c_double(),c_double()
        dS_dxy_re,dS_dxy_im,dS_dxybar_re,dS_dxybar_im = [(4 * c_double)() for _ in range(4)]
        _evaluate_series_and_derivs(
                expIL,
                xy_arr,
                self.s0,
                self.kmax,
                self.Nmax,
                pointer(sum_re),
                pointer(sum_im),
                dS_dxy_re,dS_dxy_im,dS_dxybar_re,dS_dxybar_im
            )
        dSdxy = np.concatenate((
            np.array(dS_dxy_re) + 1j * np.array(dS_dxy_im),
            np.array(dS_dxybar_re) + 1j * np.array(dS_dxybar_im)
            ))
        derivs = np.real(self.Omega @ self.dXY_dQP @ dSdxy)
        return np.float64(sum_re),derivs

    def _evaluate_with_jacobian(self,lambda_arr,xy_arr):
        expIL = np.exp( 1j * lambda_arr)
        sum_re,sum_im  = c_double(),c_double()
        dS_dxy_re,dS_dxy_im,dS_dxybar_re,dS_dxybar_im = [(4 * c_double)() for _ in range(4)]
        jac_re,jac_im = [(64 * c_double)() for _ in range(2)]
        _evaluate_series_and_jacobian(
                expIL,
                xy_arr,
                self.s0,
                self.kmax,
                self.Nmax,
                pointer(sum_re),
                pointer(sum_im),
                dS_dxy_re,dS_dxy_im,dS_dxybar_re,dS_dxybar_im,
                jac_re, jac_im
            )

        dSdxy = np.concatenate((
            np.array(dS_dxy_re) + 1j * np.array(dS_dxy_im),
            np.array(dS_dxybar_re) + 1j * np.array(dS_dxybar_im)
            ))
        derivs = np.real(self.Omega @ self.dXY_dQP @ dSdxy)
        jac_xy = (np.array(jac_re) + 1j * np.array(jac_im)).reshape(8,8)
        jac_qp = self.Omega @ np.real(self.dXY_dQP @ jac_xy @ self.dXY_dQP.T)
        return np.float64(sum_re), derivs, jac_qp, jac_xy

    def PoincareParticlesEvaluate(self,pvars,indexIn,indexOut):
        ps = (pvars.particles[indexIn],pvars.particles[indexOut])
        xy = [(p.kappa - 1j * p.eta) / np.sqrt(p.Lambda) for p in ps]
        xy += [0.5 * (p.sigma - 1j * p.rho) / np.sqrt(p.Lambda) for p in ps]
        xy = np.array(xy)
        lambdas = np.array([p.l for p in ps])
        H,vardot,jac_qp,jac_xy = self._evaluate_with_jacobian(lambdas,xy)
        return {'Hamiltonain':H,'derivatives':vardot,'Jacobian':jac_qp}
