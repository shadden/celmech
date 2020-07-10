import numpy as np
from ctypes import *
from .disturbing_function import  DFCoeff_C,eval_DFCoeff_dict
#from . import clibcelmech
from celmech.disturbing_function import DFArguments_dictionary
libname = "/Users/shadden/Projects/celmech/src/libcelmech.so"
clibcelmech = CDLL(libname)

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

def zcombos_iter(ztot):
    for z1 in range(ztot+1):
        for z2 in range(ztot+1-z1):
            for z3 in range(ztot+1-z1-z2):
                z4 = ztot - z1 - z2 - z3
                yield (z1,z2,z3,z4)
                
def generate_resonance_terms(j,k,Nmin,Nmax):
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
                    for zc in zcombos_iter(ztot):
                        js = (j1,k1 - j1,*arg)
                        args.append((js,zc))
    return args
def get_generic_DFCoeff_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4):
    return symbols("C_{0}\,{1}\,{2}\,{3}\,{4}\,{5}^{6}\,{7}\,{8}\,{9}".format(
        k1,k2,k3,k4,k5,k6,z1,z2,z3,z4)
    )
from sympy import symbols,exp,I
def get_term_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4):
    X,Xprime,Y,Yprime = symbols('X,Xprime,Y,Yprime')
    Xbar,Xprimebar,Ybar,Yprimebar = symbols('Xbar,Xbarprime,Ybar,Ybarprime')
    l,lprime = symbols('lambda,lambdaprime')
    coeff = get_generic_DFCoeff_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4)

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
    def __init__(self,resterm_list,alpha):
        self.expression = 0 
        self.alpha = alpha
        kmax = 0
        Nmax = 0
        self.slast_pointer = None
        self.s_dXbar_last_pointer = None
        for ks,zs in resterm_list:
            self.expression += get_term_symbol(*ks,*zs)
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
            s.coeff = eval_DFCoeff_dict(DFCoeff_C(*ks,*zs),self.alpha)
            s.next = self.slast_pointer
            self.slast_pointer = pointer(s)
        self.s0 = s
        self.Nmax = Nmax
        self.kmax = kmax
        self.dXY_dQP  = np.block([
            [-1j * np.eye(4), +1j * np.eye(4)],
            [np.eye(4),np.eye(4)]
            ])
        Zeros = np.zeros((4,4))
        Id = np.eye(4)
        self.Omega = np.block([[Zeros,Id],[-Id,Zeros]])
    @classmethod
    def from_resonance_range(cls,j,k,Nmin,Nmax):
        terms = generate_resonance_terms(j,k,Nmin,Nmax)
        alpha = ((j-k)/j)**(2/3)
        return cls(terms,alpha)

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
        dH_dq = np.array(dS_dxy_im) - np.array(dS_dxybar_im)
        dH_dp = np.array(dS_dxybar_re) + np.array(dS_dxy_re)
        return np.float64(sum_re),np.hstack((dH_dp,-dH_dq))

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
        return {'series':H,'derivs':vardot,'jacobian':jac_qp,'jac_xy':jac_xy}
