import numpy as np
import warnings
from abc import ABC, abstractmethod
from .hamiltonian import Hamiltonian
from .disturbing_function import get_fg_coeffs, general_order_coefficient, secular_DF,laplace_B, laplace_coefficient
from .disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
from .transformations import masses_to_jacobi, masses_from_jacobi
from .poincare import Poincare
class EvolutionOperator(ABC):
    def __init__(self,initial_state,dt):
        self._dt = dt
        self.state = initial_state
        self.particles = initial_state.particles

    @abstractmethod
    def apply(self):
        pass

    @property
    @abstractmethod
    def dt(self):
        return self._dt
    @dt.setter
    def dt(self,value):
        self._dt = dt

class KeplerianEvolutionOperator(EvolutionOperator):
    def __init__(self,initial_state,dt):
        super(KeplerianEvolutionOperator,self).__init__(initial_state,dt)
        self.G =  self.state.G
        self.m = np.array([p.m for p in self.state.particles[1:]]) 
        self.M = np.array([p.M for p in self.state.particles[1:]]) 
        self.GGMMmmm = (self.G*self.M)**2 * self.m**3
    def apply(self):
        lambda_dot = self.get_lambda_dot()
        dlambda = self.dt * lambda_dot
        ps = self.particles
        for p,dl in zip(ps[1:],dlambda):
            p.l += dl
        
    def get_lambda_dot(self):
        ps = self.particles
        L = np.array([p.Lambda for p in ps[1:]])
        lambda_dot = self.GGMMmmm/L/L/L
        return lambda_dot

    @property
    def dt(self):
        return super().dt
    @dt.setter
    def dt(self,val):
        self._dt = val
from .poincare import LaplaceLagrangeSystem
from scipy.linalg import expm

class LinearSecularEvolutionOperator(EvolutionOperator):
    def __init__(self,initial_state,dt):
        super(LinearSecularEvolutionOperator,self).__init__(initial_state,dt)
        LL_system = LaplaceLagrangeSystem.from_Poincare(self.state)
        self.ecc_matrix = LL_system.Neccentricity_matrix
        self.inc_matrix = LL_system.Ninclination_matrix
        self.ecc_operator_matrix = expm(-1j * self.dt * self.ecc_matrix)
        self.inc_operator_matrix = expm(-1j * self.dt * self.inc_matrix)

    @property
    def dt(self):
        return super().dt
    @dt.setter
    def dt(self,val):
        self._dt = val
        self.ecc_operator_matrix = expm(-1j * self.dt * self.ecc_matrix)
        self.inc_operator_matrix = expm(-1j * self.dt * self.inc_matrix)

    def _get_x_vector(self):
        eta = np.array([p.eta for p in self.particles[1:]])
        kappa = np.array([p.kappa for p in self.particles[1:]])
        x =  (kappa - 1j * eta) / np.sqrt(2)
        return x

    def _get_y_vector(self):
        rho = np.array([p.rho for p in self.particles[1:]])
        sigma = np.array([p.sigma for p in self.particles[1:]])
        y =  (sigma - 1j * rho) / np.sqrt(2)
        return y

    def _set_x_vector(self,x):
        for p,xi in zip(self.particles[1:],x):
            p.kappa = np.sqrt(2) * np.real(xi)
            p.eta =  np.sqrt(2) * np.real(1j * xi)

    def _set_y_vector(self,y):
        for p,yi in zip(self.particles[1:],y):
            p.sigma = np.sqrt(2) * np.real(yi)
            p.rho =  np.sqrt(2) * np.real(1j * yi)

    def apply(self):
        x = self._get_x_vector()
        y = self._get_y_vector()
        xnew = self.ecc_operator_matrix @ x
        ynew = self.inc_operator_matrix @ y
        self._set_x_vector(xnew)
        self._set_y_vector(ynew)
        

class LinearInclinationResonancOperator(EvolutionOperator):
    r"""
    Evolution operator for linear equation of the form:
    .. math::
        dy/dt = -i e^{2i\theta} A \cdot y^* 
    Equations of motion derived from second-order resonances,
    expanded to leading order in inclinations can be written in this form.

    Parameters
    ----------
    A : ndarray, shape (M,M)
        The matrix appearing in the right-hand side
        of the equation.
    b : ndarray, shape (M)
        The vector in the r.h.s. of the equation.
    dt : float
        The timestep of the opertor.
    """
    def __init__(self,initial_state,indexIn,indexOut,res_vec, A,dt):
        super(LinearInclinationResonancOperator,self).__init__(initial_state,dt)        
        self.A = A
        self.Ainv = np.linalg.inv(A)
        vals,T = np.linalg.eigh(self.A)
        self.T = T
        self.Ttr = T.transpose()
        self.D = np.diag(vals)
        self.eigs = vals
        self._dt = dt
        self.cosh_lmbda_dt = np.cosh(self._dt * self.eigs)
        self.sinh_lmbda_dt = np.sinh(self._dt * self.eigs)
        self.indexIn = indexIn
        self.indexOut = indexOut
        self.particleIn = self.particles[indexIn]
        self.particleOut = self.particles[indexOut]
        self.res_vec = res_vec
        self.Lambdas_vec = np.array([-1 * res_vec[1] , res_vec[0]])
        self.Lambdas_Mtrx = np.linalg.inv([ self.Lambdas_vec , [1,1]])
    @property
    def dt(self):
        return super().dt
    @dt.setter
    def dt(self,val):
        self._dt = val
        self.cosh_lmbda_dt = np.cosh(val * self.eigs)
        self.sinh_lmbda_dt = np.sinh(val * self.eigs)
    def _get_rho_vec(self):
        return np.array([self.particleIn.rho,self.particleOut.rho])
    def _get_sigma_vec(self):
        return np.array([self.particleIn.sigma,self.particleOut.sigma])
    def _get_theta(self):
        lvec = np.array([ self.particleIn.l, self.particleOut.l])
        return self.res_vec @ lvec
    def _get_Lambdas(self):
        return np.array([self.particleIn.Lambda,self.particleOut.Lambda])
    def _get_Qs(self):
        return np.array([self.particleIn.Q,self.particleOut.Q])
    def _set_rho_vec(self,rho_vec):
        rhoIn,rhoOut = rho_vec
        self.particleIn.rho = rhoIn
        self.particleOut.rho = rhoOut
    def _set_sigma_vec(self,sigma_vec):
        sigmaIn,sigmaOut = sigma_vec
        self.particleIn.sigma = sigmaIn
        self.particleOut.sigma = sigmaOut
    def _get_RS_vecs(self,s,c):
        rho = self._get_rho_vec()
        sigma = self._get_sigma_vec()
        R = self.Ttr @ rho 
        S = self.Ttr @ sigma 
        return R,S
    def _RS_to_rhosigma(self,R,S,s,c):
        r = self.T @ R 
        s = self.T @ S
        return r,s
    def _RSmap(self,Rvec,Svec,s,c):
        s2 = 2 * s * c
        c2 = c*c - s*s
        S1vec = self.cosh_lmbda_dt * Svec  +  self.sinh_lmbda_dt * (s2 * Svec + c2 * Rvec)
        R1vec = self.cosh_lmbda_dt * Rvec  +  self.sinh_lmbda_dt * (c2 * Svec - s2 * Rvec)
        return R1vec,S1vec
    def _advance_inclination_variables(self):
        theta = self._get_theta()   
        s,c = np.sin(theta),np.cos(theta)
        R,S = self._get_RS_vecs(s,c)
        R1,S1  = self._RSmap(R,S,s,c)
        rho1,sigma1 = self._RS_to_rhosigma(R1,S1,s,c)
        self._set_rho_vec(rho1)
        self._set_sigma_vec(sigma1)
    def _set_Lambdas(self,Lambdas):
        self.particleIn.Lambda = Lambdas[0]
        self.particleOut.Lambda = Lambdas[1]
    def apply(self):
        Lambdas = self._get_Lambdas()
        Qs0  = self._get_Qs()
        C1 = self.Lambdas_vec @ Lambdas
        C2 = np.sum(Lambdas) - np.sum(Qs0)
        self._advance_inclination_variables()
        Qs1 = self._get_Qs()
        Lambdas1 = self.Lambdas_Mtrx @ np.array([ C1, C2 + np.sum(Qs1) ])
        self._set_Lambdas(Lambdas1)

class LinearEccentricityResonancOperator(EvolutionOperator):
    r"""
    Evolution operator for linear equation of the form:
    .. math::
        dx/dt = -i e^{2i\theta} A \cdot x^* - \frac{i}{2}e^{i\theta} b
    Equations of motion derived from first- and second-order resonances,
    expanded to second order in eccentricities can be written in this form.

    An instance of `LinearEquationEvolutionOperator` provides
    a callable function that gives the evolution of h and k, where
        .. math::
            x = k - i h
    for a timestep dt and fixed angle value $\theta$.

    Parameters
    ----------
    A : ndarray, shape (M,M)
        The matrix appearing in the right-hand side
        of the equation.
    b : ndarray, shape (M)
        The vector in the r.h.s. of the equation.
    dt : float
        The timestep of the opertor.
    """
    def __init__(self,initial_state,indexIn,indexOut,res_vec, A, b,dt):
        super(LinearEccentricityResonancOperator,self).__init__(initial_state,dt)        
        self.A = A
        self.Ainv = np.linalg.inv(A)
        self.b = b
        self.Ainv_dot_b = self.Ainv @ b
        vals,T = np.linalg.eigh(self.A)
        self.T = T
        self.Ttr = T.transpose()
        self.D = np.diag(vals)
        self.eigs = vals
        self._dt = dt
        self.cosh_lmbda_dt = np.cosh(self._dt * self.eigs)
        self.sinh_lmbda_dt = np.sinh(self._dt * self.eigs)
        self.indexIn = indexIn
        self.indexOut = indexOut
        self.particleIn = self.particles[indexIn]
        self.particleOut = self.particles[indexOut]
        self.res_vec = res_vec
        self.Lambdas_vec = np.array([-1 * res_vec[1] , res_vec[0]])
        self.Lambdas_Mtrx = np.linalg.inv([ self.Lambdas_vec , [1,1]])
    @property
    def dt(self):
        return super().dt

    @dt.setter
    def dt(self,val):
        self._dt = val
        self.cosh_lmbda_dt = np.cosh(val * self.eigs)
        self.sinh_lmbda_dt = np.sinh(val * self.eigs)

    def _get_eta_vec(self):
        return np.array([self.particleIn.eta,self.particleOut.eta])

    def _get_kappa_vec(self):
        return np.array([self.particleIn.kappa,self.particleOut.kappa])

    def _get_theta(self):
        lvec = np.array([ self.particleIn.l, self.particleOut.l])
        return self.res_vec @ lvec

    def _get_Lambdas(self):
        return np.array([self.particleIn.Lambda,self.particleOut.Lambda])
    def _get_Gammas(self):
        return np.array([self.particleIn.Gamma,self.particleOut.Gamma])

    def _set_eta_vec(self,eta_vec):
        etaIn,etaOut = eta_vec
        self.particleIn.eta = etaIn
        self.particleOut.eta = etaOut
        
    def _set_kappa_vec(self,kappa_vec):
        kappaIn,kappaOut = kappa_vec
        self.particleIn.kappa = kappaIn
        self.particleOut.kappa = kappaOut

    def _get_HK_vecs(self,s,c):
        eta = self._get_eta_vec()
        kappa = self._get_kappa_vec()
        H = self.Ttr @ (eta - 0.5 * np.sqrt(2) * self.Ainv_dot_b * s)
        K = self.Ttr @ (kappa + 0.5 * np.sqrt(2) * self.Ainv_dot_b * c)
        return H,K

    def _HK_to_etakappa(self,H,K,s,c):
        h = self.T @ H + 0.5 * np.sqrt(2) * self.Ainv_dot_b * s
        k = self.T @ K - 0.5 * np.sqrt(2) * self.Ainv_dot_b * c
        return h,k

    def _HKmap(self,Hvec,Kvec,s,c):
        s2 = 2 * s * c
        c2 = c*c - s*s
        K1vec = self.cosh_lmbda_dt * Kvec  +  self.sinh_lmbda_dt * (s2 * Kvec + c2 * Hvec)
        H1vec = self.cosh_lmbda_dt * Hvec  +  self.sinh_lmbda_dt * (c2 * Kvec - s2 * Hvec)
        return H1vec,K1vec

    def _advance_eccentricity_variables(self):
        theta = self._get_theta()   
        s,c = np.sin(theta),np.cos(theta)
        H,K = self._get_HK_vecs(s,c)
        H1,K1  = self._HKmap(H,K,s,c)
        eta1,kappa1 = self._HK_to_etakappa(H1,K1,s,c)
        self._set_eta_vec(eta1)
        self._set_kappa_vec(kappa1)
    def _set_Lambdas(self,Lambdas):
        self.particleIn.Lambda = Lambdas[0]
        self.particleOut.Lambda = Lambdas[1]

    def apply(self):
        Lambdas = self._get_Lambdas()
        Gammas0  = self._get_Gammas()
        C1 = self.Lambdas_vec @ Lambdas
        C2 = np.sum(Lambdas) - np.sum(Gammas0)
        self._advance_eccentricity_variables()
        Gammas1 = self._get_Gammas()
        Lambdas1 = self.Lambdas_Mtrx @ np.array([ C1, C2 + np.sum(Gammas1) ])
        self._set_Lambdas(Lambdas1)

class FirstOrderEccentricityResonanceOperator(LinearEccentricityResonancOperator):
    def __init__(self,initial_state,dt,j,indexIn=1,indexOut=2,Lambda0=None):
        res_vec = np.array([1-j,j])
        pIn = initial_state.particles[indexIn]
        pOut = initial_state.particles[indexOut]
        self.mIn = pIn.m
        self.MIn = pIn.M
        self.mOut = pOut.m
        self.MOut = pOut.M
        self.j = j
        if Lambda0 is None:
            _pvars = Poincare(initial_state.G,[pIn,pOut])
            _,Lambda0 = get_res_chain_reference_Lambdas_and_semimajor_axes(_pvars,[j-1])
            _,self.Lambda0In,self.Lambda0Out = Lambda0
        else:
            self.Lambda0In = Lambda0[indexIn]
            self.Lambda0Out = Lambda0[indexOut]
        Amtrx,bvec = get_first_order_eccentricity_resonance_matrix_and_vector(
                self.j,
                initial_state.G,
                self.mIn,
                self.mOut,
                self.MIn,
                self.MOut,
                self.Lambda0In,
                self.Lambda0Out
        )
        super(FirstOrderEccentricityResonanceOperator,self).__init__(initial_state,indexIn,indexOut,res_vec ,Amtrx,bvec,dt)

class SecondOrderEccentricityResonanceOperator(LinearEccentricityResonancOperator):
    def __init__(self,initial_state,dt,j,indexIn=1,indexOut=2,Lambda0=None):
        res_vec = np.array([2-j,j]) / 2
        pIn = initial_state.particles[indexIn]
        pOut = initial_state.particles[indexOut]
        self.mIn = pIn.m
        self.MIn = pIn.M
        self.mOut = pOut.m
        self.MOut = pOut.M
        self.j = j
        if Lambda0 is None:
            _pvars = Poincare(initial_state.G,[pIn,pOut])
            _,Lambda0 = get_res_chain_reference_Lambdas_and_semimajor_axes(_pvars,[(j-2)/2])
            _,self.Lambda0In,self.Lambda0Out = Lambda0
        else:
            self.Lambda0In = Lambda0[indexIn]
            self.Lambda0Out = Lambda0[indexOut]
        Amtrx = get_second_order_eccentricity_resonance_matrix(
                self.j,
                initial_state.G,
                self.mIn,
                self.mOut,
                self.MIn,
                self.MOut,
                self.Lambda0In,
                self.Lambda0Out
        )
        bvec = np.zeros(2)
        super(SecondOrderEccentricityResonanceOperator,self).__init__(initial_state,indexIn,indexOut,res_vec,Amtrx,bvec,dt)

class SecondOrderInclinationResonanceOperator(LinearInclinationResonancOperator):
    def __init__(self,initial_state,dt,j,indexIn=1,indexOut=2,Lambda0=None):
        res_vec = np.array([2-j,j]) / 2
        pIn = initial_state.particles[indexIn]
        pOut = initial_state.particles[indexOut]
        self.mIn = pIn.m
        self.MIn = pIn.M
        self.mOut = pOut.m
        self.MOut = pOut.M
        self.j = j
        if Lambda0 is None:
            _pvars = Poincare(initial_state.G,[pIn,pOut])
            _,Lambda0 = get_res_chain_reference_Lambdas_and_semimajor_axes(_pvars,[(j-2)/2])
            _,self.Lambda0In,self.Lambda0Out = Lambda0
        else:
            self.Lambda0In = Lambda0[indexIn]
            self.Lambda0Out = Lambda0[indexOut]
        Amtrx = get_second_order_inclination_resonance_matrix(
                self.j,
                initial_state.G,
                self.mIn,
                self.mOut,
                self.MIn,
                self.MOut,
                self.Lambda0In,
                self.Lambda0Out
        )
        bvec = np.zeros(2)
        super(SecondOrderInclinationResonanceOperator,self).__init__(initial_state,indexIn,indexOut,res_vec,Amtrx,dt)

def get_res_chain_reference_Lambdas_and_semimajor_axes(pvars,slist):
    coeffs = np.zeros(pvars.N)
    alpha_inv = np.zeros(pvars.N)
    alpha_inv[1] = 1    
    coeffs[1] = 1 + slist[0]
    ps = pvars.particles
    m = np.array([p.m for p in ps])
    M = np.array([p.M for p in ps])
    tot = coeffs[1] * m[1] * np.sqrt(M[1])
    for i in range(2,len(coeffs)):
        coeffs[i] = coeffs[i-1] * slist[i-2] / (1 + slist[i-2])
        alpha_inv[i] = alpha_inv[i-1]  * ((1 + slist[i-2]) / slist[i-2])**(2/3)
        tot += coeffs[i] * m[i] * np.sqrt(M[i] * alpha_inv[i])        
    Lvals = np.append([0],[p.Lambda for p in ps[1:]])
    C = coeffs @ Lvals
    a10 = (C / tot)**2
    a0 = a10 * alpha_inv 
    L0 = m * np.sqrt(M * a0)
    return a0,L0

def get_first_order_eccentricity_resonance_matrix_and_vector(j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    """
    Get matrix A and vector b that define the Hamiltonian terms
    .. math::
        H_{res} = \frac{e^{2i\theta}}{2} \left( \bar{x} \cdot A \cdot \bar{x} + b e^{-i\theta} \cdot \bar{x} \right)

    associated with a first-order MMR in the disturbing function expansion.

    Arguments
    ---------
    j : int
        Determines the j:j-1 first order MMR.
    G : float
        Gravitational constant.
    mIn : float
        Mass of inner planet
    mOut : float
        Mass of outer planet
    MIn : float
        Stellar mass for inner planet
    Mout : float
        Stellar mass for outer planet
    Lambda0In : float
        Reference value of Lambda for inner planet
    Lambda0Out : float
        Reference value of Lambda for outer planet
    Returns
    -------
    A : sympy.Matrix , shape (2,2)
        The matrix A
    b : sympy.Matrix, shape (2,)
        The vector b

    """

    zs = [0,0,0,0]
    aIn0 = (Lambda0In / mIn)**2 / MIn
    aOut0 = (Lambda0Out / mOut)**2 / MOut
    alpha0 = aIn0 / aOut0
    A = get_second_order_eccentricity_resonance_matrix(2 * j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out)
    b = np.zeros(2)
    js = [j,1-j,-1,0,0,0]
    b[0]  = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0)
    js = [j,1-j,0,-1,0,0]
    b[1] = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0)
    # scale X --> x
    scaleMtrx = np.diag([np.sqrt(2/Lambda0In),np.sqrt(2/Lambda0Out)])
    b = scaleMtrx @ b
    prefactor = -G**2*MOut**2*mOut**3 * ( mIn / MIn) / (Lambda0Out**2)
    return  A, prefactor * b

def get_second_order_eccentricity_resonance_matrix(j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    zs = [0,0,0,0]
    aIn0 = (Lambda0In / mIn)**2 / MIn
    aOut0 = (Lambda0Out / mOut)**2 / MOut
    alpha0 = aIn0 / aOut0
    params = dict()
    A = np.zeros((2,2))

    js = [j,2-j,-2,0,0,0]
    A[0,0]  = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0) 

    js = [j,2-j,0,-2,0,0]
    A[1,1]  = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0)

    js = [j,2-j,-1,-1,0,0]
    A[1,0] = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0) / 2
    A[0,1] = A[1,0]
    # scale X --> x
    scaleMtrx = np.diag([np.sqrt(2/Lambda0In),np.sqrt(2/Lambda0Out)])
    A = scaleMtrx @ A @ scaleMtrx
    prefactor = -G**2*MOut**2*mOut**3 * ( mIn / MIn) / (Lambda0Out**2)
    return prefactor * A

def get_second_order_inclination_resonance_matrix(j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    zs = [0,0,0,0]
    aIn0 = (Lambda0In / mIn)**2 / MIn
    aOut0 = (Lambda0Out / mOut)**2 / MOut
    alpha0 = aIn0 / aOut0
    params = dict()
    A = np.zeros((2,2))

    js = [j,2-j,0,0,-2,0]
    A[0,0]  = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0) 

    js = [j,2-j,0,0,0,-2]
    A[1,1]  = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0)

    js = [j,2-j,0,0,-1,-1]
    A[1,0] = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0) / 2
    A[0,1] = A[1,0]
    # scale X --> x
    scaleMtrx = np.diag([np.sqrt(0.5 / Lambda0In),np.sqrt(0.5 / Lambda0Out)])
    A = scaleMtrx @ A @ scaleMtrx
    prefactor = -G**2*MOut**2*mOut**3 * ( mIn / MIn) / (Lambda0Out**2)
    return prefactor * A

def get_first_order_resonance_third_order_terms_array(j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    """
    (z @ Psi3[i] @ zbar ) @ zbar + c.c. 
    """
    aIn0 = (Lambda0In / mIn)**2 / MIn
    aOut0 = (Lambda0Out / mOut)**2 / MOut
    alpha0 = aIn0 / aOut0
    Phi3 = np.zeros((2,2,2))

    # Diagonal term
    js1 = [j,1-j,-1,0,0,0]
    js2 = [j,1-j,0,-1,0,0]
    for i,zi in zip([0,1],[2,3]):
        zs = [0,0,0,0]
        zs[zi] = 1
        Phi3[0,i,i] = eval_DFCoeff_dict(DFCoeff_C(*js1,*zs),alpha0) 
        Phi3[1,i,i] = eval_DFCoeff_dict(DFCoeff_C(*js2,*zs),alpha0) 

    zs = [0,0,0,0]
    js1 = [j,1-j,-2,1,0,0]
    Phi3[0,1,0] = eval_DFCoeff_dict(DFCoeff_C(*js1,*zs),alpha0) 
    
    zs = [0,0,0,0]
    js1 = [j,1-j,1,-2,0,0]
    Phi3[1,0,1] = eval_DFCoeff_dict(DFCoeff_C(*js1,*zs),alpha0) 


    # scale X --> x
    scaleMtrx = np.diag([np.sqrt(2/Lambda0In), np.sqrt(2/Lambda0Out)]) # , np.sqrt(0.5 / Lambda0In),np.sqrt(0.5 / Lambda0Out)])
    for i in range(2):
        Phi3[i] = scaleMtrx @ Phi3[i] @ scaleMtrx
    Phi3[0] *= scaleMtrx[0,0]
    Phi3[1] *= scaleMtrx[1,1]
    prefactor = -G**2*MOut**2*mOut**3 * ( mIn / MIn) / (Lambda0Out**2)

    # return prefactor , Phi3 , scaleMtrx
    return prefactor * Phi3
