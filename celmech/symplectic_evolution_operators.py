import numpy as np
import warnings
from abc import ABC, abstractmethod
from .hamiltonian import Hamiltonian
from .disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
from .disturbing_function import terms_list_to_HamiltonianCoefficients_dict, _add_dicts, resonant_secular_contribution_dictionary
from .transformations import masses_to_jacobi, masses_from_jacobi
from .poincare import Poincare
from .miscellaneous import getOmegaMatrix
from scipy.linalg import solve as lin_solve
from scipy.linalg import expm
from numpy import sqrt

_rt2 = sqrt(2)
_rt2_inv = 1 / _rt2 
_machine_eps = np.finfo(np.float64).eps
# Runge-Kutte Butcher tableaus
_ImplicitMidpoint = {
        'a':np.array([[0.5]]),
        'c':np.array([0.5]),
        'b':np.array([1])
        }
_LobattoIIIB = {
    'a':np.array([
        [1/6, -1/6, 0],
        [1/6, 1/3, 0],
        [1/6, 5/6, 0]
    ]),
    'c':np.array([0, 1/2, 1]),
    'b':np.array([1/6, 2/3, 1/6])
}

_GL4 = {
    'a':np.array([
        [1/4, 1/4-1/6*sqrt(3)],
        [ 1/4+1/6*sqrt(3), 1/4]
    ]),
    'c':np.array([1/2 - 1/6*sqrt(3),1/2 + 1/6*sqrt(3)]),
    'b':np.array([1/2,1/2])
}

_GL6 = {
    'a':np.array([
        [5/36,2/9-1/15*sqrt(15),5/36-1/30*sqrt(15)],
        [5/36 + 1/24*sqrt(15),2/9,5/36 - 1/24*sqrt(15)],
        [5/36 + 1/30*sqrt(15),   2/9 + 1/15*sqrt(15),5/36]
    ]),
    'c':np.array([1/2 - 1/10*sqrt(15), 1/2, 1/2 + 1/10*sqrt(15)]),
    'b':np.array([5/18,  4/9,  5/18])
}
_rk_methods = {'ImplicitMidpoint':_ImplicitMidpoint,'LobattoIIIB':_LobattoIIIB,'GL4':_GL4,'GL6':_GL6}

class EvolutionOperator(ABC):
    def __init__(self,initial_state,dt):
        self._dt = dt
        self.state = initial_state
        self.particles = initial_state.particles
    def _state_vector_to_individual_vectors(self,state_vec):
        return np.reshape(state_vec,(-1,6))

    @abstractmethod
    def apply(self):
        pass
    
    @abstractmethod
    def apply_to_state_vector(self,state_vec):
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

    def apply_to_state_vector(self,state_vector):
        vecs = self._state_vector_to_individual_vectors(state_vector)
        L = vecs[:,2]
        lambda_dot = self.GGMMmmm/ L / L /L
        dlambda = self.dt * lambda_dot
        vecs[:,3] += dlambda
        return vecs.reshape(-1)

    
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
    def indices(self):
        return self.indexIn-1,self.indexOut-1
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
    def _RS_to_rhosigma(self,R,S):
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
        rho1,sigma1 = self._RS_to_rhosigma(R1,S1)
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

    def apply_to_state_vector(self,state_vector):
        vecs = self._state_vector_to_individual_vectors(state_vector)
        Lambdas = vecs[self.indices,2]
        lambdas = vecs[self.indices,3]
        sigma,rho = vecs[self.indices,4],vecs[self.indices,5]
        Qs0 = 0.5 *  (rho**2 + sigma**2)
        C1 = self.Lambdas_vec @ Lambdas
        C2 = np.sum(Lambdas) - np.sum(Qs0)
        theta = self.res_vec @ lambdas
        s,c = np.sin(theta),np.cos(theta)
        R = self.Ttr @ rho
        S = self.Ttr @ sigma
        R1,S1 = self._RSmap(R,S,s,c)
        rho1,sigma1=self._RS_to_rhosigma(R1,S1)
        vecs[self.indices,4] = sigma1
        vecs[self.indices,5] = rho1
        Qs1 = 0.5 * (rho1**2 + sigma1**2)
        Lambdas1 = self.Lambdas_Mtrx @ np.array([ C1, C2 + np.sum(Qs1) ])
        vecs[self.indices,2] = Lambdas1
        return vecs.reshape(-1)
        
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
    @property
    def indices(self):
        return self.indexIn-1,self.indexOut-1

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
        H = self.Ttr @ (eta - 0.5 * _rt2 * self.Ainv_dot_b * s)
        K = self.Ttr @ (kappa + 0.5 * _rt2 * self.Ainv_dot_b * c)
        return H,K

    def _HK_to_etakappa(self,H,K,s,c):
        h = self.T @ H + 0.5 * _rt2 * self.Ainv_dot_b * s
        k = self.T @ K - 0.5 * _rt2 * self.Ainv_dot_b * c
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

    def apply_to_state_vector(self,state_vector):
        vecs = self._state_vector_to_individual_vectors(state_vector)
        Lambdas = vecs[self.indices,2]
        lambdas = vecs[self.indices,3]
        theta = self.res_vec @ lambdas
        kappa,eta = vecs[self.indices,0],vecs[self.indices,1]
        Gammas0 = 0.5 *  (kappa**2 + eta**2)
        C1 = self.Lambdas_vec @ Lambdas
        C2 = np.sum(Lambdas) - np.sum(Gammas0)
        s,c = np.sin(theta),np.cos(theta)
        H = self.Ttr @ (eta - 0.5 * _rt2 * self.Ainv_dot_b * s)
        K = self.Ttr @ (kappa + 0.5 * _rt2 * self.Ainv_dot_b * c)
        H1,K1  = self._HKmap(H,K,s,c)
        eta1,kappa1 = self._HK_to_etakappa(H1,K1,s,c)
        vecs[self.indices,0] = kappa1
        vecs[self.indices,1] = eta1
        Gammas1 = 0.5 * (kappa1**2 + eta1**2)
        Lambdas1 = self.Lambdas_Mtrx @ np.array([ C1, C2 + np.sum(Gammas1) ])
        vecs[self.indices,2] = Lambdas1
        return vecs.reshape(-1)

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


from .poisson_series import DFTermSeries
from scipy.optimize import root
from .disturbing_function import ResonanceTermsList, SecularTermsList

class MeanMotionResonanceDFTermsEvolutionOperator(EvolutionOperator):
    """
    Evolution operator for a collection of disturbing function
    terms associated with a specific mean motion resonance between
    a pair of planets. 
    
    The evolution of inclination/eccentricity variables are
    computed by the implicit midpoint method. This is a second order
    symplectic method. The evolution of semi-major axes are computed
    so that both the planet pair's angular momentum and the conserved
    quantity associated with the mean motion resonance are conserved.
    The planets' mean longtiudes do not evolve.

    Arguments
    ---------
    initial state : celmech.Poincare
        Poincare variables state from which the operator is initialized.
    dt : float
        Time-step of the operator.
    j : int
        Integer determining the resonance as the j:j-k resonance.
    k : int
        The order of the resonance.
    terms_list : list
        List of distrubing function terms to include.
        List items are tuples of the form (kvec, zvec).
    indexIn : int
        Index of the inner planet particle in the resonance.
    indexOut : int
        Index of the outer planet particle in the resonance.
    Lambda0 : array-like
        The Poincare momenta Lambda are treated as constant
        in the disturing function. Lambda0 sets the values
        of these constant momenta and should be an array
        with an entry for each particle in the system.
        If no value is supplied, initial values are chosen
        at the center of exact resonance using conservation of
        angular momentum and (j-k) * LambdaOut + j * LambdaIn
        presuming only two planets are in the system.
    """

    def __init__(self, initial_state, dt, j, k, terms_list, indexIn=1, indexOut=2, Lambda0=None,**kwargs):

        pIn = initial_state.particles[indexIn]
        pOut = initial_state.particles[indexOut]
        self.indexIn = indexIn
        self.indexOut = indexOut
        self.mIn = pIn.m
        self.MIn = pIn.M
        self.mOut = pOut.m
        self.MOut = pOut.M
        self.G = initial_state.G
        
        if Lambda0 is None:
            _pvars = Poincare(initial_state.G,[pIn,pOut])
            _,Lambda0 = get_res_chain_reference_Lambdas_and_semimajor_axes(_pvars,[(j-k)/k])
            _,self.Lambda0In,self.Lambda0Out = Lambda0
        else:
            self.Lambda0In = Lambda0[indexIn]
            self.Lambda0Out = Lambda0[indexOut]

        self.rtLambda0_inv = 1 / sqrt([self.Lambda0In,self.Lambda0Out])
        self.qp_to_XY_factors = np.concatenate((self.rtLambda0_inv,0.5*self.rtLambda0_inv))
        self.DF_prefactor = -self.G**2*self.MOut**2*self.mOut**3 * ( self.mIn / self.MIn) / (self.Lambda0Out**2)
        self._dt = dt
        self._h = self.DF_prefactor * self._dt
        self.alpha = ((j-k)/j)**(2/3)
        self.terms_list = terms_list 
        self.DFSeries = DFTermSeries.from_resonance_list(
                self.terms_list,
                self.alpha,
                [self.Lambda0In,self.Lambda0Out]
        )
        self.res_vec = np.array([k - j , j])
        self.Lambdas_vec = np.array([-1 * self.res_vec[1] , self.res_vec[0]])
        self.Lambdas_Mtrx = np.linalg.inv([ self.Lambdas_vec , [1,1]])
    
    @classmethod
    def from_Order_Range(cls, initial_state, dt, j, k, Nmin, Nmax, indexIn=1, indexOut=2, Lambda0=None, include_secular_terms = False):
        terms = ResonanceTermsList(j,k,Nmin,Nmax)
        if include_secular_terms:
            terms += SecularTermsList(Nmin,Nmax)
        return cls(initial_state, dt, j, k, terms, indexIn, indexOut, Lambda0)

    @property
    def dt(self):
        return super().dt

    @dt.setter
    def dt(self,value):
        self._dt = value
        self._h = self.DF_prefactor * self._dt 

    @property
    def indices(self):
        return self.indexIn-1,self.indexOut-1

    def state_vec_to_qp_vec_and_lambdaLambda(self,state_vec):
        vecs = self._state_vector_to_individual_vectors(state_vec)
        kappa = vecs[self.indices,0] 
        eta = vecs[self.indices,1] 
        sigma = vecs[self.indices,4] 
        rho = vecs[self.indices,5] 
        lambdas = vecs[self.indices,3]
        Lambdas = vecs[self.indices,2]
        
        return np.concatenate((eta,rho,kappa,sigma)),lambdas,Lambdas

    def deriv_and_jacobian_from_qp_vec(self,qp_vec,lambda_arr):  
        XYvec = (qp_vec[4:] - 1j * qp_vec[:4]) * self.qp_to_XY_factors
        _,qp_dot,qp_dotJac,_ = self.DFSeries._evaluate_with_jacobian(lambda_arr,XYvec)
        return qp_dot, qp_dotJac

    def implicit_midpoint_f_and_Df(self,qp_vec1,qp_vec0,lambda_arr):
        h = self._h
        qp_vec_mid = 0.5 * (qp_vec1+qp_vec0)
        qp_dot,qp_dotJac = self.deriv_and_jacobian_from_qp_vec(qp_vec_mid,lambda_arr)
        f = qp_vec1 - qp_vec0 - h * qp_dot
        Df = np.eye(8) - 0.5 * h * qp_dotJac
        return f,Df

    def implicit_midpoint_step(self,qp_vec,lambda_arr):
        rt = root(
                self.implicit_midpoint_f_and_Df,
                qp_vec,
                args = (qp_vec,lambda_arr),
                jac=True
            )
        return rt.x

    def apply(self):
        pass

    def apply_to_state_vector(self,state_vec):
        vecs = self._state_vector_to_individual_vectors(state_vec)
        kappa = vecs[self.indices,0] 
        eta = vecs[self.indices,1] 
        sigma = vecs[self.indices,4] 
        rho = vecs[self.indices,5] 
        lambdas = vecs[self.indices,3]
        Lambdas = vecs[self.indices,2]
        qp_vec =  np.concatenate((eta,rho,kappa,sigma))
        AMD0 = 0.5 * np.sum(qp_vec**2)
        C1 = self.Lambdas_vec @ Lambdas
        C2 = np.sum(Lambdas) - AMD0
        qp_vec_new = self.implicit_midpoint_step(qp_vec,lambdas)
        AMD1 = 0.5 * np.sum(qp_vec_new**2)
        Lambdas_new = self.Lambdas_Mtrx @ np.array([C1, C2 + np.sum(AMD1)])
        
        vecs[self.indices,2] = Lambdas_new

        # eta
        vecs[self.indices,1] = qp_vec_new[0:2]  

        # kappa
        vecs[self.indices,0] = qp_vec_new[4:6]  
        
        # rho
        vecs[self.indices,5] = qp_vec_new[2:4]  
        
        # sigma
        vecs[self.indices,4] = qp_vec_new[6:8]  
        
        return vecs.reshape(-1)

class SecularDFTermsEvolutionOperator(EvolutionOperator):
    """
    Evolution operator for a collection of secular disturbing function
    terms.  
    The evolution of inclination/eccentricity variables are
    computed by the implicit midpoint method. This is a second order
    symplectic method. 

    Arguments
    ---------
    initial state : celmech.Poincare
        Poincare variables state from which the operator is initialized.
    dt : float
        Time-step of the operator.
    terms_dict : dictionary
        Dictionary keys are given in the form (iIn,iOut) where 
        iIn and iIout are the indices of the inner and outer planet.
        Each dictionary entry contains either: 
            - A list of distrubing function terms to include 
            for a given pair where list items are tuples of the form 
            (kvec, zvec). Disturbing function term values are computed
            automatically. 
            - A dictionary of terms to include with keys in the form
            (kvec,zvec) and values giving the coefficient value for the
            disturbing function term. 
    Lambda0 : array-like
        The Poincare momenta Lambda are treated as constant
        in the disturing function. Lambda0 sets the values
        of these constant momenta and should be an array
        with an entry for each particle in the system.
        If no value is supplied, initial values are chosen.
    """
    def __init__(
            self,
            initial_state, 
            dt, 
            terms_dict, 
            Lambda0=None,
            rtol = _machine_eps, atol = 0.0, max_iter = 10,
            rkmethod='LobattoIIIB',rk_root_method='Newton'
    ):
        # Set Lambda0 constants in secular Hamiltonian.
        if Lambda0 is None:
            Lambda0 = [p.Lambda for p in initial_state.particles]
        self.rtLambda0_inv = 1 / sqrt(Lambda0[1:])
        self.qp_to_XY_factors = np.concatenate((self.rtLambda0_inv,0.5*self.rtLambda0_inv))

        self._dt = dt
        
        self.N = initial_state.N
        self.Npl = self.N - 1
        self.Ndim = 4 * self.Npl
        G = initial_state.G
        ps = initial_state.particles

        # Set tolerance and iteration parameters
        tols_allowed = atol >=0 and rtol >=0
        tols_allowed = tols_allowed and (atol > 0 or rtol > 0)
        assert tols_allowed, "Tolerances must be non-negative and at least one tolerance must be positive." 
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        self._rk_root_method=rk_root_method
        if rk_root_method == 'Newton':
            self.implicit_rk_step = self._implicit_rk_step_jacobian
        elif rk_root_method == 'fixed_point':
            self.implicit_rk_step = self._implicit_rk_step_fixed_point
        else:
            raise ValueError("'rk_root_method' must be either 'Newton' or 'fixed_point'")
        # Generate DFTermsSeries objects for each planet pair
        self.DFSeries_dict = dict()       
        for iPair, pair_terms in terms_dict.items():
            iIn,iOut = iPair
            assert iIn < iOut, "Dictionary keys must have iIn < iOut."
            pIn = ps[iIn]
            pOut = ps[iOut]
            Lambda0Out = Lambda0[iOut] 
            Lambda0In = Lambda0[iIn] 
            MOut = pOut.M
            MIn = pIn.M
            mOut = pOut.m
            mIn = pIn.m
            if type(pair_terms) is list:
                all_secular = np.alltrue( [x[0][0] == 0 and x[0][1] == 0 for x in pair_terms] )
                assert all_secular, "Only secular terms may be inlcuded in the DF term lists."
                dfseries = DFTermSeries.from_resonance_list(pair_terms,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out)
            elif type(pair_terms) is dict:
                all_secular = np.alltrue( [x[0][0] == 0 and x[0][1] == 0 for x in pair_terms.keys()] )
                assert all_secular, "Only secular terms may be inlcuded in the DF term lists."
                dfseries = DFTermSeries(pair_terms,Lambda0In,Lambda0Out)
            else:
                raise ValueError("'terms_dict' entry for pair ({0},{1}) is type {2}. Entries must be either lists or dictionaries!".format(iIn,iOut,type(pair_terms)))
            self.DFSeries_dict[iPair] = dfseries

        # Set Runge-Kutta integration parameters.
        self.rkmethod = rkmethod
    
    @property 
    def rk_root_method(self):
        return self._rk_root_method
    @rk_root_method.setter
    def rk_root_method(self,rk_root_method):
        if rk_root_method == 'Newton':
            self.implicit_rk_step = self._implicit_rk_step_jacobian
        elif rk_root_method == 'fixed_point':
            self.implicit_rk_step = self._implicit_rk_step_fixed_point
        else:
            raise ValueError("'rk_root_method' must be either 'Newton' or 'fixed_point'")
        self._rk_root_method = rk_root_method

    @property
    def rkmethod(self):
        return self._rkmethod

    @rkmethod.setter
    def rkmethod(self,rkmethod):
        if rkmethod in _rk_methods.keys():
            self._rkmethod = rkmethod
            self._rk_tableau=_rk_methods[rkmethod]
            self.rk_a= self._rk_tableau['a']
            self.rk_b= self._rk_tableau['b']
            self.rk_c= self._rk_tableau['c']
            self.rk_s = len(self.rk_c)
        else:
            methods_list = "\n".join(["\t{:s}".format(method) for method in _rk_methods.keys()])
            raise ValueError("'rkmethod' must be one of:\n" + methods_list) 

    @classmethod
    def fromOrderRange(cls,initial_state, dt,Nmin,Nmax,resonances_info={}, **kwargs):
        """
        Initialize operator that includes all eccentricity and
        inclination terms with orders ranging from Nmin to Nmax.
        Interactions for all planet pairs are included.

        Arguments
        ---------
        initial_state : celmech.Poincare
          System to intialize operator for.
        dt : float
          Operator timestep.
        Nmin : int
          Minimum order of secular terms to include.
        Nmax : int
          Maximum order of secular terms to include.
        resonant_terms_dict : dict
          A dictionary specifying a list of mu^2
          resonant contributions to the secular dynamics
          to include
        Lambda0 : array-like
            The Poincare momenta Lambda are treated as constant
            in the disturing function. Lambda0 sets the values
            of these constant momenta and should be an array
            with an entry for each particle in the system.
            If no value is supplied, initial values are chosen.

        Returns
        -------
        operator : SecularDFTermsEvolutionOperator
        """
        terms = SecularTermsList(Nmin,Nmax)
        N = initial_state.N
        terms_dict = {}
        for iOut in range(1,N):
            for iIn in range(1,iOut):
                terms_dict[(iIn,iOut)] = terms

        # Add any resonant contrbutions to the secular dynamics
        Lambda0 = kwargs.get('Lambda0',[p.Lambda for p in initial_state.particles])
        ps = initial_state.particles
        G = initial_state.G

        for key,res_jk_list in resonances_info.items():
            iIn,iOut = key
            pIn,pOut = ps[iIn],ps[iOut]
            mIn,mOut = pIn.m,pOut.m
            MIn,MOut = pIn.M,pOut.M
            Lambda0In,Lambda0Out = Lambda0[iIn],Lambda0[iOut]
            sec_terms = terms_dict[key]
            extra_args = G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out
            dsec = terms_list_to_HamiltonianCoefficients_dict(sec_terms,*extra_args)
            for j,k in res_jk_list:
                dres = resonant_secular_contribution_dictionary(j,k,Nmin,Nmax,*extra_args)
                dsec = _add_dicts(dsec,dres)
            terms_dict[key] = dsec

        return cls(initial_state, dt, terms_dict,**kwargs) 

    @property
    def dt(self):
        return super().dt

    @dt.setter
    def dt(self,value):
        self._dt = value

    def state_vec_to_qp_vec(self,state_vec):
        """
        Convert full state vector to vector of variables
        that enter in the secular equations.

        Arguments
        ---------
        state_vec :  array-like
          Full state vector of the system in Poincare 
          variables.   

        Returns
        -------
        qp_vec : ndarray
         Vector containing eccentricity and inclination
         variables eta,rho,kappa,sigma. The variables 
         are returned in the order:
          [eta1,eta2,...,etaN,rho1,...,rhoN,kappa1,...,kappaN,sigma1,...sigmaN]
        """
        vecs = self._state_vector_to_individual_vectors(state_vec)
        kappa = vecs[:,0] 
        eta = vecs[:,1] 
        sigma = vecs[:,4] 
        rho = vecs[:,5] 
        return np.concatenate((eta,rho,kappa,sigma))

    def Hamiltonian_from_qp_vec(self,qp_vec):  
        """
        Compute Hamiltonian of the operator from the 
        equations of motion for the 'qp_vec' variables 
        returned by method 'state_vec_to_qp_vec'.

        Arguments
        ---------
        qp_vec : ndarray
          Input variable vector in the from
           [eta1,eta2,...,etaN,rho1,...,rhoN,kappa1,...,kappaN,sigma1,...sigmaN]

        Returns
        -------
        Hamiltonian : float
          The value of the Hamiltonian (i.e., the sum of the disturbing
          function terms modeled by the operator)
        """
        l = np.zeros(2)
        eta,rho,kappa,sigma = qp_vec.reshape(-1,self.Npl)
        H = eta * self.rtLambda0_inv
        K = kappa * self.rtLambda0_inv
        R = 0.5 * rho * self.rtLambda0_inv
        S = 0.5 * sigma * self.rtLambda0_inv
        Hamiltonian = 0.
        for iPair,series in self.DFSeries_dict.items():
            iIn, iOut = iPair
            indices = np.array(iPair) - 1
            X = K[indices] - 1j * H[indices]
            Y = S[indices] - 1j * R[indices]
            XYvec = np.concatenate((X,Y))
            dH = series._evaluate(l,XYvec)
            Hamiltonian += dH
        return Hamiltonian

    def deriv_from_qp_vec(self,qp_vec):  
        """
        Compute the time derivatives from the 
        equations of motion for the 'qp_vec' 
        variables returned by method 'state_vec_to_qp_vec'.

        Arguments
        ---------
        qp_vec : ndarray
          Input variable vector in the from
           [eta1,eta2,...,etaN,rho1,...,rhoN,kappa1,...,kappaN,sigma1,...sigmaN]

        Returns
        -------
        qp_vec_dot : ndarray
          Time derivative of qp_vec.
        """
        derivs = np.zeros(self.Ndim) 
        l = np.zeros(2)
        eta,rho,kappa,sigma = qp_vec.reshape(-1,self.Npl)
        H = eta * self.rtLambda0_inv
        K = kappa * self.rtLambda0_inv
        R = 0.5 * rho * self.rtLambda0_inv
        S = 0.5 * sigma * self.rtLambda0_inv
        for iPair,series in self.DFSeries_dict.items():
            iIn, iOut = iPair
            indices = np.array(iPair) - 1
            X = K[indices] - 1j * H[indices]
            Y = S[indices] - 1j * R[indices]
            XYvec = np.concatenate((X,Y))
            _, _deriv= series._evaluate_with_derivs(l,XYvec)
            index_list = np.array([
                iIn,iOut,
                iIn + self.Npl,iOut + self.Npl,
                iIn + 2*self.Npl,iOut + 2*self.Npl,
                iIn + 3*self.Npl,iOut + 3*self.Npl
                ]) - 1
            for i,I in enumerate(index_list):
                derivs[I] += _deriv[i]
        return derivs

    def deriv_and_jacobian_from_qp_vec(self,qp_vec):  
        """
        Compute the time derivatives and Jacobian from the 
        equations of motion for the 'qp_vec' variables returned 
        by method 'state_vec_to_qp_vec'.

        Arguments
        ---------
        qp_vec : ndarray shape (4 * Nplanet,)
          Input variable vector in the from
           [eta1,eta2,...,etaN,rho1,...,rhoN,kappa1,...,kappaN,sigma1,...sigmaN]

        Returns
        -------
        qp_vec_dot : ndarray, shape (4 * Nplanet,)
          Time derivative of qp_vec.

        qp_vec_dot_jac : ndarray, shape (4 * Nplanet, 4 * Nplanet)
          Jacobian matrix of the equations of motion for the
          variables contained in qp_vec.
        """
        derivs = np.zeros(self.Ndim) 
        jac = np.zeros((self.Ndim,self.Ndim)) 
        l = np.zeros(2)
        eta,rho,kappa,sigma = qp_vec.reshape(-1,self.Npl)
        H = eta * self.rtLambda0_inv
        K = kappa * self.rtLambda0_inv
        R = 0.5 * rho * self.rtLambda0_inv
        S = 0.5 * sigma * self.rtLambda0_inv
        for iPair,series in self.DFSeries_dict.items():
            iIn, iOut = iPair
            indices = np.array(iPair) - 1
            X = K[indices] - 1j * H[indices]
            Y = S[indices] - 1j * R[indices]
            XYvec = np.concatenate((X,Y))
            _, _deriv, _jac, _ = series._evaluate_with_jacobian(l,XYvec)
            index_list = np.array([
                iIn,iOut,
                iIn + self.Npl,iOut + self.Npl,
                iIn + 2*self.Npl,iOut + 2*self.Npl,
                iIn + 3*self.Npl,iOut + 3*self.Npl
                ]) - 1
            for i,I in enumerate(index_list):
                derivs[I] += _deriv[i]
                for j,J in enumerate(index_list):
                    jac[I,J] += _jac[i,j]
        return derivs, jac

    def _rk4_step(self,y,ydot,h):
        f = self.deriv_from_qp_vec
        h_by_2 = 0.5 * h
        k1 = ydot
        y2 = y + h_by_2 * k1
        k2 = f(y2)
        y3 = y + h_by_2 * k2
        k3 = f(y3)
        y4 = y + h * k3
        k4 = f(y4)
        yout = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6.
        return yout,f(yout)

    def _implicit_rk_step_fixed_point(self,qp_vec):
        """
        Advance ODE for input qpve for a timestep h
        using an implicit Runge-Kutta method defined by the
        Butcher tableau [a,b,c].

        Arguments
        ---------
        qp_vec
        """
        h = self._dt
        a = self.rk_a
        b = self.rk_b
        c = self.rk_c
        f = self.deriv_from_qp_vec
        y = qp_vec
        ktemp = f(qp_vec)
        max_iter = self.max_iter
        k = np.zeros((self.rk_s,self.Ndim))
        rtol = self.rtol
        atol = self.atol
        # 
        for i,ci in enumerate(c):
            if ci == 0:
                k[i,:] = ktemp 
            else:
                _,k[i,:] = self._rk4_step(y,ktemp,ci*h)
        for itr in xrange(max_iter):
            k_old = k.copy()
            ytemps = y + h * a @ k 
            k = np.array([f(ytemp) for ytemp in ytemps])
            delta_ks = np.abs(k-k_old)
            if np.alltrue( delta_ks < rtol * np.abs(k_old) + atol ):
                break
        else:
            warnings.warn("'implicit_rk_step' reached maximum number of iterations ({})".format(max_iter))
        ynew = y + h * b @ k
        return ynew

    def _implicit_rk_step_jacobian(self,qp_vec):
        """
        Advance ODE for input qpve for a timestep h
        using an implicit Runge-Kutta method defined by the
        Butcher tableau [a,b,c].

        Arguments
        ---------
        qp_vec
        """
        h = self._dt
        a = self.rk_a
        b = self.rk_b
        c = self.rk_c
        s = self.rk_s
        Ndim = self.Ndim
        f = self.deriv_from_qp_vec
        f_and_Df = self.deriv_and_jacobian_from_qp_vec
        y = qp_vec
        
        max_iter = self.max_iter
        k = np.zeros((s,Ndim))
        Dkdy = np.zeros((s,Ndim,Ndim))
        rtol = self.rtol
        atol = self.atol
        Imtrx = np.eye(s * Ndim)
        #
        
        # Generate initial guesses from second-order approx.
        ####################################################
        # This appears to be sligtly faster than
        # than generating guesses via RK4, as in
        #  _implicit_rk_step_fix_point, but I haven't
        # tested it extensively.
        ####################################################
        ktemp,Dktemp = f_and_Df(qp_vec)
        d2ydt2 = Dktemp @ ktemp
        k = np.array([ktemp + ci*h*d2ydt2 for ci in c])

        # Main loop
        K = np.hstack(k)
        for itr in xrange(max_iter):
            ytemps = y + h * a @ k 
            for i,ytemp in enumerate(ytemps):
                k[i],Dkdy[i] = f_and_Df(ytemp)
            fOfK=np.hstack(k)
            g = K - fOfK
            Dg = Imtrx - h * np.block([[ a[i,j] * Dkdy[i] for j in range(s)] for i in range(s)])
            dK = lin_solve(Dg,-1*g)
            K+=dK
            k=K.reshape(s,Ndim)
            if np.alltrue( np.abs(dK) < rtol * np.abs(K) + atol ):
                break
        else:
            warnings.warn("'implicit_rk_step' reached maximum number of iterations ({})".format(max_iter))
        ynew = y + h * b @ k
        return ynew

    def apply(self):
        warnings.warn("'SecularDFTermsEvolutionOperator.apply' method not implemented.")
        pass


    def apply_to_state_vector(self,state_vec):
        """
        Apply evolution operator to state vector.

        Arguments
        ---------
        state_vec : ndarray
          State vector of system in Poincare variables
           [ 
            kappa1,eta1,Lambda1,lambda1,sigma1,rho1,
            ...,
            kappaN,etaN,LambdaN,lambdaN,sigmaN,rhoN
           ]
        Returns
        -------
        state_vec : ndarray
          Updated state vector of the system.
        """
        qp_vec = self.state_vec_to_qp_vec(state_vec)
        qp_vec_new = self.implicit_rk_step(qp_vec)
        for i in xrange(self.Npl):
            # eta
            state_vec[6*i+1] = qp_vec_new[i]
            
            # kappa
            state_vec[6*i] = qp_vec_new[i + 2 * self.Npl]
            
            # rho
            state_vec[6*i+5] = qp_vec_new[i + self.Npl]

            # sigma
            state_vec[6*i+4] = qp_vec_new[i + 3 * self.Npl]
        return state_vec
    
    def calculate_Hamiltonian(self,state_vec):
        """
        Compute the value of the Hamiltonian components
        of this operator for an input state vector.

        Arguments
        ---------
        state_vec : ndarray
          State vector of system in Poincare variables
           [ 
            kappa1,eta1,Lambda1,lambda1,sigma1,rho1,
            ...,
            kappaN,etaN,LambdaN,lambdaN,sigmaN,rhoN
           ]
        Returns
        -------
        Hamiltonian : float
          Value of the Hamiltonian.
        """
        qp_vec = self.state_vec_to_qp_vec(state_vec)
        return self.Hamiltonian_from_qp_vec(qp_vec)

def get_res_chain_reference_Lambdas_and_semimajor_axes(pvars,slist):
    coeffs = np.zeros(pvars.N)
    alpha_inv = np.zeros(pvars.N)
    alpha_inv[1] = 1    
    coeffs[1] = 1 + slist[0]
    ps = pvars.particles
    m = np.array([p.m for p in ps])
    M = np.array([p.M for p in ps])
    tot = coeffs[1] * m[1] * sqrt(M[1])
    for i in range(2,len(coeffs)):
        coeffs[i] = coeffs[i-1] * slist[i-2] / (1 + slist[i-2])
        alpha_inv[i] = alpha_inv[i-1]  * ((1 + slist[i-2]) / slist[i-2])**(2/3)
        tot += coeffs[i] * m[i] * sqrt(M[i] * alpha_inv[i])        
    Lvals = np.append([0],[p.Lambda for p in ps[1:]])
    C = coeffs @ Lvals
    a10 = (C / tot)**2
    a0 = a10 * alpha_inv 
    L0 = m * sqrt(M * a0)
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
    scaleMtrx = np.diag([sqrt(2/Lambda0In),sqrt(2/Lambda0Out)])
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
    scaleMtrx = np.diag([sqrt(2/Lambda0In),sqrt(2/Lambda0Out)])
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
    scaleMtrx = np.diag([sqrt(0.5 / Lambda0In),sqrt(0.5 / Lambda0Out)])
    A = scaleMtrx @ A @ scaleMtrx
    prefactor = -G**2*MOut**2*mOut**3 * ( mIn / MIn) / (Lambda0Out**2)
    return prefactor * A

