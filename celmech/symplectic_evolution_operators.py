import numpy as np
import warnings
from abc import ABC, abstractmethod
from .hamiltonian import Hamiltonian
from .disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
from .disturbing_function import terms_list_to_HamiltonianCoefficients_dict, _add_dicts, resonant_secular_contribution_dictionary,_consolidate_dictionary_terms
from .transformations import masses_to_jacobi, masses_from_jacobi
from .poincare import Poincare
from .miscellaneous import getOmegaMatrix
from scipy.linalg import solve as lin_solve
from scipy.linalg import expm
from numpy import sqrt

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
        self.mu = np.array([p.mu for p in self.state.particles[1:]]) 
        self.M = np.array([p.M for p in self.state.particles[1:]]) 
        self.GGMMmmm = (self.G*self.M)**2 * self.mu**3

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

    
def get_res_chain_reference_Lambdas_and_semimajor_axes(pvars,slist):
    coeffs = np.zeros(pvars.N)
    alpha_inv = np.zeros(pvars.N)
    alpha_inv[1] = 1    
    coeffs[1] = 1 + slist[0]
    ps = pvars.particles
    mu = np.array([p.mu for p in ps])
    M = np.array([p.M for p in ps])
    tot = coeffs[1] * mu[1] * sqrt(M[1])
    for i in range(2,len(coeffs)):
        coeffs[i] = coeffs[i-1] * slist[i-2] / (1 + slist[i-2])
        alpha_inv[i] = alpha_inv[i-1]  * ((1 + slist[i-2]) / slist[i-2])**(2/3)
        tot += coeffs[i] * mu[i] * sqrt(M[i] * alpha_inv[i])        
    Lvals = np.append([0],[p.Lambda for p in ps[1:]])
    C = coeffs @ Lvals
    a10 = (C / tot)**2
    a0 = a10 * alpha_inv 
    L0 = mu * sqrt(M * a0)
    return a0,L0

def get_first_order_eccentricity_resonance_matrix_and_vector(j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    r"""
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
        Total mass of star + inner planet
    Mout : float
        Total mass of star + outer planet
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
    muIn = mIn * (MIn - mIn) / MIn
    muOut = mOut * (MOut - mOut) / MOut
    aIn0 = (Lambda0In / muIn)**2 / MIn / G
    aOut0 = (Lambda0Out / muOut)**2 / MOut / G
    alpha0 = aIn0 / aOut0
    aOut_inv = G*MOut*muOut*muOut / LambdaOut / LambdaOut  
    prefactor = -G * mIn * mOut * aOut_inv
    A = get_second_order_eccentricity_resonance_matrix(2 * j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out)
    b = np.zeros(2)
    js = [j,1-j,-1,0,0,0]
    b[0]  = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0)
    js = [j,1-j,0,-1,0,0]
    b[1] = eval_DFCoeff_dict(DFCoeff_C(*js,*zs),alpha0)
    # scale X --> x
    scaleMtrx = np.diag([sqrt(2/Lambda0In),sqrt(2/Lambda0Out)])
    b = scaleMtrx @ b
    return  A, prefactor * b

def get_second_order_eccentricity_resonance_matrix(j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    zs = [0,0,0,0]
    muIn = mIn * (MIn - mIn) / MIn
    muOut = mOut * (MOut - mOut) / MOut
    aIn0 = (Lambda0In / muIn)**2 / MIn / G
    aOut0 = (Lambda0Out / muOut)**2 / MOut / G
    alpha0 = aIn0 / aOut0
    aOut_inv = G*MOut*muOut*muOut / LambdaOut / LambdaOut  
    prefactor = -G * mIn * mOut * aOut_inv
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
    return prefactor * A

def get_second_order_inclination_resonance_matrix(j,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out):
    zs = [0,0,0,0]
    muIn = mIn * (MIn - mIn) / MIn
    muOut = mOut * (MOut - mOut) / MOut
    aIn0 = (Lambda0In / muIn)**2 / MIn / G
    aOut0 = (Lambda0Out / muOut)**2 / MOut / G
    alpha0 = aIn0 / aOut0
    aOut_inv = G*MOut*muOut*muOut / LambdaOut / LambdaOut  
    prefactor = -G * mIn * mOut * aOut_inv
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
    
    return prefactor * A

