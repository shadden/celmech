import numpy as np
import warnings
from . import Poincare
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, atan2, expand_trig,diff,Matrix
from .disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
from scipy.linalg import expm
from .poincare import single_true
from .rk_integrator import RKIntegrator, _rk_methods 
from scipy.linalg import expm
from celmech.miscellaneous import getOmegaMatrix, _machine_eps

from celmech.disturbing_function import SecularTermsList
from celmech.disturbing_function import DFCoeff_C, _add_dicts,_consolidate_dictionary_terms
from celmech.disturbing_function import terms_list_to_HamiltonianCoefficients_dict
from celmech.disturbing_function import resonant_secular_contribution_dictionary
from celmech.poisson_series import DFTermSeries



class LaplaceLagrangeSystem(Poincare):
    r"""
    A class for representing the classical Laplace-Lagrange secular
    solution for a planetary system.
    
    Attributes
    ----------
    eccentricity_matrix : sympy.Matrix
      The matrix :math:`\pmb{S}_e` appearing in the secular equations
      of motion for the eccentricity variables, 

      .. math::
            \frac{d}{dt}(\eta_i + i\kappa_i) = [\pmb{S}_e]_{ij} (\eta_j + i\kappa_j)~.
    
      or, equivalently,

      .. math::
            \frac{d}{dt}\pmb{x} = -i \pmb{S}_e \cdot \pmb{x}

      The matrix is given in symbolic form.
    inclination_matrix : sympy.Matrix
      The matrix :math:`\pmb{S}_I` appearing in the secular equations
      of motion for the eccentricity variables, 

      .. math::
            \frac{d}{dt}(\rho_i + i \sigma_i ) =[\pmb{S}_I]_{ij} (\rho_j + i \sigma_j)~.
    
      or, equivalently,

      .. math::
            \frac{d}{dt}\pmb{y} = -i \pmb{S}_I \cdot \pmb{y}

      The matrix is given in symbolic form.
    Neccentricity_matrix : ndarray
        Numerical value of the eccentricity matrix :math:`\pmb{S}_e`
    Ninclination_matrix : ndarray
        Numerical value of the inclination matrix :math:`\pmb{S}_I`
    Tsec : float
        The secular timescale of the system, defined as the shortest
        secular period among the system's inclination and eccentricity
        modes.
    eccentricity_eigenvalues : ndarray
        Array of the eccentricity mode eigenvalues  (i.e., secular frequencies)
    inclination_eigenvalues : ndarray
        Array of the inclination mode eigenvalues  (i.e., secular frequencies)
    """
    def __init__(self,G,poincareparticles=[]):
        super(LaplaceLagrangeSystem,self).__init__(G,poincareparticles)
        self.params = {S('G'):self.G}
        for i,particle in enumerate(self.particles):
            if i is not 0:
                m,mu,M,Lambda = symbols('m{0},mu{0},M{0},Lambda{0}'.format(i)) 
                self.params.update({m:particle.m,mu:particle.mu,M:particle.M,Lambda:particle.Lambda})
        self.ecc_entries  = {(j,i):S(0) for i in xrange(1,self.N) for j in xrange(1,i+1)}
        self.inc_entries  = {(j,i):S(0) for i in xrange(1,self.N) for j in xrange(1,i+1)}
        self.tol = np.min([p.m for p in self.particles[1:]]) * np.finfo(np.float).eps
        ps = self.particles[1:]
        self.eta0_vec = np.array([p.eta for p in ps])
        self.kappa0_vec = np.array([p.kappa for p in ps]) 
        self.rho0_vec = np.array([p.rho for p in ps])
        self.sigma0_vec = np.array([p.sigma for p in ps]) 
        self._update()
    @classmethod
    def from_Poincare(cls,pvars):
        """
        Initialize a Laplace-Lagrange system directly from a 
        :class:`celmech.poincare.Poincare` object.
        
        Arguments
        ---------
        pvars : :class:`celmech.poincare.Poincare` 
            Instance of Poincare variables from which to initialize
            Laplace-Lagrange system

        Returns
        -------
        system : :class:`celmech.secular.LaplaceLagrangeSystem`
        """
        return cls(pvars.G,pvars.particles[1:])

    @classmethod
    def from_Simulation(cls,sim):
        """
        Initialize a Laplace-Lagrange system directly from a 
        :class:`rebound.Simulation` object.
        
        Arguments
        ---------
        pvars : :class:`rebound.Simulation` 
            rebound simulation from which to initialize
            Laplace-Lagrange system

        Returns
        -------
        sim : :class:`celmech.secular.LaplaceLagrangeSystem`
        """
        pvars = Poincare.from_Simulation(sim)
        return cls.from_Poincare(pvars)

    @property
    def eccentricity_matrix(self):
        return Matrix([
            [self.ecc_entries[max(i,j),min(i,j)] for i in xrange(1,self.N)]
            for j in xrange(1,self.N) 
            ])

    @property
    def inclination_matrix(self):
        return Matrix([
            [self.inc_entries[max(i,j),min(i,j)] for i in xrange(1,self.N)]
            for j in xrange(1,self.N) 
            ])
    @property 
    def Neccentricity_matrix(self):
        return np.array(self.eccentricity_matrix.subs(self.params)).astype(np.float64)
    @property 
    def Ninclination_matrix(self):
        return np.array(self.inclination_matrix.subs(self.params)).astype(np.float64)
    @property
    def Tsec(self):
        Omega_e = np.max( np.abs(self.eccentricity_eigenvalues()) )
        Omega_i = np.max( np.abs(self.inclination_eigenvalues()) )
        return 2 * np.pi / max(Omega_e,Omega_i)
    def _chop(self,arr):
        arr[np.abs(arr)<self.tol] = 0
        return arr
    def eccentricity_eigenvalues(self):
        return np.linalg.eigvalsh(self.Neccentricity_matrix)
    def inclination_eigenvalues(self):
        answer = np.linalg.eigvalsh(self.Ninclination_matrix)
        return self._chop(answer)

    def secular_solution(self,times,epoch=0):
        """
        Get the solution of the Laplace-Lagrange
        secular equations of motion at the 
        user-specified times.

        Arguments
        ---------
        times : ndarray
            Array of times at which to evaluate 
            the solution to the equations of motion.
        epoch : float, optional
            Current time of system state. Default is 
            t=0.
        Returns
        -------
        soln : dict
            The solution dictionary contains various dynamical
            quantites computed at the input times. 
        """
        e_soln = self.secular_eccentricity_solution(times,epoch)
        solution = {key:val.T for key,val in e_soln.items()}
        T,D = self.diagonalize_inclination()
        R0 = T.T @ self.rho0_vec
        S0 = T.T @ self.sigma0_vec
        t1 = times - epoch
        freqs = np.diag(D)
        cos_vals = np.array([np.cos(freq * t1) for freq in freqs]).T
        sin_vals = np.array([np.sin(freq * t1) for freq in freqs]).T
        S = S0 * cos_vals - R0 * sin_vals    
        R = S0 * sin_vals + R0 * cos_vals
        rho = np.transpose(T @ R.T)
        sigma = np.transpose(T @ S.T)
        Yre = 0.5 * sigma / np.sqrt([p.Lambda for p in self.particles[1:]])
        Yim = -0.5 * rho / np.sqrt([p.Lambda for p in self.particles[1:]])
        kappa,eta = solution['kappa'],solution['eta']
        Xre = kappa / np.sqrt([p.Lambda for p in self.particles[1:]])
        Xim = -eta / np.sqrt([p.Lambda for p in self.particles[1:]])
        Ytozeta = 1 / np.sqrt(1 - 0.5 * (Xre**2 + Xim**2))
        zeta_re = Yre * Ytozeta
        zeta_im = Yim * Ytozeta
        zeta = zeta_re + 1j * zeta_im
        solution.update({
            "rho":rho,
            "sigma":sigma,
            "R":R,
            "S":S,
            "p":zeta_im,
            "q":zeta_re,
            "zeta":zeta,
            "inc":2 * np.arcsin(np.abs(zeta)),
            "Omega":np.angle(zeta)
        })
        return {key:val.T for key,val in solution.items()}

    def secular_eccentricity_solution(self,times,epoch=0):
        T,D = self.diagonalize_eccentricity()
        H0 = T.T @ self.eta0_vec
        K0 = T.T @ self.kappa0_vec
        t1 = times - epoch
        freqs = np.diag(D)
        cos_vals = np.array([np.cos(freq * t1) for freq in freqs]).T
        sin_vals = np.array([np.sin( freq * t1) for freq in freqs]).T
        K = K0 * cos_vals - H0 * sin_vals    
        H = K0 * sin_vals + H0 * cos_vals
        eta = np.transpose(T @ H.T)
        kappa = np.transpose(T @ K.T)
        Xre = kappa / np.sqrt([p.Lambda for p in self.particles[1:]])
        Xim = -eta / np.sqrt([p.Lambda for p in self.particles[1:]])
        Xtoz = np.sqrt(1 - 0.25 * (Xre**2 + Xim**2))
        zre = Xre * Xtoz
        zim = Xim * Xtoz
        solution = {
                "time":times,
                "H":H,
                "K":K,
                "eta":eta,
                "kappa":kappa,
                "k":zre,
                "h":zim,
                "z":zre + 1j * zim,
                "e":np.sqrt(zre*zre + zim*zim),
                "pomega":np.arctan2(zim,zre)
                }
        return {key:val.T for key,val in solution.items()}
    def diagonalize_eccentricity(self):
        r"""
        Solve for matrix S, that diagonalizes the
        matrix T in the equations of motion:

        .. math::

            \frac{d}{dt}(\eta + i\kappa) = i A \cdot (\eta + i\kappa)

        The matrix S satisfies

        .. math::

                T^{T} \cdot A \cdot T = D
        
        where D is a diagonal matrix.
        The equations of motion are decoupled harmonic
        oscillators in the variables (P,Q) defined by 

        .. math::

            H + i K = S^{T} \cdot (\eta + i \kappa)
        
        Returns
        -------
        (T , D) : tuple of n x n numpy arrays
        """
        vals,T = np.linalg.eigh(self.Neccentricity_matrix)
        return T, np.diag(vals)

    def diagonalize_inclination(self):
        r"""
        Solve for matrix U, that diagonalizes the
        matrix B in the equations of motion:
        
        .. math::
            \frac{d}{dt}(\rho + i\sigma) = i B \cdot (\rho + i\sigma)

        The matrix S satisfies

        .. math::
                U^{T} \cdot B \cdot U = D

        where D is a diagonal matrix.
        The equations of motion are decoupled harmonic
        oscillators in the variables :math:`(R,S)` defined by 

        .. math::
            R + i S = U^{T} \cdot (\rho + i \sigma)
        
        Returns
        -------
        (U , D) : tuple of n x n numpy arrays
        """
        vals,U = np.linalg.eigh(self.Ninclination_matrix)
        return U, self._chop(np.diag(vals))
    
    def _update(self):
        G = symbols('G')
        ecc_diag_coeff = DFCoeff_C(*[0 for _ in range(6)],0,0,1,0)
        inc_diag_coeff = DFCoeff_C(*[0 for _ in range(6)],1,0,0,0)
        js_dpomega = 0,0,1,-1,0,0
        js_dOmega = 0,0,0,0,1,-1
        ecc_off_coeff = DFCoeff_C(*js_dpomega,0,0,0,0)
        inc_off_coeff = DFCoeff_C(*js_dOmega,0,0,0,0)
        for i in xrange(1,self.N):
            for j in xrange(1,self.N):
                if j==i:
                    continue
                indexIn = min(i,j)
                indexOut = max(i,j)
                particleIn = self.particles[indexIn]
                particleOut = self.particles[indexOut]
                alpha = particleIn.a / particleOut.a
                mIn,muIn,MIn,LambdaIn = symbols('m{0},mu{0},M{0},Lambda{0}'.format(indexIn)) 
                mOut,muOut,MOut,LambdaOut = symbols('m{0},mu{0},M{0},Lambda{0}'.format(indexOut)) 
                Cecc_diag = get_DFCoeff_symbol(*[0 for _ in range(6)],0,0,1,0,indexIn,indexOut)
                Cinc_diag = get_DFCoeff_symbol(*[0 for _ in range(6)],1,0,0,0,indexIn,indexOut)
                aOut_inv = G*MOut*muOut*muOut / LambdaOut / LambdaOut  
                prefactor = -G * mIn * mOut * aOut_inv
                self.params[Cecc_diag] = eval_DFCoeff_dict(ecc_diag_coeff,alpha)
                self.params[Cinc_diag] = eval_DFCoeff_dict(inc_diag_coeff,alpha)
                if i > j:
                    particleIn = self.particles[indexIn]
                    Cecc = get_DFCoeff_symbol(*js_dpomega,0,0,0,0,indexIn,indexOut)
                    Cinc = get_DFCoeff_symbol(*js_dOmega,0,0,0,0,indexIn,indexOut)
                    alpha = particleIn.a/particleOut.a
                    assert alpha<1, "Particles must be in order by increasing semi-major axis!"
                    Necc_coeff = eval_DFCoeff_dict(ecc_off_coeff,alpha)
                    Ninc_coeff = eval_DFCoeff_dict(inc_off_coeff,alpha)
                    self.params[Cecc] = Necc_coeff
                    self.params[Cinc] = Ninc_coeff
                    ecc_entry = prefactor  * Cecc / sqrt(LambdaIn) / sqrt(LambdaOut)
                    inc_entry = prefactor  * Cinc / sqrt(LambdaIn) / sqrt(LambdaOut) / 4
                    self.ecc_entries[(indexOut,indexIn)] = ecc_entry
                    self.inc_entries[(indexOut,indexIn)] = inc_entry
                else:
                    pass
                LmbdaI = S('Lambda{}'.format(i))
                self.ecc_entries[(i,i)] += 2 * prefactor * Cecc_diag / LmbdaI
                self.inc_entries[(i,i)] += 2 * prefactor * Cinc_diag / LmbdaI / 4

    def add_first_order_resonance_terms(self,resonances_dictionary):
        for indices,resonance_k in resonances_dictionary.items():
            self.add_first_order_resonance_term(*indices,resonance_k) 

    def add_first_order_resonance_term(self,indexIn, indexOut,jres):
        """
        Include a correction to the Laplace-Lagrange 
        secular equations of motion that arise due to a nearby 
        first-order mean motion resonance between two planets.

        Note-- corrections are not valid for planets in resonance!
    
        Arguments
        ---------
        indexIn : int
            Index of inner planet near resonance.
        indexOut : int
            Index of inner planet near resonance.
        jres : int
            Specify the jres:jres-1 mean motion resonance.
        """
        assert indexIn < indexOut, "Input 'indexIn' must be less than 'indexOut'."
        particleIn = self.particles[indexIn]
        particleOut = self.particles[indexOut]
        alpha = particleIn.a / particleOut.a
        G = symbols('G')
        mIn,muIn,MIn,LambdaIn = symbols('m{0},mu{0},M{0},Lambda{0}'.format(indexIn)) 
        mOut,muOut,MOut,LambdaOut = symbols('m{0},mu{0},M{0},Lambda{0}'.format(indexOut)) 
        
        CIn = get_DFCoeff_symbol(*[jres,1-jres,-1,0,0,0],0,0,0,0,indexIn,indexOut)
        COut = get_DFCoeff_symbol(*[jres,1-jres,0,-1,0,0],0,0,0,0,indexIn,indexOut)
        self.params[CIn] = eval_DFCoeff_dict(DFCoeff_C(*[jres,1-jres,-1,0,0,0],0,0,0,0),alpha)
        self.params[COut] = eval_DFCoeff_dict(DFCoeff_C(*[jres,1-jres,0,-1,0,0],0,0,0,0),alpha)
        aOut_inv = G*MOut*muOut*muOut / LambdaOut / LambdaOut  
        eps = -G * mIn * mOut * aOut_inv
        omegaIn = G * G * MIn * MIn * muIn**3 / (LambdaIn**3)
        omegaOut = G * G * MOut * MOut * muOut **3 / (LambdaOut**3)
        domegaIn = -3 * G * G * MIn * MIn * muIn**3 / (LambdaIn**4)
        domegaOut = -3 * G * G * MOut * MOut * muOut**3 / (LambdaOut**4)
        kIn = 1 - jres
        kOut = jres
        k_Domega_k = kIn**2 * domegaIn + kOut**2 * domegaOut
        prefactor = (k_Domega_k / (kIn * omegaIn + kOut * omegaOut)**2) 
        xToXIn = sqrt(2/LambdaIn)
        xToXOut = sqrt(2/LambdaOut)

        InIn = eps**2 * prefactor * CIn * CIn * xToXIn * xToXIn / 4
        InOut = eps**2 * prefactor * COut * CIn * xToXOut * xToXIn / 4
        OutOut = eps**2 * prefactor * COut * COut * xToXOut * xToXOut / 4
        
        self.ecc_entries[(indexIn,indexIn)] += InIn
        # Note-- only upper entries are stored so 
        # changing (indexOut,indexIn) also implicitly 
        # changes (indexIn,indexOut)
        self.ecc_entries[(indexOut,indexIn)] += InOut
        self.ecc_entries[(indexOut,indexOut)] += OutOut
        
def _get_pair_SecularHamiltonian_coefficients(Nmin,Nmax,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out,res_jk_list=[]):
    """
    Calculate the coefficients appearing in secular Hamiltonian expansion.
    """
    terms = SecularTermsList(Nmin,Nmax)
    extra_args = G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out
    dsec = terms_list_to_HamiltonianCoefficients_dict(terms,G,mIn,mOut,MIn,MOut,Lambda0In,Lambda0Out)
    for j,k in res_jk_list:
        dres = resonant_secular_contribution_dictionary(j,k,Nmin,Nmax,*extra_args)
        dsec = _add_dicts(dsec,dres)
        dsec = _consolidate_dictionary_terms(dsec)
    return _consolidate_dictionary_terms(dsec)
    
class _SecularDerivativesSystem():
    """
    A class for calculating the derivatives of a secular system.
    """
    def __init__(self,N,Hamiltonian_coefficients_dictionary,Lambda0):

        self.Lambda0 = np.array(Lambda0[1:])
        self.rtLambda0_inv = 1 / np.sqrt(Lambda0[1:])
        self.qp_to_XY_factors = np.concatenate((self.rtLambda0_inv,0.5*self.rtLambda0_inv))
        self.N=N
        self.Npl = self.N-1
        self.Ndim = 4*self.Npl

        self.DFSeries_dict=dict()
        for key,term_coeff_dict in Hamiltonian_coefficients_dictionary.items():
            i,j=key
            Lambda0In = self.Lambda0[i-1]
            Lambda0Out = self.Lambda0[j-1]
            self.DFSeries_dict[key] = DFTermSeries(term_coeff_dict,Lambda0In,Lambda0Out)

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

class SecularRKIntegrator(RKIntegrator):
    """
    Integrator for direct integration of secular equations of motion.
    """
    def __init__(self,N,hamiltonian_coefficients_dictionary,Lambda0s,dt,
                 rtol=_machine_eps,
                 atol=0,
                 rk_method='ImplicitMidpoint',
                 rk_root_method='Newton',
                 max_iter=10
                ):
        self._derivatives = _SecularDerivativesSystem(N,hamiltonian_coefficients_dictionary,Lambda0s)
        Ndim = 4 * (N-1)
        super(SecularRKIntegrator,self).__init__(
            self._derivatives.deriv_from_qp_vec,
            self._derivatives.deriv_and_jacobian_from_qp_vec,
            Ndim, dt, rtol, atol, rk_method, rk_root_method, max_iter
        )
    
        self.step = self.rk_step
        
    def init_step(self,qpvec):
        return qpvec
    
    def final_step(self,qpvec):
        return qpvec
    
    def calculate_energy(self,qp):
        return self._derivatives.Hamiltonian_from_qp_vec(qp)

class SecularSplittingIntegrator(RKIntegrator):
    """
    Integrator for applying splitting method to secular equations of motion.
    """
    def __init__(self,N,hamiltonian_coefficients_dictionary,Lambda0s,dt,
                 rtol=_machine_eps,
                 atol=0,
                 rk_method='ImplicitMidpoint',
                 rk_root_method='Newton',
                 max_iter=10
                ):
        
        all_hcoeffs = hamiltonian_coefficients_dictionary.copy()
        Npl = N-1
        Ndim = 4 * Npl
        self.Se = np.zeros((Npl,Npl))
        self.SI = np.zeros((Npl,Npl))
        for i in range(Npl):
            for j in range(i+1,Npl):
                hcoeffs=all_hcoeffs[(i+1,j+1)]
                # Get second-order eccentricity terms and add them to Laplace-Lagrange matrix
                self.Se[i,i] += hcoeffs.pop(((0,0,0,0,0,0),(0,0,1,0)),0)
                self.Se[j,j] += hcoeffs.pop(((0,0,0,0,0,0),(0,0,0,1)),0)
                term = hcoeffs.pop(((0,0,1,-1,0,0),(0,0,0,0)),0)
                self.Se[i,j] += 0.5 * term
                self.Se[j,i] += 0.5 * term
                # Get second-order inclination terms and add them to Laplace-Lagrange matrix
                self.SI[i,i] += hcoeffs.pop(((0,0,0,0,0,0),(1,0,0,0)),0)
                self.SI[j,j] += hcoeffs.pop(((0,0,0,0,0,0),(0,1,0,0)),0)
                term = hcoeffs.pop(((0,0,0,0,1,-1),(0,0,0,0)),0)
                self.SI[i,j] += 0.5 * term
                self.SI[j,i] += 0.5 * term
        rtLambda0s_inv = 1/np.sqrt(Lambda0s[1:])
        
        self.Se = 2 * np.diag(rtLambda0s_inv) @ self.Se @ np.diag(rtLambda0s_inv)
        self.SI = 0.5 * np.diag(rtLambda0s_inv) @ self.SI @ np.diag(rtLambda0s_inv)                
        self._derivatives = _SecularDerivativesSystem(N,all_hcoeffs,Lambda0s)
        Se,SI = self.Se,self.SI
        Zeros = np.zeros((Npl,Npl))
        self.generator_matrix = np.block([
            [Zeros,Zeros,Se,Zeros],
            [Zeros,Zeros,Zeros,SI],
            [-Se,Zeros,Zeros,Zeros],
            [Zeros,-SI,Zeros,Zeros],
        ])
        self._update_A_step_matrices(dt)
        self.corrector= False
        
        super(SecularSplittingIntegrator,self).__init__(
            self._derivatives.deriv_from_qp_vec,
            self._derivatives.deriv_and_jacobian_from_qp_vec,
            Ndim, dt, rtol, atol, rk_method, rk_root_method, max_iter
        )
        
        self.Bstep = self.rk_step
    
    def _update_A_step_matrices(self,dt):
        gen_matrix = self.generator_matrix
        self.A_half_step_forward_matrix = expm(0.5 * dt * gen_matrix)
        self.A_half_step_backward_matrix = expm(-0.5 * dt * gen_matrix)
        self.A_full_step_forward_matrix = self.A_half_step_forward_matrix @ self.A_half_step_forward_matrix

  
    @property
    def dt(self):
        return self._dt
    
    @dt.setter
    def dt(self,dt):
        self._update_A_step_matrices(dt)
        self._dt = dt
        
    def init_step(self,qpvec):
        if self.corrector:
                qpvec = self.corrector3(qpvec)
        return self.A_half_step_forward_matrix @ qpvec
    
    def step(self,qpvec):
        qpvec = self.Bstep(qpvec)
        return self.A_full_step_forward_matrix @ qpvec
    
    def final_step(self,qpvec):
        qpvec = self.A_half_step_backward_matrix @ qpvec
        if self.corrector:
            qpvec = self.corrector3inv(qpvec)
        return qpvec
    
    def _apply_A_step_for_dt(self,qpvec,h):
        return expm(h * self.generator_matrix) @ qpvec
    
    def _apply_B_step_for_dt(self,qpvec,h):
        h0 = self.dt
        self.dt = h
        qpvec1 = self.rk_step(qpvec)
        self.dt = h0
        return qpvec1
        
    def X(self, qpvec, a, b, h):
        qpvec = self._apply_A_step_for_dt(qpvec,-a*h)
        qpvec = self._apply_B_step_for_dt(qpvec,b*h)
        qpvec = self._apply_A_step_for_dt(qpvec,a*h)
        return qpvec

    def Z(self, qpvec, a, b, h):
        qpvec = self.X(qpvec, -a, -b, h)
        qpvec = self.X(qpvec, a, b, h)
        return qpvec

    def corrector3(self, qpvec, h):
        alpha = (7./40.)**0.5
        beta = 1/48./alpha
        a1 = -alpha
        a2 = alpha
        b2 = beta/2.
        b1 = -beta/2.
        
        qpvec = self.Z(qpvec, a2, b2, h)
        qpvec = self.Z(qpvec, a1, b1, h)
        return qpvec

    def corrector3inv(self, qpvec, h):
        alpha = (7./40.)**0.5
        beta = 1/48./alpha
        a1 = -alpha
        a2 = alpha
        b2 = beta/2.
        b1 = -beta/2.
        
        qpvec = self.Z(qpvec, a1, -b1, h)
        qpvec = self.Z(qpvec, a2, -b2, h)
        return qpvec

    def calculate_energy(self,qp):
        Epert = self._derivatives.Hamiltonian_from_qp_vec(qp)
        Omega = getOmegaMatrix(self.Ndim//2)
        E0 = -0.5 * qp @ Omega @ self.generator_matrix @ qp
        return E0 + Epert




class SecularSystemSimulation():
    """
    A class for integrating the secular equations of motion governing a planetary system.

    Arguments
    ---------
    state : :class:`celmech.poincare.Poincare`
        The initial dynamical state of the system.

    dt : float, optional
        The timestep to use for the integration. Either dt or dtFraction must be
        specified.

    dtFraction : float, optional
        Set the timestep to a constant fraction the period of shortest-period linear
        secular eigenmode.

    max_order : int, optional
        The maximum order of disturbing function terms to include in the integration.
        By default, the equations of motion include terms up to 4th order.

    method : str
        Integration method to use. Options include:
            - 'RK' - Direct integration of equations of motion using an Runge-Kutta integration method.
            - 'splitting' - Applies splitting to the flow generated by the system's Hamiltonian.

    resonances_to_include : dict, optional
        A dictionary containing information that sets the list of MMRs for which the
        secular contribution will be accounted for (at second order on planet masses).
        Dictionary key and value pairs are specified in the form

              .. code::

                {(iIn,iOut) : [(j_0,k_0),...(j_N,k_N)]}

        include the resonances :math:`j_i` : :math:`j_i-k_i` with :math:`i=0,...,N` between planets
        iIn and iOut. Note that harmonics should **NOT** be explicitly included. I.e.,
        if (2,1) appears in the list [(j_0,k_0),...(j_N,k_N)] then the term (4,2) should
        **NOT** also appear; these terms' contributions will be added automatically when
        the (2,1) term is added.
        By default, no MMRs are included.

    rk_kwargs : dict, optional
        Keyword arguments that determine details of the Runge-Kutta scheme used to integrate
        equations of motion. Keywoards include:
             - :code:`rtol`: float
                Relative tolerance of root-finding step in the implicit Runge-Kutta scheme.
                Default is set to machine precision.

            - :code:`atol`: float
                Absolute tolerance of the root-finding step in the implicit Runge-Kutta scheme.
                Default is 0 so that tolerance is set solely by :code:`rtol` argument.

            - :code:`rk_method`: str
                Runge-Kutta method to use.  Available options include:
                    - 'ImplicitMidpoint'
                    - 'LobattoIIIB'
                    - 'GL4'
                    - 'GL6'
                    - 'ExplicitMidpoint'
                    - 'RK4'
                'GL4' and 'GL6' are `Gauss-Legendre methods <https://en.wikipedia.org/wiki/Gauss–Legendre_method>`_ of order 4 and 6, respectively.
                'ImplicitMidpoint', 'GL4', and 'GL6' are symplectic methods while 'LobattoIIIB' is a 4th order time-reversible method (but not symplectic).
                'ExplicitMidpoint' and 'RK4' are explicit methods that are neither symplectic nor time-reversible.

            -:code:`rk_root_method` : str
                Method to use for root-finding during implicit RK step. Available options are:
                        - 'Newton'
                        - 'qausi-Newton'
                        - 'fixed_point'
                        - 'explicit'
                'Newton' (default) uses Newton's method whereas 'fixed_point' uses a fixed point iteration method.
                Newton's method requires computing the Jacobian of the equations of motion but has quadratic convergence.
    """

    def __init__(self,
                 state,
                 dt=None,dtFraction=None,
                 max_order=4,integrator='RK',
                 resonances_to_include={},
                 rk_kwargs = {}
               ):

        self.state = state
        self._hamiltonian_coefficients_dictionary = dict()
        Npl = self.N-1

        self.resonances_to_include = resonances_to_include
        self._hamiltonian_coefficients_dictionary = {(i+1,j+1):{} for i in range(Npl) for j in range(i+1,Npl)}
        self._max_order = 0
        self._update_matrcies_and_coefficient_dictionary(max_order)

        ecc_eigenvalues = np.linalg.eigvals(self.Se)
        inc_eigenvalues = np.linalg.eigvals(self.SI)
        self.Tsec_e = np.abs(2 * np.pi / ecc_eigenvalues)
        self.Tsec_inc = np.abs(2 * np.pi / inc_eigenvalues)
        self.Tsec = min(np.min(self.Tsec_e),np.min(self.Tsec_inc))
        self.t = 0

        if dt:
            self._dt = dt
        elif dtFraction:
            self._dt = dtFraction * self.Tsec
        else:
            raise AttributeError("Must specify either 'dt' or 'dtFraction'")

        self.secular_rk_integrator = SecularRKIntegrator(
            self.N,
            self.Hamiltonian_coefficients_dictionary,
            self.Lambda0s,
            self.dt,
            **rk_kwargs
        )

        self.secular_splitting_integrator = SecularSplittingIntegrator(
            self.N,
            self.Hamiltonian_coefficients_dictionary,
            self.Lambda0s,
            self.dt,
            **rk_kwargs
        )

        self.integrator = integrator

    @property
    def integrator(self):
        return self._integrator_name

    @integrator.setter
    def integrator(self,integrator):
        if integrator is 'RK':
            self._integrator_name = 'RK'
            self._integrator = self.secular_rk_integrator
        elif integrator is 'splitting':
            self._integrator_name = 'splitting'
            self._integrator = self.secular_splitting_integrator
        else:
            raise ValueError("{} is not a valid integrator option.".format(integrator))

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self,h):
        self._dt = h
        self.secular_rk_integrator.dt = h
        self.secular_splitting_integrator.dt = h

    @classmethod
    def from_Simulation(cls,sim,
                        dt=None,dtFraction=None,
                        max_order=4,integrator='RK',
                        resonances_to_include={},
                        rk_kwargs = {}
                       ):
        """
        Initialize a :class:`SecularSystemSimulation <celmech.secular.SecularSystemSimulation>` object
        from a rebound simulation.

        Arguments
        ---------
        sim : :class:`rebound.Simulation`
            REBOUND simulation to convert to :class:`SecularSystemSimulation <celmech.secular.SecularSystemSimulation>`

        dt : float, optional
            The timestep to use for the integration. Either dt or dtFraction must be
            specified.

        dtFraction : float, optional
            Set the timestep to a constant fraction the period of shortest-period linear
            secular eigenmode.

        max_order : int, optional
            The maximum order of disturbing function terms to include in the integration.
            By default, the equations of motion include terms up to 4th order.

        method : str
            Integration method to use. Options include:
                - 'RK' - Direct integration of equations of motion using an Runge-Kutta integration method.
                - 'splitting' - Applies splitting to the flow generated by the system's Hamiltonian.

        resonances_to_include : dict, optional
            A dictionary containing information that sets the list of MMRs for which the
            secular contribution will be accounted for (at second order on planet masses).
            Dictionary key and value pairs are specified in the form

                  .. code::

                    {(iIn,iOut) : [(j_0,k_0),...(j_N,k_N)]}

            include the resonances :math:`j_i` : :math:`j_i-k_i` with :math:`i=0,...,N` between planets
            iIn and iOut. Note that harmonics should **NOT** be explicitly included. I.e.,
            if (2,1) appears in the list [(j_0,k_0),...(j_N,k_N)] then the term (4,2) should
            **NOT** also appear; these terms' contributions will be added automatically when
            the (2,1) term is added.
            By default, no MMRs are included.

        rk_kwargs : dict, optional
            Keyword arguments that determine details of the Runge-Kutta scheme used to integrate
            equations of motion. Keywoards include:
                 - :code:`rtol`: float
                    Relative tolerance of root-finding step in the implicit Runge-Kutta scheme.
                    Default is set to machine precision.

                - :code:`atol`: float
                    Absolute tolerance of the root-finding step in the implicit Runge-Kutta scheme.
                    Default is 0 so that tolerance is set solely by :code:`rtol` argument.

                - :code:`rk_method`: str
                    Runge-Kutta method to use.  Available options include:
                        - 'ImplicitMidpoint'
                        - 'LobattoIIIB'
                        - 'GL4'
                        - 'GL6'
                    'GL4' and 'GL6' are `Gauss-Legendre methods <https://en.wikipedia.org/wiki/Gauss–Legendre_method>`_ of order 4 and 6, respectively.
                    'ImplicitMidpoint', 'GL4', and 'GL6' are symplectic methods while 'LobattoIIIB' is a 4th order time-reversible method (but not symplectic).

                -:code:`rk_root_method` : str
                    Method to use for root-finding during implicit RK step. Available options are:
                            - 'Newton'
                            - 'qausi-Newton'
                            - 'fixed_point'
                            - 'explicit'
                    'Newton' (default) uses Newton's method whereas 'fixed_point' uses a fixed point iteration method.
                    Newton's method requires computing the Jacobian of the equations of motion but has quadratic convergence.
    """


        pvars = Poincare.from_Simulation(sim)
        return cls(pvars,dt,dtFraction,max_order,integrator,resonances_to_include,rk_kwargs)

    @property
    def Lambda0s(self):
        """Values of particle's :math:`\Lambda` canonical momenta"""
        return np.array([p.Lambda for p in self.state.particles])

    @property
    def state_vector(self):
        state_vec = []
        for p in self.state.particles[1:]:
            state_vec += [p.kappa,p.eta,p.Lambda,p.l,p.sigma,p.rho]
        return np.array(state_vec)

    @property
    def max_order(self):
        """Maximum order of disturbing function expansion in :math:`e` and :math:`I`"""
        return self._max_order

    @max_order.setter
    def max_order(self,new_max_order):
        self._update_matrcies_and_coefficient_dictionaries(new_max_order)
        self._set_up_derivative_systems()
        self.EccentricityLinearEvolutionOperatorMatrix = expm(0.5 * self.dt * self.Se)
        self.InclinationLinearEvolutionOperatorMatrix = expm(0.5 * self.dt * self.SI)

    @property
    def N(self):
        """Number of particles (including central body)"""
        return self.state.N

    @property
    def Npl(self):
        return self.N - 1

    @property
    def G(self):
        """Value of the gravitational constant"""
        return self.state.G

    @property
    def Hamiltonian_coefficients_dictionary(self):
        """
        A dictionary containing the coefficients appearing in the Hamiltonian.
        Dictionary is organized by pair-wise interaction terms.
        """
        return self._hamiltonian_coefficients_dictionary


    def update_state_from_vector(self,state_vec):
        vecs =  np.reshape(state_vec,(-1,6))
        for vals,p in zip(vecs,self.state.particles[1:]):
            p.kappa,p.eta,p.Lambda,p.l,p.sigma,p.rho = vals


    def _update_matrcies_and_coefficient_dictionary(self,new_max_order):

        ps=self.state.particles
        G = self.G
        Npl = self.Npl
        old_max_order = self._max_order
        self.Se = np.zeros((Npl,Npl))
        self.SI = np.zeros((Npl,Npl))

        for i in range(1,self.N):
            I = i-1
            for j in range(i+1,self.N):
                J = j-1

                pIn = ps[i]
                pOut = ps[j]
                extra_args = G,pIn.m,pOut.m,pIn.M,pOut.M,pIn.Lambda,pOut.Lambda
                res_jk_list = self.resonances_to_include.get((i,j),[])
                hamiltonian_coeff_dict = self._hamiltonian_coefficients_dictionary[(i,j)]

                if new_max_order > self.max_order:

                    coeff_dict = _get_pair_SecularHamiltonian_coefficients(
                        old_max_order+2,
                        new_max_order,
                        *extra_args,
                        res_jk_list
                    )

                    hamiltonian_coeff_dict.update(coeff_dict)

                elif new_max_order < self.max_order:
                    for k,z in hamiltonian_coeff_dict:
                        order= np.sum(np.abs(k)) + 2 * np.sum(z)
                        if order > new_max_order:
                            hamiltonian_coeff_dict.pop((k,z))

                # Get second-order eccentricity terms and add them to Laplace-Lagrange matrix
                self.Se[I,I] += hamiltonian_coeff_dict.get(((0,0,0,0,0,0),(0,0,1,0)),0)
                self.Se[J,J] += hamiltonian_coeff_dict.get(((0,0,0,0,0,0),(0,0,0,1)),0)
                term = hamiltonian_coeff_dict.get(((0,0,1,-1,0,0),(0,0,0,0)),0)
                self.Se[I,J] += 0.5 * term
                self.Se[J,I] += 0.5 * term

                # Get second-order inclination terms and add them to Laplace-Lagrange matrix
                self.SI[I,I] += hamiltonian_coeff_dict.get(((0,0,0,0,0,0),(1,0,0,0)),0)
                self.SI[J,J] += hamiltonian_coeff_dict.get(((0,0,0,0,0,0),(0,1,0,0)),0)
                term = hamiltonian_coeff_dict.get(((0,0,0,0,1,-1),(0,0,0,0)),0)
                self.SI[I,J] += 0.5 * term
                self.SI[J,I] += 0.5 * term

        Lambda0s = self.Lambda0s[1:]
        rtLambda0s_inv = 1/np.sqrt(Lambda0s)
        self.Se = 2 * np.diag(rtLambda0s_inv) @ self.Se @ np.diag(rtLambda0s_inv)
        self.SI = 0.5 * np.diag(rtLambda0s_inv) @ self.SI @ np.diag(rtLambda0s_inv)

    def state_to_qp_vec(self):
        r"""
        Convert full state vector to vector of variables
        that enter in the secular equations.

        Returns
        -------
        qp_vec : ndarray
         Vector containing eccentricity and inclination
         variables :math:`\eta,\rho,\kappa,\sigma`. The variables
         are returned in the order:

         math::

          (\eta_1,\eta_2,...,\eta_N,\rho_1,...,\rho_N,\kappa_1,...,\kappa_N,\sigma_1,...,\sigma_N)
        """

        vecs = np.reshape(self.state_vector,(-1,6))
        kappa = vecs[:,0]
        eta = vecs[:,1]
        sigma = vecs[:,4]
        rho = vecs[:,5]
        return np.concatenate((eta,rho,kappa,sigma))


    def integrate(self,time,exact_finish_time=False):
        assert time >= self.t, "Backward integration is currently not implemented."
        Nstep = int( np.ceil( (time-self.t) / self.dt) )
        Npl=self.Npl
        state_vec = self.state_vector
        qp = self.state_to_qp_vec()

        qp = self._integrator.init_step(qp)
        for _ in xrange(Nstep):
            qp = self._integrator.step(qp)
        qp = self._integrator.final_step(qp)

        if exact_finish_time:
            warnings.warn("Exact finish time is not currently implemented.")

        for i in xrange(Npl):
            # eta
            state_vec[6*i+1] = qp[i]
            # kappa
            state_vec[6*i] = qp[i + 2 * Npl]
            # rho
            state_vec[6*i+5] = qp[i + Npl]
            # sigma
            state_vec[6*i+4] = qp[i + 3 * Npl]
        self.update_state_from_vector(state_vec)
        self.t += Nstep * self.dt

    def calculate_energy(self):
        qp = self.state_to_qp_vec()
        return self._integrator.calculate_energy(qp)

    def calculate_AMD(self):
        return sum([p.Q + p.Gamma for p in self.state.particles[1:]])


