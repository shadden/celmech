import numpy as np
import warnings
from . import Poincare
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, atan2, expand_trig,diff,Matrix
from .disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
from scipy.linalg import expm
from .poincare import single_true
_rt2 = np.sqrt(2)
_rt2_inv = 1 / _rt2 
_machine_eps = np.finfo(np.float64).eps

class LaplaceLagrangeSystem(Poincare):
    def __init__(self,G,poincareparticles=[]):
        super(LaplaceLagrangeSystem,self).__init__(G,poincareparticles)
        self.params = {S('G'):self.G}
        for i,particle in enumerate(self.particles):
            if i is not 0:
                m,M,Lambda = symbols('m{0},M{0},Lambda{0}'.format(i)) 
                self.params.update({m:particle.m,M:particle.M,Lambda:particle.Lambda})
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
        return cls(pvars.G,pvars.particles[1:])
    @classmethod
    def from_Simulation(cls,sim):
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
        oscillators in the variables (P,Q) defined by 
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
                mIn,MIn,LambdaIn = symbols('m{0},M{0},Lambda{0}'.format(indexIn)) 
                mOut,MOut,LambdaOut = symbols('m{0},M{0},Lambda{0}'.format(indexOut)) 
                Cecc_diag = get_DFCoeff_symbol(*[0 for _ in range(6)],0,0,1,0,indexIn,indexOut)
                Cinc_diag = get_DFCoeff_symbol(*[0 for _ in range(6)],1,0,0,0,indexIn,indexOut)
                prefactor = -G**2*MOut**2*mOut**3 *( mIn / MIn) / (LambdaOut**2)
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

    def add_first_order_resonance_term(self,indexIn, indexOut,kres):
        assert indexIn < indexOut, "Input 'indexIn' must be less than 'indexOut'."
        particleIn = self.particles[indexIn]
        particleOut = self.particles[indexOut]
        alpha = particleIn.a / particleOut.a
        G = symbols('G')
        mIn,MIn,LambdaIn = symbols('m{0},M{0},Lambda{0}'.format(indexIn)) 
        mOut,MOut,LambdaOut = symbols('m{0},M{0},Lambda{0}'.format(indexOut)) 
        
        CIn = get_DFCoeff_symbol(*[kres,1-kres,-1,0,0,0],0,0,0,0,indexIn,indexOut)
        COut = get_DFCoeff_symbol(*[kres,1-kres,0,-1,0,0],0,0,0,0,indexIn,indexOut)
        self.params[CIn] = eval_DFCoeff_dict(DFCoeff_C(*[kres,1-kres,-1,0,0,0],0,0,0,0),alpha)
        self.params[COut] = eval_DFCoeff_dict(DFCoeff_C(*[kres,1-kres,0,-1,0,0],0,0,0,0),alpha)
        eps = -G**2*MOut**2*mOut**3 *( mIn / MIn) / (LambdaOut**2)
        omegaIn = G * G * MIn * MIn * mIn**3 / (LambdaIn**3)
        omegaOut = G * G * MOut * MOut * mOut **3 / (LambdaOut**3)
        domegaIn = -3 * G * G * MIn * MIn * mIn**3 / (LambdaIn**4)
        domegaOut = -3 * G * G * MOut * MOut * mOut**3 / (LambdaOut**4)
        kIn = 1 - kres
        kOut = kres
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
        
from .symplectic_evolution_operators import EvolutionOperator
from .symplectic_evolution_operators import SecularDFTermsEvolutionOperator as DFOp
class LinearSecularEvolutionOperator(EvolutionOperator):
    def __init__(self,initial_state,dt,first_order_resonances={}):
        super(LinearSecularEvolutionOperator,self).__init__(initial_state,dt)
        LL_system = LaplaceLagrangeSystem.from_Poincare(self.state)
        for pair,res_j_list in first_order_resonances.items():
            for j in res_j_list:
                LL_system.add_first_order_resonance_term(*pair,j)
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
        x =  (kappa - 1j * eta) * _rt2_inv
        return x

    def _get_y_vector(self):
        rho = np.array([p.rho for p in self.particles[1:]])
        sigma = np.array([p.sigma for p in self.particles[1:]])
        y =  (sigma - 1j * rho) * _rt2_inv
        return y

    def _set_x_vector(self,x):
        for p,xi in zip(self.particles[1:],x):
            p.kappa = _rt2 * np.real(xi)
            p.eta =  _rt2 * np.real(1j * xi)

    def _set_y_vector(self,y):
        for p,yi in zip(self.particles[1:],y):
            p.sigma = _rt2 * np.real(yi)
            p.rho =  _rt2 * np.real(1j * yi)

    def apply(self):
        x = self._get_x_vector()
        y = self._get_y_vector()
        xnew = self.ecc_operator_matrix @ x
        ynew = self.inc_operator_matrix @ y
        self._set_x_vector(xnew)
        self._set_y_vector(ynew)
        
    def apply_to_state_vector(self,state_vector):
        vecs = self._state_vector_to_individual_vectors(state_vector)
        x = (vecs[:,0] - 1j * vecs[:,1]) * _rt2_inv
        y = (vecs[:,4] - 1j * vecs[:,5]) * _rt2_inv
        xnew = self.ecc_operator_matrix @ x
        ynew = self.inc_operator_matrix @ y
        vecs[:,0] = _rt2 * np.real(xnew)
        vecs[:,1] = -1 * _rt2 * np.imag(xnew)
        vecs[:,4] = _rt2 * np.real(ynew)
        vecs[:,5] = -1 * _rt2 * np.imag(ynew)
        return vecs.reshape(-1)
    
    def calculate_Hamiltonian(self,state_vector):
        vecs = self._state_vector_to_individual_vectors(state_vector)
        x = (vecs[:,0] - 1j * vecs[:,1]) * _rt2_inv
        y = (vecs[:,4] - 1j * vecs[:,5]) * _rt2_inv
        H = np.conj(x) @ self.ecc_matrix @ x + np.conj(y) @ self.inc_matrix @ y
        return np.real(H)
class SecularSystemSimulation():
    def __init__(self, state, dt = None, dtFraction = None, max_order = 4,NsubB=1, resonances_to_include={}, DFOp_kwargs = {}):
        """
        A class for integrating the secular equations of motion governing a planetary system.

        The integrations are carried out using a symplectic splitting scheme. The scheme
        separates the equations of motion into an (integrable) linear component equivalent
        to the Laplace-Largange equations of motion, and a component containing all 
        higher-order terms. The linear components are solved exactly while the higher-order
        terms are solved using the symplectic implicit midpoint method.
        
        Arguments
        ----------
        state : celmech.Poincare
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
        NsubB : int, optional
            The 'B' step in the splitting scheme is divided in NsubB sub-steps which each integrate for a time dt/NsubB.  By default, NsubB = 1.
        resonances_to_include : dict, optional
            A dictionary containing information that sets the list of MMRs for which the 
            secular contribution will be accounted for (at second order on planet masses).
            Dictionary should be in the from:
                {
                  ....
                  (iIn,iOut):[(j_0,k_0),...(j_N,k_N)]
                  ....
                }
            in order to include the resonances j_i : j_i-k_i with i=0,...,N between planets
            iIn and iOut. Note that harmonics should *NOT* be explicitly included. I.e.,
            if (2,1) appears in the list [(j_0,k_0),...(j_N,k_N)] then the term (4,2) should
            *NOT* also appear; these terms' contributions will be added automatically when 
            the (2,1) term is added.
            By default, no MMRs are included.
        DFOp_kwargs : dict, optional
            Keyword arguments to use when initialzing the operator used to evolve the non-linear terms. 
            See celmech.symplectic_evolution_operators.SecularDFTermsEvolutionOperator for list of
            keyword arguments.
        """
        assert max_order > 3, "'max_order' must be greater than or equal to 4."
        if not single_true([dt,dtFraction]):
            raise AttributeError("Can only pass one of dt or dtFraction")
        llsys = LaplaceLagrangeSystem.from_Poincare(state)
        first_order_resonances_to_include = {}
        for pair,res_list in resonances_to_include.items():
            first_order_js = [ j for j,k in res_list if k==1]
            first_order_resonances_to_include.update({pair:first_order_js})
            for j in first_order_js:
                llsys.add_first_order_resonance_term(*pair,j)
        Tsec_e = np.min(np.abs(2 * np.pi / llsys.eccentricity_eigenvalues()))
        Tsec_inc = np.min(np.abs(2 * np.pi / llsys.inclination_eigenvalues()[1:]))
        self.Tsec = min(Tsec_e,Tsec_inc)
        self._NsubB = NsubB
        if dt:
            self._dtA = dt
        elif dtFraction:
            self._dtA = dtFraction * self.Tsec
        self._dtB = self._dtA / self._NsubB
        self.linearSecOp = LinearSecularEvolutionOperator(state,self._dtA)
        self.nonlinearSecOp = DFOp.fromOrderRange(
                state,
                self._dtB,
                4,max_order,
                resonances_to_include=resonances_to_include,
                **DFOp_kwargs
        )
        self.state = state
        self._half_step_forward_e_matrix = expm(-1j * 0.5 * self.dt * self.linearSecOp.ecc_matrix)
        self._half_step_backward_e_matrix = expm(+1j * 0.5 * self.dt * self.linearSecOp.ecc_matrix)
        self._half_step_forward_inc_matrix = expm(-1j * 0.5 * self.dt * self.linearSecOp.inc_matrix)
        self._half_step_backward_inc_matrix = expm(+1j * 0.5 * self.dt * self.linearSecOp.inc_matrix)
        self.t = 0

    @classmethod
    def from_Simulation(cls,sim, dt = None, dtFraction = None, max_order = 4,NsubB=1,resonances_to_include={}, DFOp_kwargs = {}):
        pvars = Poincare.from_Simulation(sim)
        return cls(
                pvars,
                max_order = max_order,
                NsubB=NsubB,
                dt = dt,
                dtFraction = dtFraction,
                resonances_to_include=resonances_to_include,
                DFOp_kwargs = DFOp_kwargs
        )

    @property
    def state_vector(self):
        state_vec = []
        for p in self.state.particles[1:]:
            state_vec += [p.kappa,p.eta,p.Lambda,p.l,p.sigma,p.rho]
        return np.array(state_vec)
    @property
    def NsubB(self):
        return self._NsubB
    @NsubB.setter
    def NsubB(self,N):
        self._NsubB = N
        self._dtB = self._dtA / N
        self.nonlinearSecOp.dt = self._dtB
    @property
    def dt(self):
        return self._dtA
    @dt.setter
    def dt(self,value):
        self._dtA = value
        self._dtB = self._dtA / self._NsubB
        self.linearSecOp.dt = self._dtA
        self.nonlinearSecOp.dt = self._dtB
        self._half_step_forward_e_matrix = expm(-1j * 0.5 * value * self.linearSecOp.ecc_matrix)
        self._half_step_backward_e_matrix = expm(+1j * 0.5 * value * self.linearSecOp.ecc_matrix)
        self._half_step_forward_inc_matrix = expm(-1j * 0.5 * value * self.linearSecOp.inc_matrix)
        self._half_step_backward_inc_matrix = expm(+1j * 0.5 * value *  self.linearSecOp.inc_matrix)

    def linearOp_half_step_forward(self,state_vec):
        """
        Advance state vector a half timestep forward with the 
        linear secular evolution operator.
        """
        vecs = self.linearSecOp._state_vector_to_individual_vectors(state_vec)
        x = (vecs[:,0] - 1j * vecs[:,1]) * _rt2_inv
        y = (vecs[:,4] - 1j * vecs[:,5]) * _rt2_inv
        xnew = self._half_step_forward_e_matrix @ x
        ynew = self._half_step_forward_inc_matrix @ y
        vecs[:,0] = _rt2 * np.real(xnew)
        vecs[:,1] = -1 * _rt2 * np.imag(xnew)
        vecs[:,4] = _rt2 * np.real(ynew)
        vecs[:,5] = -1 * _rt2 * np.imag(ynew)
        return vecs.reshape(-1)
        
    def linearOp_half_step_backward(self,state_vec):
        """
        Advance state vector a half timestep backward with the 
        linear secular evolution operator.
        """
        vecs = self.linearSecOp._state_vector_to_individual_vectors(state_vec)
        x = (vecs[:,0] - 1j * vecs[:,1]) * _rt2_inv
        y = (vecs[:,4] - 1j * vecs[:,5]) * _rt2_inv
        xnew = self._half_step_backward_e_matrix @ x
        ynew = self._half_step_backward_inc_matrix @ y
        vecs[:,0] = _rt2 * np.real(xnew)
        vecs[:,1] = -1 * _rt2 * np.imag(xnew)
        vecs[:,4] = _rt2 * np.real(ynew)
        vecs[:,5] = -1 * _rt2 * np.imag(ynew)
        return vecs.reshape(-1)
    
    def update_state_from_vector(self,state_vec):
        vecs = self.linearSecOp._state_vector_to_individual_vectors(state_vec)
        for vals,p in zip(vecs,self.state.particles[1:]):
            p.kappa,p.eta,p.Lambda,p.l,p.sigma,p.rho = vals

    def integrate(self,time,exact_finish_time = False, corrector=False):
        assert time >= self.t, "Backward integration is currently not implemented."
        Nstep = int( np.ceil( (time-self.t) / self.dt) )
        state_vec = self.state_vector
        if corrector is True:
            state_vec = self.corrector3(state_vec, self.dt)
        state_vec = self.linearOp_half_step_forward(state_vec)
        for _ in xrange(Nstep):

            # B step
            state_vec = self.Bstep(state_vec)
            
            # A step
            state_vec = self.linearSecOp.apply_to_state_vector(state_vec)

        if exact_finish_time:
           warnings.warn("Exact finish time is not currently implemented.")
        state_vec = self.linearOp_half_step_backward(state_vec)
        if corrector is True:
            state_vec = self.corrector3inv(state_vec, self.dt)
        self.update_state_from_vector(state_vec)
        self.t += Nstep * self.dt

    def Bstep(self,state_vec):
        nlOp = self.nonlinearSecOp
        qp = nlOp.state_vec_to_qp_vec(state_vec)
        for _ in xrange(self.NsubB):
            qp = nlOp.implicit_rk_step(qp)
        for i in xrange(nlOp.Npl):
            # eta
            state_vec[6*i+1] = qp[i]
            # kappa
            state_vec[6*i] = qp[i + 2 * nlOp.Npl]
            # rho
            state_vec[6*i+5] = qp[i + nlOp.Npl]
            # sigma
            state_vec[6*i+4] = qp[i + 3 * nlOp.Npl]

        return state_vec
    def calculate_energy(self):
        sv = self.state_vector
        E = self.linearSecOp.calculate_Hamiltonian(sv)
        E += self.nonlinearSecOp.calculate_Hamiltonian(sv)
        return E

    def calculate_AMD(self):
        return np.sum([p.Q + p.Gamma for p in self.state.particles[1:]])

    def X(self, state_vec, a, b, h):
        state_vec = self.apply_A_step_for_dt(state_vec,-a*h)
        state_vec = self.apply_B_step_for_dt(state_vec,b*h)
        state_vec = self.apply_A_step_for_dt(state_vec,a*h)
        return state_vec

    def Z(self, state_vec, a, b, h):
        state_vec = self.X(state_vec, -a, -b, h)
        state_vec = self.X(state_vec, a, b, h)
        return state_vec

    def corrector3(self, state_vec, h):
        alpha = (7./40.)**0.5
        beta = 1/48./alpha
        a1 = -alpha
        a2 = alpha
        b2 = beta/2.
        b1 = -beta/2.
        
        state_vec = self.Z(state_vec, a2, b2, h)
        state_vec = self.Z(state_vec, a1, b1, h)
        return state_vec

    def corrector3inv(self, state_vec, h):
        alpha = (7./40.)**0.5
        beta = 1/48./alpha
        a1 = -alpha
        a2 = alpha
        b2 = beta/2.
        b1 = -beta/2.
        
        state_vec = self.Z(state_vec, a1, -b1, h)
        state_vec = self.Z(state_vec, a2, -b2, h)
        return state_vec

    def apply_A_step_for_dt(self,state_vec,dt):
        """
        Apply the linear secular evolution operator for a time
        'dt' to a state vector.

        Argruments
        ----------
        state_vec : ndarray
          State vector of planetary system.
        dt : float
          Timestep to apply operator for.

        Returns
        -------
        state_vec : ndarray
          Updated state vector after application of operator.
        """
        opMtrx_ecc = expm(-1j *  dt * self.linearSecOp.ecc_matrix)
        opMtrx_inc = expm(-1j *  dt * self.linearSecOp.inc_matrix)

        vecs = self.linearSecOp._state_vector_to_individual_vectors(state_vec)
        x = (vecs[:,0] - 1j * vecs[:,1]) * _rt2_inv
        y = (vecs[:,4] - 1j * vecs[:,5]) * _rt2_inv
        xnew = opMtrx_ecc @ x
        ynew = opMtrx_inc @ y
        vecs[:,0] = _rt2 * np.real(xnew)
        vecs[:,1] = -1 * _rt2 * np.imag(xnew)
        vecs[:,4] = _rt2 * np.real(ynew)
        vecs[:,5] = -1 * _rt2 * np.imag(ynew)
        return vecs.reshape(-1)

    def apply_B_step_for_dt(self,state_vec,dt):
        """
        Apply the secular evolution operator for the non-linear (4th and higher order)
        secular terms for a time 'dt' to a state vector.

        Argruments
        ----------
        state_vec : ndarray
          State vector of planetary system.
        dt : float
          Timestep to apply operator for.

        Returns
        -------
        state_vec : ndarray
          Updated state vector after application of operator.
        """

        # change time-step
        self.nonlinearSecOp.dt = dt

        # apply
        state_vec = self.nonlinearSecOp.apply_to_state_vector(state_vec)
        
        # set time-step back to default
        self.nonlinearSecOp.dt = self.dt
        
        # return result
        return state_vec
