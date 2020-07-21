import numpy as np
from . import Poincare
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, atan2, expand_trig,diff,Matrix
from celmech.disturbing_function import get_fg_coeffs, general_order_coefficient, secular_DF,laplace_B, laplace_coefficient
from celmech.disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
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