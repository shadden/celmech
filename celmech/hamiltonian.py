from sympy import S, diff, lambdify, symbols, sqrt, cos, numbered_symbols, simplify
from scipy.integrate import ode
import numpy as np
import rebound
from celmech.transformations import jacobi_masses_from_sim, poincare_vars_from_sim
from celmech.disturbing_function import laplace_coefficient

class Hamiltonian(object):
    def __init__(self, H, pqpairs, initial_conditions, params, Nparams):
        self.pqpairs = pqpairs
        self.params = params
        self.Nparams = Nparams
        self.initial_conditions = initial_conditions
        self.H = H
        self._update()
    def integrate(self, time):
        if time > self.integrator.t:
            try:
                self.integrator.integrate(time)
            except:
                raise AttributeError("Need to initialize Hamiltonian")
    def _update(self):
        self.derivs = {}
        for pqpair in self.pqpairs:
            p,q = pqpair
            self.derivs[p] = -diff(self.H, q)
            self.derivs[q] = diff(self.H, p)
        
        self.NH = self.H
        for i, param in enumerate(self.params):
            try:
                self.NH = self.NH.subs(param, self.Nparams[i])
            except KeyError:
                raise AttributeError("need to pass keyword {0} to hamiltonian.integrate".format(param))
        symvars = [item for pqpair in self.pqpairs for item in pqpair]
        self.Nderivs = []
        for pqpair in self.pqpairs:
            p,q = pqpair
            self.Nderivs.append(lambdify(symvars, -diff(self.NH, q), 'numpy'))
            self.Nderivs.append(lambdify(symvars, diff(self.NH, p), 'numpy'))
        
        def diffeq(t, y):
            dydt = [deriv(*y) for deriv in self.Nderivs]
            return dydt
        self.integrator = ode(diffeq).set_integrator('lsoda')
        self.integrator.set_initial_value(self.initial_conditions, 0)

class HamiltonianPoincare(Hamiltonian):
    def __init__(self):
        self.resonance_indices = []
        self.integrator = None
    def initialize_from_sim(self, sim):
        self.H = S(0)
        Nm, NM, Nmu = jacobi_masses_from_sim(sim)
        initial_conditions = poincare_vars_from_sim(sim)

        self.m = list(symbols("m0:{0}".format(sim.N)))
        self.M = list(symbols("M0:{0}".format(sim.N)))
        self.mu = list(symbols("mu0:{0}".format(sim.N)))
################
        self.Lambda = list(symbols("Lambda0:{0}".format(sim.N)))
        self.lam = list(symbols("lambda0:{0}".format(sim.N)))
        self.Gamma = list(symbols("Gamma0:{0}".format(sim.N)))
        self.gamma = list(symbols("gamma0:{0}".format(sim.N)))
################            
        self.params = self.mu + self.m + self.M
        self.Nparams = Nmu + Nm + NM
        self.initial_conditions = poincare_vars_from_sim(sim)
        self.pqpairs = [ ]
        for i in range(1,sim.N):
            self.pqpairs.append((self.Lambda[i],self.lam[i]))
            self.pqpairs.append((self.Gamma[i], self.gamma[i]))
            self.add_Hkep_term(i)
        self._update()
    
    def add_Hkep_term(self, index):
        """
        Add the Keplerian component of the Hamiltonian for planet ''.
        """
        m, M, mu, Lambda, lam, Gamma, gamma = self._get_symbols(index)
        self.H +=  -mu / (2 * Lambda**2)
    
    def add_single_resonance(self, indexIn,indexOut,res_jkl,alpha):
        """
        Add a single term associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -   index of the inner planet
        indexOut    -   index of the outer planet
        res_jkl     -   Ordered triple (j,k,l) specifying resonant term. 
                        The 'l' index picks out the eIn^(l) * eOut^(k-l) subdterm
        alpha       -   The semi-major axis ratio aIn/aOut
        """
        # Canonical variables
        mIn, MIn, muIn, LambdaIn, lambdaIn, GammaIn, gammaIn = self._get_symbols(indexIn)
        mOut, MOut, muOut, LambdaOut, lambdaOut, GammaOut, gammaOut = self._get_symbols(indexOut)
        
        # Resonance index
        j,k,l = res_jkl
        assert l<=k, "Inval resonance term, l>k."
    
        # Resonance components
        from celmech.disturbing_function import general_order_coefficient
        #
        Cjkl = symbols( "C_{0}\,{1}\,{2}".format(j,k,l) )
        self.params.append(Cjkl)
        self.Nparams.append(general_order_coefficient(j,k,l,alpha))
        #
        eccIn = sqrt(2*GammaIn/LambdaIn)
        eccOut = sqrt(2*GammaOut/LambdaOut)
        #
        costerm = cos( j * lambdaOut - (j-k) * lambdaIn + l * gammaIn + (k-l) * gammaOut )
        #
        prefactor = -muOut *( mIn / MOut) / (LambdaOut**2)
    
        # Keep track of resonances
        self.resonance_indices.append((indexIn,indexOut,res_jkl))
        # Update Hamiltonian
        self.H += prefactor * Cjkl * (eccIn**l) * (eccOut**(k-l)) * costerm
        self._update()
    
    def add_all_resonance_subterms(self, indexIn, indexOut, res_j, res_k, alpha):
        """
        Add a single term associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn    -    index of the inner planet
        indexOut    -    index of the outer planet
        res_j    -    Together with 'res_k' specifies the MMR 'res_j:res_j-res_k'
        res_k    -    Order of the resonance
        alpha    -    The semi-major axis ratio aIn/aOut
        """
        for res_l in range(res_k+1):
            self.add_single_resonance(indexIn,indexOut,(res_j,res_k,res_l),alpha)
    def _get_symbols(self, index):
        return self.m[index], self.M[index], self.mu[index], self.Lambda[index], self.lam[index], self.Gamma[index], self.gamma[index]

