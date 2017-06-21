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
            self.derivs[p] = -diff(self.h, q)
            self.derivs[q] = diff(self.h, p)
        
        self.nh = self.h
        for i, param in enumerate(self.params):
            try:
                self.nh = self.nh.subs(param, self.nparams[i])
            except keyerror:
                raise attributeerror("need to pass keyword {0} to hamiltonian.integrate".format(param))
        symvars = [item for pqpair in self.pqpairs for item in pqpair]
        self.nderivs = []
        for pqpair in self.pqpairs:
            p,q = pqpair
            self.nderivs.append(lambdify(symvars, -diff(self.nh, q), 'numpy'))
            self.nderivs.append(lambdify(symvars, diff(self.nh, p), 'numpy'))
        
        def diffeq(t, y):
            dydt = [deriv(*y) for deriv in self.Nderivs]
            return dydt
        self.integrator = ode(diffeq).set_integrator('lsoda')
        self.integrator.set_initial_value(initial_conditions, 0)

class HamiltonianPoincare(Hamiltonian):
    def __init__(self, Lambdas, lambdas, Gammas, gammas):
        self.resonance_indices = []
        self.integrator = None 
        self.H = 0
    def initialize_from_sim(self, sim):
        Nmjac, NMjac, Nmu = jacobi_masses_from_sim(sim)
        initial_conditions = poincare_vars_from_sim(sim)

        self.mjac = symbols("mjac\{0:{0}\}".format(sim.N))
        self.Mjac = symbols("Mjac\{0:{0}\}".format(sim.N))
        self.mu =  symbols("mu\{0:{0}\}".format(sim.N))
################
		self.Lambda = symbols("Lambda\{0:{0}\}".format(sim.N))
		self.lam = symbols("lambda\{0:{0}\}".format(sim.N))
		self.Gamma = symbols("Gamma\{0:{0}\}".format(sim.N))
		self.gamma = symbols("gamma\{0:{0}\}".format(sim.N))
			
        actionanglepairs = [ ]
        for i in range(sim.N):
        	actionanglepairs += [(self.Lambda[i],self.lam[i])]
        	actionanglepairs += [(self.Gamma[i], self.gamma[i])]
        
        params = self.mu        
        Nparams = [mjac[inner], Mjac[inner], mu[inner], mu[outer], m, Nf27, Nf31]
        initial_conditions = poincare_vars_from_sim(sim)[:8]
        super(HamiltonianPoincare, self).__init__(H, actionanglepairs, initial_conditions, params, Nparams)
    def add_single_resonance(idIn,idOut,res_jkl,alpha):
	    """
	    Add a single term associated the j:j-k MMR between planets 'idIn' and 'idOut'.
	    Inputs:
	    idIn-ID of the inner planet
	    idOut	-	ID of the outer planet
    	res_jkl	-	Ordered triple (j,k,l) specifying resonant term. 
    				 The 'l' index picks out the eIn^(l) * eOut^(k-l) subdterm
    	alpha	-	The semi-major axis ratio aIn/aOut
    	"""
		# Canonical variables
		LambdaIn,lambdaIn,GammaIn,gammaIn = self._get_single_id_variables(idIn)
		LambdaOut,lambdaOut,GammaOut,gammaOut = self._get_single_id_variables(idIn)
	
		# Mass variables
		muOut = self.mu[idOut]
		Mout= self.Mjac[idOut]
		mIn  = self.mjac[idIn]
	
		# Resonance index
		j,k,l = res_jkl
		assert l<=k, "Invalid resonance term, l>k."
	
		# Resonance components
		from celmech.disturbing_function import general_order_coefficient
		#
		Cjkl = symbol( "\{C_\{{0},{1},{2}\}\}".format(j,k,l) )
		self.params.append(Cjkl)
		self.Nparams.append(general_order_coefficient(j,k,l,alpha))
		#
		eccIn = sqrt(2*GammaIn/LambdaIn)
		eccOut = sqrt(2*GammaOut/LambdaOut)
		#
		costerm = cos( j * lambdaOut - (j-k) * lambdaIn + l * gammaIn + (k-l) * gammaOut )
		#
		prefactor = -muOut *( mIn / Mout) / (LambdaOut**2)
	
		# Keep track of resonances
		self.resonance_indicies.append((idIn,idOut,res_jkl))
		# Update Hamiltonian
		self.H += prefactor * Cjkl * (eccIn**l) * (eccOut**(k-l)) * costerm
    
    def add_all_resonance_subterms(idIn,idOut,res_j,res_k,alpha):
  		"""
    	Add a single term associated the j:j-k MMR between planets 'idIn' and 'idOut'.
    	Inputs:
        idIn    -    ID of the inner planet
        idOut    -    ID of the outer planet
        res_j    -    Together with 'res_k' specifies the MMR 'res_j:res_j-res_k'
        res_k    -    Order of the resonance
        alpha    -    The semi-major axis ratio aIn/aOut
  		"""
  		for res_l in range(res_k+1):
  			self.add_single_resonance(inIn,idOut,(res_j,res_k,res_l),alpha)
  	def add_Hkep_term(id):
		"""
		Add the Keplerian component of the Hamiltonian for planet 'id'.
		"""
		Lambda,lam,Gamma,gamma = self._get_single_id_variables(id)
		Mjac = self.Mjac[id]
		mjac = self.mjac[id]
		mu = self.mu[id]
		self.H +=  -mu / (2 * Lambda * Lambda)
	def _get_single_id_variables(id):
		Lambda,lam =  actionanglepairs[2 * (id - 1) ]
		Gamma,gamma =  actionanglepairs[2 * (id-1) +1]
		return (Lambda,lam,Gamma,gamma)
