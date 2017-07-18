from sympy import S, diff, lambdify, symbols, sqrt, cos,sin, numbered_symbols, simplify
from scipy.integrate import ode
from collections import OrderedDict
import numpy as np
import rebound
from celmech.transformations import jacobi_masses_from_sim, poincare_vars_from_sim
from celmech.disturbing_function import laplace_coefficient
'''
class Ham():
    def __init__():
        self.pqpairs
        self.vars
        self.params
need vars and params for transfor

pqpairs = [(x,px), (y,py)]
initial_conditions = [1,5,3,5]
params = [m, G, k]
Nparams = [3, 4, 5]
H = p^2/2m + 1/2kx^2

{x:1, px:5, 
'''

class Hamiltonian(object):
    def __init__(self, H, pqpairs, params, initial_conditions=None, Nparams=None):
        self._pqpairs = pqpairs
        self.y = OrderedDict()
        self.params = OrderedDict()
        for pqpair in self._pqpairs:
            p,q = pqpair
            self.y[p.name] = 0.
            self.y[q.name] = 0.
        for param in params:
            self.params[param.name] = None
        if initial_conditions is not None:
            self.set_y(initial_conditions)
        if Nparams is not None:
            self.set_params(Nparams)
        self.H = H
        self._update()

    def set_y(self, values):
        for i, key in enumerate(self.y.keys()):
            self.y[key] = values[i]
    
    def set_params(self, values):
        for i, key in enumerate(self.params.keys()):
            self.params[key] = values[i]

    def integrate(self, time):
        if time > self.integrator.t:
            try:
                self.integrator.integrate(time)
            except:
                raise AttributeError("Need to initialize Hamiltonian")
        for i, key in enumerate(self.y.keys()):
            self.y[key] = self.integrator.y[i]

    def _update(self):
        self.derivs = {}
        for pqpair in self._pqpairs:
            p,q = pqpair
            self.derivs[p] = -diff(self.H, q)
            self.derivs[q] = diff(self.H, p)
        
        self.NH = self.H
        for param, val in self.params.items():
            self.NH = self.NH.subs(param, val)
            #raise AttributeError("need to pass keyword {0} to hamiltonian.integrate".format(param))
        symvars = [item for pqpair in self._pqpairs for item in pqpair]
        self.Nderivs = []
        for pqpair in self._pqpairs:
            p,q = pqpair
            self.Nderivs.append(lambdify(symvars, -diff(self.NH, q), 'numpy'))
            self.Nderivs.append(lambdify(symvars, diff(self.NH, p), 'numpy'))
        
        def diffeq(t, y):
            dydt = [deriv(*y) for deriv in self.Nderivs]
            #print(t, y, dydt)
            return dydt
        self.integrator = ode(diffeq).set_integrator('lsoda')
        self.integrator.set_initial_value(list(self.y.values()), 0)

class EquibAndoyerHamiltonian(Hamiltonian):
    def __init__(self, k, NPhiprime, Phi0, phi0):
        Phi, phi, Phiprime = symbols('Phi, phi, Phiprime')
        pqpairs = [(Phi, phi)]
        params = [Phiprime]
        Nparams = [NPhiprime]
        initial_conditions = [Phi0, phi0]
        H = S(4)*Phi**2 - 3*Phiprime*Phi + (S(2)*Phi)**(k/S(2))*cos(phi)
        super(EquibAndoyerHamiltonian, self).__init__(H, pqpairs, params, initial_conditions, Nparams)

class AndoyerHamiltonian(Hamiltonian):
    def __init__(self, k, NPhiprime, Phi0, phi0):
        Phi, phi, Phiprime = symbols('Phi, phi, Phiprime')
        pqpairs = [(Phi, phi)]
        params = [Phiprime]
        Nparams = [NPhiprime]
        initial_conditions = [Phi0, phi0]
        H = S(1)/2*(Phi-Phiprime)**2 + Phi**(k/S(2))*cos(phi)
        super(AndoyerHamiltonian, self).__init__(H, pqpairs, params, initial_conditions, Nparams)
    #@classmethod
    #def from_Simulation(cls, sim, j, k):
    #
    #    return cls(k, NPhiprime, Phi0, phi0)
        
class CartesianAndoyerHamiltonian(Hamiltonian):
    def __init__(self, k, NPhiprime, Phi0, phi0):
        X,Y,Phiprime = symbols('X, Y, Phiprime')
        self.pqpairs = [(X, Y)]
        self.params = [Phiprime]
        self.Nparams = [NPhiprime]
        self.initial_conditions = [np.sqrt(2.*Phi0)*np.cos(phi0), np.sqrt(2.*Phi0)*np.sin(phi0)]
        self.H = S(1)/2*((X**2 + Y**2)/S(2)-Phiprime)**2 + S(1)/sqrt(S(2))*X*((X**2+Y**2)/2)**((k-1)/S(2))
        self._update()

class HamiltonianThetas(Hamiltonian):
    def __init__(self, sim, j, k):
        self.pham = HamiltonianPoincare()
        self.pham.initialize_from_sim(sim)
        self.pham.add_all_resonance_subterms(self, 1, 2, j, k)
        print(self.pham.H)

class HamiltonianPoincareXY(Hamiltonian):
    def __init__(self):
        self.resonance_indices = []
        self.integrator = None
        self.N = 0
    def initialize_from_PoincareHamiltonian(self,PHam):
        # add sympy symbols
        #
        #  parameters
        self.m = PHam.m
        self.M = PHam.M
        self.mu = PHam.mu
        self.N = PHam.N
        #  canonical variables
        self.X = list(symbols("X0:{0}".format(self.N)))
        self.Y = list(symbols("Y0:{0}".format(self.N)))
        self.Lambda = list(PHam.Lambda)
        self.lam = list(PHam.lam)
        
        self.H = S(0)
        self.params = list(PHam.params)
        self.pqpairs = []
        for i in range(1,PHam.N):
            self.pqpairs.append((self.Lambda[i],self.lam[i]))
            self.pqpairs.append((self.X[i], self.Y[i]))
            self.add_Hkep_term(i)
    
        self.a = list(PHam.a)
        self.Nm, self.NM, self.Nmu = list(PHam.Nm), list(PHam.NM), list(PHam.Nmu)
        self.Nparams = self.Nmu + self.Nm + self.NM
        pham_initial_conditions = list(PHam.initial_conditions)
        
        self.initial_conditions=[]
        for i,y0 in enumerate(pham_initial_conditions):
            if i%3==0 or i%4==0:
                self.initial_conditions.append(y0)
            else:
                 self.initial_conditions.append(y0)
        
        self._update()

    def add_Hkep_term(self,index):
        m, M, mu, Lambda,lam,X,Y = self._get_symbols(index)
        self.H +=  -mu / (2 * Lambda**2)

    def add_all_resonance_subterms(self, indexIn, indexOut, j, k):
        """
        Add all the terms associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -    index of the inner planet
        indexOut    -    index of the outer planet
        j           -    together with k specifies the MMR j:j-k
        k           -    order of the resonance
        """
        for l in range(k+1):
            self.add_single_resonance(indexIn,indexOut, j, k, l)

    def add_single_resonance(self, indexIn, indexOut, j, k, l):
        """
        Add a single term associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -   index of the inner planet
        indexOut    -   index of the outer planet
        j           -   together with k specifies the MMR j:j-k
        k           -   order of the resonance
        l           -   picks out the eIn^(l) * eOut^(k-l) subterm
        """
        # Canonical variables
        assert indexOut == indexIn + 1,"Only resonances for adjacent pairs are currently supported"
        mIn, MIn, muIn, LambdaIn,lambdaIn,XIn,YIn = self._get_symbols(indexIn)
        mOut, MOut, muOut, LambdaOut,lambdaOut,XOut,YOut = self._get_symbols(indexOut)
        
        # Resonance index
        assert l<=k, "Invalid resonance term, l must be less than or equal to k."
        alpha = self.a[indexIn]/self.a[indexOut]

        # Resonance components
        from celmech.disturbing_function import general_order_coefficient
        #
        Cjkl = symbols( "C_{0}\,{1}\,{2}".format(j,k,l) )
        self.params.append(Cjkl)
        self.Nparams.append(general_order_coefficient(j,k,l,alpha))
        #
        costerm = cos(j * lambdaOut  - (j-k) * lambdaIn )
        sinterm = sin(j * lambdaOut  - (j-k) * lambdaIn )
        #
        from sympy import binomial, summation
        i=symbols('i')
        lBy2 = int(np.floor(l/2.))
        k_l = k-l
        k_lBy2 = int(np.floor((k_l)/2.))

        if l> 0:
            z_to_p_In_re = summation( binomial(l,(2*i)) * XIn**(l-(2*i)) * YIn**(2*i) * (-1)**(i) , (i, 0, lBy2 ))
            z_to_p_In_im = summation( binomial(l,(2*i+1)) * XIn**(l-(2*i+1)) * YIn**(2*i+1) * (-1)**(i) , (i, 0, lBy2 ))
        else:
            z_to_p_In_re = 1
            z_to_p_In_im = 0
        if k_l>0:
            z_to_p_Out_re = summation( binomial(k_l,(2*i)) * XOut**(k_l-(2*i)) * YOut**(2*i) * (-1)**(i) , (i, 0, k_lBy2 ))
            z_to_p_Out_im = summation( binomial(k_l,(2*i+1)) * XOut**(k_l-(2*i+1)) * YOut**(2*i+1) * (-1)**(i) , (i, 0, k_lBy2 ))
        else:
            z_to_p_Out_re = 1
            z_to_p_Out_im = 0
        reFactor = (z_to_p_In_re * z_to_p_Out_re - z_to_p_In_im * z_to_p_Out_im) / sqrt(LambdaIn)**l / sqrt(LambdaOut)**k_l
        imFactor = (z_to_p_In_im * z_to_p_Out_re + z_to_p_In_re * z_to_p_Out_im) / sqrt(LambdaIn)**l / sqrt(LambdaOut)**k_l
        #
        prefactor = -muOut *( mIn / MIn) / (LambdaOut**2)
        # Keep track of resonances
        self.resonance_indices.append((indexIn,indexOut,(j,k,l)))
        # Update Hamiltonian
        self.H += prefactor * Cjkl * (reFactor * costerm - imFactor * sinterm)
        self._update()
    def _get_symbols(self, index):
        return self.m[index], self.M[index], self.mu[index], self.Lambda[index], self.lam[index], self.X[index], self.Y[index]
    @property
    def NLambda(self):
        return self.integrator.y[::4]
    @property
    def Nlambda(self):
        return np.mod(self.integrator.y[1::4],2*np.pi)
    @property
    def NX(self):
        return self.integrator.y[2::4]
    @property
    def NT(self):
        return self.integrator.y[3::4]

class HamiltonianCombineEccentricityTransform(Hamiltonian):
    def __init__(self):
        self.resonance_indices = []
        self.integrator = None
        self.N = 0
        self.special_pairs = []
    def initialize_from_sim(self,sim):
        # add sympy symbols
        #
        #  parameters
        self.m = list(symbols("m0:{0}".format(sim.N)))
        self.M = list(symbols("M0:{0}".format(sim.N)))
        self.mu = list(symbols("mu0:{0}".format(sim.N)))
        #  canonical variables
        self.Psi = list(symbols("Psi0:{0}".format(sim.N)))
        self.psi = list(symbols("psi0:{0}".format(sim.N)))
        self.Phi = list(symbols("Phi0:{0}".format(sim.N)))
        self.phi = list(symbols("phi0:{0}".format(sim.N)))
    def initialize_from_PoincareHamiltonian(self,PHam):
        # add sympy symbols
        #
        #  parameters
        self.m = PHam.m
        self.M = PHam.M
        self.mu = PHam.mu
        self.N = PHam.N
        #  canonical variables
        self.special_indices = [ i for pair in special_pairs for i in pair ]
        self.Psi = list(symbols("Psi0:{0}".format(self.N)))
        self.psi = list(symbols("psi0:{0}".format(self.N)))
        self.Phi = list(symbols("Phi0:{0}".format(self.N)))
        self.phi = list(symbols("phi0:{0}".format(self.N)))
       # for i in self.special_indices:
       #     self.Phi[i]
       #     self.phi.remove(symbols("phi0:{0}".format(i))))
        
        self.H = S(0)
        self.params = PHam.params
        self.pqpairs = []
        for i in range(1,PHam.N):
            self.pqpairs.append((self.Psi[i],self.psi[i]))
            self.pqpairs.append((self.Phi[i], self.phi[i]))
            self.add_Hkep_term(i)
    
        self.a = PHam.a # add dummy for star to match indices
        self.Nm, self.NM, self.Nmu = PHam.Nm, PHam.NM, PHam.Nmu
        self.Nparams = self.Nmu + self.Nm + self.NM
        self.initial_conditions = PHam.initial_conditions
        self._update()

    def add_Hkep_term(self,index):
        m, M, mu, Lambda = self._get_HKep_symbols(index)
        self.H +=  -mu / (2 * Lambda**2)
    def _get_HKep_symbols(self, index):
        if index==1:
            Lambda = self.Phi[index]-self.Psi[index]
        else:
            Lambda = self.Phi[index] - self.Psi[index] + self.Psi[index-1]
        return self.m[index], self.M[index], self.mu[index], Lambda 

    def add_all_resonance_subterms(self, indexIn, indexOut, j, k):
        """
        Add all the terms associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -    index of the inner planet
        indexOut    -    index of the outer planet
        j           -    together with k specifies the MMR j:j-k
        k           -    order of the resonance
        """
        for l in range(k+1):
            self.add_single_resonance(indexIn,indexOut, j, k, l)

    def add_single_resonance(self, indexIn, indexOut, j, k, l):
        """
        Add a single term associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -   index of the inner planet
        indexOut    -   index of the outer planet
        j           -   together with k specifies the MMR j:j-k
        k           -   order of the resonance
        l           -   picks out the eIn^(l) * eOut^(k-l) subterm
        """
        # Canonical variables
        assert indexOut == indexIn + 1,"Only resonances for adjacent pairs are currently supported"
        mIn, MIn, muIn, mOut, MOut, muOut, LambdaIn, LambdaOut, psi, PhiIn, phiIn, PhiOut, phiOut = self._get_pair_symbols(indexIn,indexOut)
        
        # Resonance index
        assert l<=k, "Invalid resonance term, l must be less than or equal to k."
        alpha = self.a[indexIn]/self.a[indexOut]

        # Resonance components
        from celmech.disturbing_function import general_order_coefficient
        #
        Cjkl = symbols( "C_{0}\,{1}\,{2}".format(j,k,l) )
        self.params.append(Cjkl)
        self.Nparams.append(general_order_coefficient(j,k,l,alpha))
        #
        eccIn = sqrt(2*PhiIn/LambdaIn)
        eccOut = sqrt(2*PhiOut/LambdaOut)
        #
        costerm = cos( (j-k+l) * psi + l * phiIn + (k-l) * phiOut )
        #
        prefactor = -muOut *( mIn / MIn) / (LambdaOut**2)
        # Keep track of resonances
        self.resonance_indices.append((indexIn,indexOut,(j,k,l)))
        # Update Hamiltonian
        self.H += prefactor * Cjkl * (eccIn**l) * (eccOut**(k-l)) * costerm
        self._update()
    def _get_pair_symbols(self,indexIn,indexOut):
        mIn, MIn, muIn, LambdaIn = self._get_HKep_symbols(indexIn)
        mOut, MOut, muOut, LambdaOut = self._get_HKep_symbols(indexOut)
        psi = self.psi[indexIn]
        PhiIn, PhiOut = self.Phi[indexIn],self.Phi[indexOut]
        phiIn, phiOut = self.phi[indexIn],self.phi[indexOut]
        return mIn, MIn, muIn, mOut, MOut, muOut, LambdaIn, LambdaOut, psi, PhiIn, phiIn, PhiOut, phiOut

class HamiltonianPoincare(Hamiltonian):
    def __init__(self):
        self.resonance_indices = []
        self.integrator = None

    def initialize_from_sim(self, sim):
        # add sympy symbols
        self.m = list(symbols("m0:{0}".format(sim.N)))
        self.M = list(symbols("M0:{0}".format(sim.N)))
        self.mu = list(symbols("mu0:{0}".format(sim.N)))
        self.Lambda = list(symbols("Lambda0:{0}".format(sim.N)))
        self.lam = list(symbols("lambda0:{0}".format(sim.N)))
        self.Gamma = list(symbols("Gamma0:{0}".format(sim.N)))
        self.gamma = list(symbols("gamma0:{0}".format(sim.N)))
        
        # add symbols needed by base Hamiltonian class
        self.N = sim.N
        self.H = S(0)
        self.params = self.mu + self.m + self.M
        self._pqpairs = []
        for i in range(1,sim.N):
            self._pqpairs.append((self.Lambda[i],self.lam[i]))
            self._pqpairs.append((self.Gamma[i], self.gamma[i]))
            self.add_Hkep_term(i)
        
        # add numerical values
        self.a = [0]+[sim.particles[i].a for i in range(1,sim.N)] # add dummy for star to match indices
        self.Nm, self.NM, self.Nmu = jacobi_masses_from_sim(sim)
        self.Nparams = self.Nmu + self.Nm + self.NM
        self.initial_conditions = poincare_vars_from_sim(sim, average_synodic_terms=True)
        
        # calculate Hamilton's equations symbolically and substitute numerical values
        self._update()

    def add_Hkep_term(self, index):
        """
        Add the Keplerian component of the Hamiltonian for planet ''.
        """
        m, M, mu, Lambda, lam, Gamma, gamma = self._get_symbols(index)
        self.H +=  -mu / (2 * Lambda**2)

    def add_single_resonance(self, indexIn, indexOut, j, k, l):
        """
        Add a single term associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -   index of the inner planet
        indexOut    -   index of the outer planet
        j           -   together with k specifies the MMR j:j-k
        k           -   order of the resonance
        l           -   picks out the eIn^(l) * eOut^(k-l) subterm
        """
        # Canonical variables
        mIn, MIn, muIn, LambdaIn, lambdaIn, GammaIn, gammaIn = self._get_symbols(indexIn)
        mOut, MOut, muOut, LambdaOut, lambdaOut, GammaOut, gammaOut = self._get_symbols(indexOut)
        
        # Resonance index
        assert l<=k, "Invalid resonance term, l must be less than or equal to k."
        alpha = self.a[indexIn]/self.a[indexOut]

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
        prefactor = -muOut *( mIn / MIn) / (LambdaOut**2)
    
        # Keep track of resonances
        self.resonance_indices.append((indexIn,indexOut,(j,k,l)))
        # Update Hamiltonian
        self.H += prefactor * Cjkl * (eccIn**l) * (eccOut**(k-l)) * costerm
        self._update()
    
    def add_all_resonance_subterms(self, indexIn, indexOut, j, k):
        """
        Add all the terms associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -    index of the inner planet
        indexOut    -    index of the outer planet
        j           -    together with k specifies the MMR j:j-k
        k           -    order of the resonance
        """
        for l in range(k+1):
            self.add_single_resonance(indexIn,indexOut, j, k, l)

    def _get_symbols(self, index):
        return self.m[index], self.M[index], self.mu[index], self.Lambda[index], self.lam[index], self.Gamma[index], self.gamma[index]
    
    @property
    def NLambda(self):
        return self.integrator.y[::4]
    @property
    def Nlambda(self):
        return np.mod(self.integrator.y[1::4],2*np.pi)
    @property
    def NGamma(self):
        return self.integrator.y[2::4]
    @property
    def Ngamma(self):
        return np.mod(self.integrator.y[3::4],2*np.pi)

class FastHamiltonianPoincare(Hamiltonian):
    def __init__(self, sim):
        self.initial_conditions = poincare_vars_from_sim(sim, average_synodic_terms=True)
        self.integrator = ode(lambda s: None)
        self.integrator.set_initial_value(self.initial_conditions, 0)
        
    def initialize_from_sim(self, sim):
        self.initial_conditions = poincare_vars_from_sim(sim, average_synodic_terms=True)
    def add_Hkep_term(self, index):
        """
        Add the Keplerian component of the Hamiltonian for planet ''.
        """
        m, M, mu, Lambda, lam, Gamma, gamma = self._get_symbols(index)
        self.H +=  -mu / (2 * Lambda**2)

    def add_single_resonance(self, indexIn, indexOut, j, k, l):
        """
        Add a single term associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -   index of the inner planet
        indexOut    -   index of the outer planet
        j           -   together with k specifies the MMR j:j-k
        k           -   order of the resonance
        l           -   picks out the eIn^(l) * eOut^(k-l) subterm
        """
        # Canonical variables
        mIn, MIn, muIn, LambdaIn, lambdaIn, GammaIn, gammaIn = self._get_symbols(indexIn)
        mOut, MOut, muOut, LambdaOut, lambdaOut, GammaOut, gammaOut = self._get_symbols(indexOut)
        
        # Resonance index
        assert l<=k, "Invalid resonance term, l must be less than or equal to k."
        alpha = self.a[indexIn]/self.a[indexOut]

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
        prefactor = -muOut *( mIn / MIn) / (LambdaOut**2)
    
        # Keep track of resonances
        self.resonance_indices.append((indexIn,indexOut,(j,k,l)))
        # Update Hamiltonian
        self.H += prefactor * Cjkl * (eccIn**l) * (eccOut**(k-l)) * costerm
        self._update()
    
    def add_all_resonance_subterms(self, indexIn, indexOut, j, k):
        """
        Add all the terms associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -    index of the inner planet
        indexOut    -    index of the outer planet
        j           -    together with k specifies the MMR j:j-k
        k           -    order of the resonance
        """
        for l in range(k+1):
            self.add_single_resonance(indexIn,indexOut, j, k, l)

    def _get_symbols(self, index):
        return self.m[index], self.M[index], self.mu[index], self.Lambda[index], self.lam[index], self.Gamma[index], self.gamma[index]
    
    @property
    def NLambda(self):
        return self.integrator.y[::4]
    @property
    def Nlambda(self):
        return np.mod(self.integrator.y[1::4],2*np.pi)
    @property
    def NGamma(self):
        return self.integrator.y[2::4]
    @property
    def Ngamma(self):
        return np.mod(self.integrator.y[3::4],2*np.pi)


