import numpy as np
from IPython.display import Math
try:
    from collections.abc import MutableMapping
except:
    from collections import MutableMapping
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig,diff,Matrix, series
from .hamiltonian import Hamiltonian,PhaseSpaceState
from .miscellaneous import poisson_bracket
from .disturbing_function import  _p1_p2_from_k_nu, evaluate_df_coefficient_delta_expansion, get_df_term_latex
from .disturbing_function import list_resonance_terms, list_secular_terms, _nucombos, _lcombos
from .disturbing_function import k_nu_depend_on_eccentricities, k_nu_depend_on_inclinations
from .disturbing_function import df_coefficient_C,get_df_coefficient_symbol,evaluate_df_coefficient_delta_expansion
from .nbody_simulation_utilities import reb_add_poincare_particle, reb_calculate_orbits
from itertools import combinations
import rebound
import warnings
from IPython.display import Math
def get_re_im_components(x,y,k):
    """
    Get the real and imaginary components of
        (x + sgn(k) * i y)^|k|
    """
    if k==0:
        return 1,0
    absk = abs(k)
    sgnk = np.sign(k)
    re,im=0,0
    for l in range(0,absk+1):
        b = binomial(absk,l)
        if l%2==0:
            re += b * (sgnk * y)**l * x**(absk-l) * (-1)**(l//2)
        else:
            im += b * (sgnk * y)**l * x**(absk-l) * (-1)**((l-1)//2)
    return re,im

def num_passed(iterable): # returns num of entries in iterable that are not None
    ctr = 0
    for i in iterable:
        if i is not None:
            ctr += 1
    return ctr

def _get_a0_symbol(i):
    return symbols(r"a_{{{0}\,0}}".format(i),positive=True)

def _get_Lambda0_symbol(i):
    return symbols(r"\Lambda_{{{0}\,0}}".format(i),positive=True)

class PoincareParticle(object):
    """
    A class representing an individual member (star, planet, or test particle) of a planetary system.
    The appropriate value for mu and M depends on the adopted coordinate system and Kepler splitting
    (see e.g., Hernandez and Dehnen 2017 for a review and comparison). celmech supports canonical
    heliocentric coordinates (default) and democratic heliocentric coordinates.

    Parameters 
    ----------
    coordinates: str
      Specifices the canonical coordinate system. This determines the appropriate definitions of mu and M. Options:
      'canonical heliocentric' (default): canonical heliocentric coordinates in the COM frame e.g. Laskar & Robutel 1995
      'democratic heliocentric': e.g. Duncan et al. 1998
    G : float
      Gravitational constant (Default: 0)
    m : float
      Physical mass of particle.
    Mstar : float
      Physical mass of central body.
    mu : float
      'Canonical' mass of body. mu=reduced mass for canonical heliocentric coordinates (default)
      mu=m for democratic heliocentric coordinates.
    M : float
      'Canonical' central mass. M=Mstar+m for canonical heliocentric coordinates (default)
      M=Mstar for democratic heliocentric coordinates.
    Lambda, sLambda, a: float
      These variables specify the semimajor axis of the orbit. 
      Can pass any of the three, but at least one must be specified.
    l : float
      Mean longitude of the orbit. If not passed, defaults to 0.
    Gamma, sGamma, e: float 
      These variables specify the orbital eccentricity. Any one can be passed. If none passed, defaults to 0
    gamma, pomega: float
      These variables specify the pericenter orientation. Any one can be passed. If none passed, defaults to 0
    Q, sQ, inc: float 
      These variables specify the orbital inclination. Any one can be passed. If none passed, defaults to 0
    q, Omega: float
      These variables specify the node longitude. Any one can be passed. If none passed, defaults to 0
    """
    def __init__(self, coordinates='canonical heliocentric', G=1., m=None, Mstar=None, mu=None, M=None, sLambda=None, l=None, sGamma=None, gamma=None, sQ=None, q=None, Lambda=None, Gamma=None, Q=None, a=None, e=None, inc=None, pomega=None, Omega=None, skappa=None, seta=None, ssigma=None, srho=None, kappa=None, eta=None, sigma=None, rho=None):
        """
        We store Cartesian components of specific actions to support test particles
        """
        self.G = G  
        self.coordinates = coordinates

        massError = False
        if m is not None and Mstar is not None: # passed both physical masses
            if mu is not None or M is not None: # also passed one of others
                massError = True
            else: # calculate from physical masses
                if coordinates == 'democratic heliocentric':
                    self.mu = m
                    self.M = Mstar
                elif coordinates == 'canonical heliocentric':
                    self.mu = m*Mstar/(Mstar + m)
                    self.M = Mstar+m
                else:
                    raise AttributeError("coordinates must either be 'canonical heliocentric' (default) or 'democratic heiocentric")
        else: # didn't pass both physical masses
            if m is not None or Mstar is not None: # passed only one physical mass
                massError = True
            elif mu is None or M is None: # didn't pass both can. masses
                massError = True
            else:
                self.mu = mu
                self.M = M

        if massError == True:   
            raise AttributeError("Have to either pass physical masses (m, Mstar) or 'canonical masses' (mu and M). Can't mix or pass both.")

        # if the variables we actually store are passed then we save them and are done
        num = num_passed([sLambda, l, skappa, seta, ssigma, srho])
        if num == 6:
            self.sLambda = sLambda
            self.l = l
            self.skappa = skappa
            self.seta = seta
            self.ssigma = ssigma
            self.srho = srho
            return
        
        # if passed our massive variables, just normalize by mu
        num = num_passed([Lambda, l, kappa, eta, sigma, rho])
        if num == 6:
            self.sLambda = Lambda/self.mu
            self.l = l
            self.skappa = kappa/np.sqrt(self.mu) # these are all sqrt(Gamma) variables so norm by sqrt(mass)
            self.seta = eta/np.sqrt(self.mu)
            self.ssigma = sigma/np.sqrt(self.mu)
            self.srho = rho/np.sqrt(self.mu)
            return

        # need to calculate the variables we store
        num = num_passed([sLambda, Lambda, a])
        if num == 0:
            raise AttributeError("Must pass exactly 1 of Lambda, sLambda (specific Lambda, i.e., per unit mass), and a.")
        elif num > 1:
            raise AttributeError("Can't pass more than 1 of Lambda, sLambda (specific Gamma, i.e. per unit mass), and a (semimajor axis)")
        num = num_passed([sGamma, Gamma, e])
        if num == 0:
            sGamma = 0 # default
        elif num > 1:
            raise AttributeError("Can't pass more than 1 of Gamma, sGamma (specific Gamma, i.e. per unit mass), and e (eccentricity)")
        num = num_passed([sQ, Q, inc])
        if num == 0:
            sQ = 0 # default
        elif num > 1:
            raise AttributeError("Can't pass more than 1 of Q, sQ (specific Q, i.e. per unit mass), and inc (inclination)")
        num = num_passed([gamma, pomega])
        if num == 0:
            gamma = 0 # default
        elif num > 1:
            raise AttributeError("Can't pass more than 1 of gamma (-pomega) and pomega (longitude of pericenter)")
        num = num_passed([q, Omega])
        if num == 0:
            q = 0 # default
        elif num > 1:
            raise AttributeError("Can't pass more than 1 of q (-Omega) and Omega (longitude of ascending node)")
        num = num_passed([l])
        if num == 0:
            l = 0 # default
       
        self.l = l

        if pomega is not None:
            gamma = -pomega
        if Omega is not None:
            q = -Omega

        if sLambda is not None:
            self.sLambda = sLambda
        elif Lambda is not None:
            try:
                self.sLambda = Lambda/self.mu
            except:
                raise AttributeError("Need to pass specific actions (sLambda, sGamma, and sQ) or a, e, and inc for test particles")
        elif a is not None:
            self.sLambda = np.sqrt(self.G*self.M*a)

        if Gamma is not None:
            try:
                sGamma = Gamma/self.mu
            except:
                raise AttributeError("Need to pass specific actions (sLambda, sGamma, and sQ) or a, e, and inc for test particles")
        elif e is not None:
            sGamma = self.sLambda*(1.-np.sqrt(1.-e**2))

        if Q is not None:
            try:
                sQ = Q/self.mu
            except:
                raise AttributeError("Need to pass specific actions (sLambda, sGamma, and sQ) or a, e, and inc for test particles")
        elif inc is not None:
            sQ = (self.sLambda - sGamma) * (1 - np.cos(inc))

        self.skappa = np.sqrt(2.*sGamma)*np.cos(gamma) # X per unit sqrt(mass)
        self.seta = np.sqrt(2.*sGamma)*np.sin(gamma)

        self.ssigma = np.sqrt(2.*sQ)*np.cos(q) # Xinc per unit sqrt(mass)
        self.srho = np.sqrt(2.*sQ)*np.sin(q)

    def _mu_M_to_m_Mstar(self, mu, M):
        """
        Takes reduced mass mu = mMstar/(Mstar+m) and M=Mstar+m
        and returns m and Mstar 
        """
        d = np.sqrt(M**2 - 4*mu*M) # m_0 - m_i
        Mstar = (M+d)/2
        m = mu*M/Mstar
        return m, Mstar
   
    @property
    def m(self):
        if self.coordinates == "democratic heliocentric":
            return self.mu
        elif self.coordinates == "canonical heliocentric":
            m, Mstar = self._mu_M_to_m_Mstar(self.mu, self.M)
            return m
    @m.setter
    def m(self, value):
        if self.coordinates == "democratic heliocentric":
            self.mu = value
        elif self.coordinates == "canonical heliocentric":
            raise AttributeError("Can't change physical masses after initialization with canonical heliocentric coordinates.")
    @property
    def Mstar(self):
        if self.coordinates == "democratic heliocentric":
            return self.M
        elif self.coordinates == "canonical heliocentric":
            m, Mstar = self._mu_M_to_m_Mstar(self.mu, self.M)
            return Mstar
    @Mstar.setter
    def Mstar(self, value):
        if self.coordinates == "democratic heliocentric":
            self.M = value
        elif self.coordinates == "canonical heliocentric":
            raise AttributeError("Can't change physical masses after initialization with canonical heliocentric coordinates.")
    @property
    def x(self):
        return (self.kappa - 1j * self.eta) / np.sqrt(2)
    @property
    def X(self):
        return self.x * np.sqrt(2 / self.Lambda)
    @property
    def y(self):
        return (self.sigma - 1j * self.rho) / np.sqrt(2)
    @property
    def Y(self):
        return self.y * np.sqrt(0.5 / self.Lambda)

    @property
    def xbar(self):
        return np.conj(self.x)
    @property
    def Xbar(self):
        return np.conj(self.X)
    @property
    def ybar(self):
        return np.conj(self.y)
    @property
    def Ybar(self):
        return np.conj(self.Y)

    @property
    def kappa(self):
        return np.sqrt(self.mu)*self.skappa
    @kappa.setter
    def kappa(self, value):
        self.skappa = value/np.sqrt(self.mu)
    @property
    def eta(self):
        return np.sqrt(self.mu)*self.seta
    @eta.setter
    def eta(self, value):
        self.seta = value/np.sqrt(self.mu)

    @property
    def sigma(self):
        return np.sqrt(self.mu)*self.ssigma
    @sigma.setter
    def sigma(self, value):
        self.ssigma = value/np.sqrt(self.mu)

    @property
    def rho(self):
        return np.sqrt(self.mu)*self.srho
    @rho.setter
    def rho(self, value):
        self.srho = value/np.sqrt(self.mu)

    @property
    def Lambda(self):
        return self.mu*self.sLambda
    @Lambda.setter
    def Lambda(self, value):
        self.sLambda = value/self.mu

    @property
    def Gamma(self):
        return self.mu*(self.skappa**2+self.seta**2)/2.
    @Gamma.setter
    def Gamma(self, value):
        self.sGamma = value/self.mu

    @property
    def Q(self):
        return self.mu*(self.ssigma**2+self.srho**2)/2.
    @Q.setter
    def Q(self, value):
        self.sQ = value/self.mu

    @property
    def sGamma(self):
        return (self.skappa**2+self.seta**2)/2.
    @property
    def gamma(self):
        return np.arctan2(self.seta, self.skappa)

    @property
    def sQ(self):
        return (self.ssigma**2+self.srho**2)/2.
    @property
    def q(self):
        return np.arctan2(self.srho,self.ssigma)

    @property
    def a(self):
        return self.sLambda**2/self.G/self.M
    @property
    def n(self):
        return self.G**2*self.M**2/self.sLambda**3
    @property
    def P(self):
        return 2 * np.pi / self.n
    @property
    def e(self):
        GbyL = self.sGamma/self.sLambda
        if 1-(1.-GbyL)*(1.-GbyL) < 0:
            raise AttributeError("sGamma:{0}, sLambda:{1}, GbyL:{2}, val:{3}".format(self.sGamma, self.sLambda, GbyL, 1-(1.-GbyL)*(1.-GbyL)))
        return np.sqrt(1 - (1-GbyL)*(1-GbyL))
    @property
    def inc(self):
        QbyLminusG = self.sQ / (self.sLambda - self.sGamma)
        cosi = 1 - QbyLminusG
        if np.abs(cosi) > 1:
            raise AttributeError("sGamma:{0}, sLambda:{1}, sQ:{2}, cosi:{3}".format(self.sGamma, self.sLambda, self.sQ,cosi))
        return np.arccos(cosi)
    @property
    def pomega(self):
        return -self.gamma
    @property
    def Omega(self):
        return -self.q

    def __repr__(self):
        """ 
        Returns a string with the state of the particle.
        """ 
        return '<{0}.{1} object, mu={2} M={3} sLambda={4} l={5} skappa={6} seta={7} srho={8} ssigma={9}>'.format(self.__module__, type(self).__name__, self.mu, self.M, self.sLambda, self.l, self.skappa, self.seta, self.srho, self.ssigma)

class PoincareParticles(MutableMapping):
    """
    """
    def __init__(self, poincare):
        self.poincare = poincare 

    def __getitem__(self, i):
        # go from int key and generate a PoincareParticle
        # need G and masses
        if i == 0:
            return PoincareParticle(G=np.nan, m=np.nan, Mstar=np.nan, l=np.nan, eta=np.nan, rho=np.nan, Lambda=np.nan, kappa=np.nan, sigma=np.nan)    #raise AttributeError("No Poincare elements for the central star")
        p = self.poincare
        if isinstance(i, slice):
            return [self[i] for i in range(*i.indices(p.N))]

        if i < 0: # accept negative indices
            i += p.N
        if i < 0 or i >= p.N:
            raise AttributeError("Index {0} used to access particles out of range.".format(i))
   
        if p.masses[i] == 0: 
            raise AttributeError("Current implementation of Poincare does not work with test particles")

        val = p.values
        j = i-1 # index starting at 0 instead of 1
        l = val[j * 3]
        eta = val[j * 3 + 1]
        rho = val[j * 3 + 2]
        Lambda = val[p.N_dof + j * 3]
        kappa = val[p.N_dof + j * 3 + 1]
        sigma = val[p.N_dof + j * 3 + 2]

        return PoincareParticle(coordinates=p.coordinates, G=p.G, m=p.masses[i], Mstar=p.masses[0], l=l, eta=eta, rho=rho, Lambda=Lambda, kappa=kappa, sigma=sigma)

    def __setitem__(self, key, value):
        # we could allow user to set only the stored variables sLambda, l, skappa etc.
        raise AttributeError("Can't set Poincare particle attributes")

    def __delitem__(self, key):
        raise AttributeError("deleting variables not implemented.")

    def __iter__(self):
        for p in self[:self.poincare.N]:
            yield p

    def __len__(self):
        return self.poincare.N

# Poincare is a phasespacestate with G, masses and coordinates
class Poincare(PhaseSpaceState):
    """
    A class representing a collection of Poincare particles constituting a planetary system.
    """
    def __init__(self, G, poincareparticles, coordinates="canonical heliocentric",t=0):
        # additional variables that need storing in addition to phasespacestate variables
        self.G = G
        self.masses = [poincareparticles[0].Mstar] + [p.m for p in poincareparticles]
        self.coordinates = coordinates
        initial_p_values = []
        initial_q_values = []
        qvars = []
        pvars = []

        for i,particle in enumerate(poincareparticles):
            q = list(symbols("lambda{0}, eta{0}, rho{0}".format(i+1)))
            p = list(symbols("Lambda{0}, kappa{0}, sigma{0}".format(i+1)))
            pvals = [particle.Lambda,particle.kappa,particle.sigma]
            qvals = [particle.l,particle.eta,particle.rho]
            initial_p_values += pvals
            initial_q_values += qvals
            qvars += q
            pvars += p
        qpvars = qvars + pvars
        initial_values = initial_q_values + initial_p_values
        super(Poincare,self).__init__(qpvars,initial_values,t=t) 
    
    @property
    def N(self):
        """
        Return total number of bodies, including central star (analogous to REBOUND sim.N)
        """
        return int(self.N_dof/3) + 1 # +1 for star

    @property
    def particles(self):
        particles = PoincareParticles(self)
        return particles

    @classmethod
    def from_Simulation(cls, sim, coordinates="canonical heliocentric"):
        """ 
        Convert REBOUND Simulation to Poincare object, using specified canonical coordinates.
        Assumes the dominant mass is sim.particles[0].

        Parameters 
        ----------
        
        sim : rebound.Simulation
          Simulation to convert.
        coordinates: str
          Specifices the canonical coordinate system. This determines the appropriate definitions of mu and M. Options:
          'canonical heliocentric' (default): canonical heliocentric coordinates in the COM frame e.g. Laskar & Robutel 1995
          'democratic heliocentric': e.g. Duncan et al. 1998
    
        Returns
        -------

        Poincare object
        """ 
        sim = sim.copy()
        # Move to COM frame so P0 = 0 in canonical heliocentric coordinates
        sim.move_to_com()
        particles = []
        #pvars = Poincare(G=sim.G, coordinates=coordinates,t=sim.t)
        o = reb_calculate_orbits(sim, coordinates=coordinates)
        for i in range(1,sim.N_real):
            orb = o[i-1]
            if orb.a <= 0. or orb.e >= 1.:
                raise AttributeError("Celmech error: Poincare.from_Simulation only support elliptical orbits. Particle {0}'s (heliocentric) a={1}, e={2}".format(i, orb.a, orb.e))
            # always pass physical masses, PoincareParticle will calculate appropriate canonical mass based on coord
            particle = PoincareParticle(m=sim.particles[i].m, Mstar=sim.particles[0].m, a=orb.a, l=orb.l, e=orb.e, pomega=orb.pomega, inc=orb.inc, Omega=orb.Omega,coordinates=coordinates,G=sim.G)
            particles.append(particle)
        return cls(G=sim.G,poincareparticles=particles, coordinates=coordinates,t=sim.t)

    def to_Simulation(self):
        """ 
        Convert Poincare object to a REBOUND simulation in COM frame.

        Returns
        -------
        sim : rebound.Simulation
        """ 

        sim = rebound.Simulation()
        sim.G = self.G
        sim.t = self.t
        ps = self.particles
        Mstar = ps[1].Mstar # use first Poincare particle to extract Mstar
        sim.add(m=Mstar)
        for i in range(1, self.N):
            reb_add_poincare_particle(ps[i], sim)
        return sim
    
    def copy(self):
        return Poincare(G=self.G, coordinates=self.coordinates, poincareparticles=self.particles[1:self.N],t=self.t)

# If we wanted Poincare.from_Hamiltonian, would need PoincareHamiltonian to hold both m_0 (for star) and coordinates
class PoincareHamiltonian(Hamiltonian):
    """
    A class representing the Hamiltonian governing the dynamical evolution of a system of particles,
    stored as a :class:`celmech.poincare.Poincare` instance.

    Attributes
    ----------
    H : sympy expression
        Symbolic expression for the Hamiltonian.
    NH : sympy expression
        Symbolic expression for the Hamiltonian with 
        numerical values of parameters substituted
        where applicable.
    N : int
        Number of particles
    particles : list
        List of :class:`celmech.poincare.PoincareParticle`s 
        making up the system.
    state : :class:`celmech.poincare.Poincare`
        A set of Poincare variables to which 
        transformations are applied.
    """
    def __init__(self, pvars):
        H_params = {symbols('G'):pvars.G}
        ps = pvars.particles
        H = S(0) 
        self.Lambda0s = [None] + [_get_Lambda0_symbol(i) for i in range(1,pvars.N)]
        for i in range(1, pvars.N):
            H_params[symbols("mu{0}".format(i))] = ps[i].mu
            H_params[symbols("m{0}".format(i))] = ps[i].m
            H_params[symbols("M{0}".format(i))] = ps[i].M
            H_params[self.Lambda0s[i]] = ps[i].Lambda
            H_params[_get_a0_symbol(i)] = ps[i].a
            for j in range(i+1,pvars.N):
                alpha_sym = symbols(r"\alpha_{{{0}\,{1}}}".format(i,j))
                alpha_val = ps[i].a/ps[j].a
                H_params[alpha_sym] = alpha_val

            H = self.add_Hkep_term(H, i)
        self.resonance_indices = []
        super(PoincareHamiltonian, self).__init__(H, H_params, pvars) 
    
    @property
    def particles(self):
        return self.state.particles

    @property
    def N(self):
        return len(self.particles)

    @property 
    def t(self):
        return self.state.t

    @property
    def df_raw_latex(self):
        """
        Get the raw latex string (r"...") for the terms currently included in the disturbing function.
        This property is likely not what you're looking for.
        Most users will either want df to pretty print df in a jupyter notebook,
        or df_latex for the actual latex (with single slashes) to copy-paste into a paper.
        """
        d = r""
        for i1, i2, (kvec, nuvec, lvec) in self.resonance_indices:
            d += get_df_term_latex(*kvec,*nuvec,*lvec,i1,i2)
        return d
    
    @property
    def df(self):
        """
        Pretty print the terms currently included in the disturbing function in a jupyter notebook.
        """
        display(Math(self.df_raw_latex))
    
    @property
    def df_latex(self):
        """
        Print the latex for the terms currently included in the disturbing function
        (e.g., to copy-paste into a paper)
        """
        print(self.df_raw_latex)
    
    def add_Hkep_term(self, H, index):
        """
        Add the Keplerian component of the Hamiltonian for planet ''.
        """
        G, M, mu, Lambda = symbols('G, M{0}, mu{0}, Lambda{0}'.format(index))
        #m, M, mu, Lambda, lam, Gamma, gamma = self._get_symbols(index)
        H +=  -G**2*M**2*mu**3 / (2 * Lambda**2)
        return H

    def add_cosine_term(self,k_vec,max_order=None,l_max=0,nu_vecs=None,l_vecs=None,indexIn=1,indexOut=2,eccentricities=True,inclinations=True):
        """
        Add a specific cosine term to the disturbing function up to an optional max_order in the eccentricities and inclinations.

        Arguments
        ---------
        k_vec : Tuple of 6 ints 
            Coefficients (k1, k2, k3, k4, k5, k6) for a cosine term of the form 
            cos(k1*lambdaOut + k2*lambdaIn + k3*pomegaIn + k4*pomegaOut + k5*OmegaIn + k6*OmegaOut)
            Note the ks must sum to zero, and by convention we write lambdaOut first, e.g., (3, -2, -1, 0, 0, 0) is
            cos(3*lambdaOut - 2*lambdaIn - pomega1)
        max_order : int, optional
            Maximum order to go to in the (combined) eccentricities and inclinations.
            Can specify either max_order OR nu_vecs (see below), but not both (most users will use max_order).
            If neither are passed, max_order defaults to the leading order of the cosine term specified by k_vec.
        l_max : int, optional
            Maximum degree of expansion in :math:`\delta = (\Lambda-\Lambda_0)/\Lambda_0` to include in cosine coefficients. Default is 0.
            Can specify either l_max OR l_vecs (see below), but not both (most users will use l_max).
        nu_vecs: List of 4-tuples, optional
            A list of the specific combinations of nu indices to include for the cosine term coefficient, e.g., [(0, 0, 0, 0), (1, 0, 0, 0), ...]
            See paper and examples for definition and use.
            Can specify either max_order OR nu_vecs, but not both (max_order makes more sense for most use cases).
        l_vecs: List of 2-tuples, optional
            A list of the specific combinations of l indices to include for the cosine term coefficient, e.g., [(0, 0), (1, 0), (2, 0), ...] 
            See paper and examples for definition and use.
            Can specify either l_max OR l_vecs, but not both (l_max makes more sense for most use cases).
            One use case for passing particular l_vecs is if one body is massless, so the other Lambda doesn't vary (and doesn't need expansion)
        indexIn : int, optional
            Index of inner planet. Default 1.
        indexOut : int, optional
            Index of outer planet. Default 2.
        eccentricities: bool, optional
            By default, includes all eccentricity terms.
            Can set to False to exclude any eccentricity terms (e.g., fixed circular orbits).
        inclinations: bool, optional
            By default, includes all inclination terms.
            Can set to False to exclude any inclination terms (e.g., co-planar systems).
        """
        k1,k2,k3,k4,k5,k6 = k_vec
        k_vec = tuple(k_vec) # ensure k_vec is a tuple
        if np.sum(k_vec) != 0:
            raise AttributeError("Invalid k_vec={0}. The coefficients must sum to zero to satisfy the d'Alembert relation (rotational symmetry).".format(k_vec))
        if (k5+k6) % 2 != 0:
            raise AttributeError("Invalid k_vec={0}. Last two entries ({1} and {2}) must be divisible by 2 to maintain problem's reflection symmetry.".format(k_vec, k5, k6))
        if nu_vecs:
            nu_vecs = [tuple(nu_vec) for nu_vec in nu_vecs]
            if max_order:
                raise AttributeError('Must pass EITHER max_order OR nu_vecs to add_cos_term, but not both. See docstring.')
        else:
            # this is the leading order for the cosine term chosen (k3+k4+k5+k6 = -(k1+k2) by d'Alembert)
            min_order = abs(k3) + abs(k4) + abs(k5) + abs(k6)
            if not max_order:
                max_order = min_order
            if max_order < min_order:
                raise AttributeError("Passed a k_vec={0} whose order={1} is higher than the specified max_order={2}".format(k_vec, min_order, max_order))
            nu_max = (max_order - min_order)//2 # each nu contributes 2 powers of e or i, so divide by 2 rounding down
            nu_vecs = []
            for nu_tot in range(nu_max+1):
                nu_vecs += _nucombos(nutot=nu_tot) # Make list of [(nu1, nu2, nu3, nu4), ... ] tuples
        if l_vecs:
            l_vecs = [tuple(l_vec) for l_vec in l_vecs]
            if l_max > 0:
                raise AttributeError('Can only pass l_max OR l_vecs to add_cos_term. See docstring.')
        else:
            l_vecs = []
            for l_tot in range(l_max+1):
                l_vecs += _lcombos(ltot=l_tot) # Make list of [(l1, l2), ... ] tuples
      
        try:
            if np.array(l_vecs).shape[1] != 2:
                raise
        except:
            raise AttributeError("l_vecs = {0} must be a list of (l1, l2) pairs, e.g., [(0, 0), (0, 1), ...]".format(l_vecs))
        if np.array(nu_vecs).shape[1] != 4:
            raise AttributeError("nu_vecs = {0} must be a list of (nu1, nu2, nu3, nu4) tuples, e.g., [(0, 0, 0, 0), (1, 0, 0, 0), ...]".format(nu_vecs))
        lmax = np.max([l1+l2 for l1, l2 in l_vecs]) # maximum total l=l1+l2

        G = symbols('G')
        mIn,muIn,MIn,LambdaIn,lambdaIn,kappaIn,etaIn,sigmaIn,rhoIn = symbols('m{0},mu{0},M{0},Lambda{0},lambda{0},kappa{0},eta{0},sigma{0},rho{0}'.format(indexIn)) 
        mOut,muOut,MOut,LambdaOut,lambdaOut,kappaOut,etaOut,sigmaOut,rhoOut = symbols('m{0},mu{0},M{0},Lambda{0},lambda{0},kappa{0},eta{0},sigma{0},rho{0}'.format(indexOut)) 
        
        Lambda0In,Lambda0Out = _get_Lambda0_symbol(indexIn),_get_Lambda0_symbol(indexOut)
        alpha_sym = symbols(r"\alpha_{{{0}\,{1}}}".format(indexIn,indexOut))
        alpha_val = self.H_params[alpha_sym]
        aOut0 = _get_a0_symbol(indexOut)
        deltaIn = (LambdaIn - Lambda0In) / Lambda0In
        deltaOut = (LambdaOut - Lambda0Out) / Lambda0Out
        # alpha = aIn/aOut
        # Resonance components
        
        for nu_vec in nu_vecs:
            if eccentricities == False and k_nu_depend_on_eccentricities(k_vec, nu_vec) == True:
                continue
            if inclinations == False and k_nu_depend_on_inclinations(k_vec, nu_vec) == True:
                continue

            C_dict = df_coefficient_C(*k_vec, *nu_vec)#k1,k2,k3,k4,k5,k6,nu1,nu2,nu3,nu4)
            p1,p2 = _p1_p2_from_k_nu(k_vec,nu_vec)
            C_delta_expansion_dict = evaluate_df_coefficient_delta_expansion(C_dict,p1,p2,lmax,alpha_val)
            Ctot = 0
            for l_vec,C_val in C_delta_expansion_dict.items():
                if l_vec not in l_vecs: # have to calculate all terms up to lmax, but only consider if in l_vecs
                    continue
                if l_vecs and (indexIn,indexOut,(k_vec,nu_vec,l_vec)) in self.resonance_indices:
                    warnings.warn("Cosine term k_vec={0}, nu_vec={1}, l_vec={2} already included Hamiltonian; no new term added.".format(k_vec, nu_vec, l_vec))
                    continue
                if l_vecs and (indexIn,indexOut,(tuple([-k for k in k_vec]),nu_vec,l_vec)) in self.resonance_indices:
                    warnings.warn("The same cosine term k_vec={0}, nu_vec={1}, l_vec={2} (with all k_vec entries made negative) already included Hamiltonian; no new term added to avoid duplication (cos(x) = cos(-x)).".format(k_vec, nu_vec, l_vec))
                    continue
                self.resonance_indices.append((indexIn,indexOut,(k_vec,nu_vec,l_vec)))
                Csym = get_df_coefficient_symbol(*k_vec,*nu_vec,*l_vec,indexIn,indexOut)
                self.H_params[Csym] = C_val
                l1,l2=l_vec
                Ctot += Csym * deltaIn**l1 * deltaOut**l2
            nu1,nu2,nu3,nu4 = nu_vec
            rtLIn = sqrt(Lambda0In)
            rtLOut = sqrt(Lambda0Out)
            xin,yin = get_re_im_components(kappaIn/rtLIn ,-etaIn / rtLIn,k3)
            xout,yout = get_re_im_components( kappaOut/rtLOut, -etaOut/rtLOut,k4)
            uin,vin = get_re_im_components(sigmaIn/rtLIn/2, -rhoIn/rtLIn/2,k5)
            uout,vout = get_re_im_components(sigmaOut/rtLOut/2, -rhoOut/rtLOut/2,k6)

            re = uin*uout*xin*xout - vin*vout*xin*xout - uout*vin*xout*yin - uin*vout*xout*yin - uout*vin*xin*yout - uin*vout*xin*yout - uin*uout*yin*yout + vin*vout*yin*yout
            im = uout*vin*xin*xout + uin*vout*xin*xout + uin*uout*xout*yin - vin*vout*xout*yin + uin*uout*xin*yout - vin*vout*xin*yout - uout*vin*yin*yout - uin*vout*yin*yout
            
            GammaIn = (kappaIn*kappaIn + etaIn*etaIn)/2
            GammaOut = (kappaOut*kappaOut + etaOut*etaOut)/2
            QIn = (sigmaIn*sigmaIn + rhoIn*rhoIn)/2
            QOut = (sigmaOut*sigmaOut + rhoOut*rhoOut)/2
            
            eIn_sq_term = (2 * GammaIn / Lambda0In )**nu3
            eOut_sq_term = (2 * GammaOut / Lambda0Out )**nu4
            incIn_sq_term = ( QIn / Lambda0In / 2 )**nu1
            incOut_sq_term = ( QOut / Lambda0Out / 2 )**nu2
            
            # Update internal Hamiltonian
            prefactor1 = -G * mIn * mOut / aOut0
            prefactor2 = eIn_sq_term * eOut_sq_term * incIn_sq_term * incOut_sq_term 
            trig_term = re * cos(k1 * lambdaOut + k2 * lambdaIn) - im * sin(k1 * lambdaOut + k2 * lambdaIn) 
            
            self.H += prefactor1 * Ctot * prefactor2 * trig_term
    
    def add_MMR_terms(self,p,q,max_order=None,l_max=0,indexIn=1,indexOut=2,eccentricities=True,inclinations=True):
        """
        Add all eccentricity and inclination disturbing function terms associated with a p:p-q mean
        motion resonance up to a given order.

        Arguments
        ---------
        p : int
            Coefficient of lambdaOut in resonant argument
                j*lambdaOut - (j-k)*lambdaIn
        q : int
            Order of the mean motion resonance.
        max_order : int
            Maximum order of terms to add.
        l_max : int, optional
            Maximum degree of expansion in :math:`\delta = (\Lambda-\Lambda_0)/\Lambda_0` to include in cosine coefficients. Default is 0.
        indexIn : int
            Index of inner planet.
        indexOut : int
            Index of outer planet.
        eccentricities: bool, optional
            By default, includes all eccentricity terms.
            Can set to False to exclude any eccentricity terms (e.g., fixed circular orbits).
        inclinations: bool, optional
            By default, includes all inclination terms.
            Can set to False to exclude any inclination terms (e.g., co-planar systems).
        """
        if p<q or q<0:
            warnings.warn("""
            MMRs with p<q or q<0 are not supported. 
            If you really want to include these terms, 
            they may be added individually with the 
            'add_cosine_term' method.
            """)
        if abs(p) % q == 0 and q != 1:
            warnings.warn("p and q share a common divisor. Some important terms may be omitted!")
        
        if not max_order:
            max_order = q
        if max_order < q:
            warnings.warn("""Maximum order is lower than order of the resonance!""")
        assert max_order>0, "max_order= {:d} not allowed, must be non-negative.".format(max_order)
  
        terms = list_resonance_terms(p,q,min_order=q,max_order=max_order,eccentricities=eccentricities,inclinations=inclinations)
        for k_vec, nu_vec in terms:
            self.add_cosine_term(k_vec=k_vec,l_max=l_max,nu_vecs=[nu_vec],indexIn=indexIn,indexOut=indexOut)
    
    def add_secular_terms(self,min_order=2,max_order=2,l_max=0,indexIn=1,indexOut=2,eccentricities=True,inclinations=True):
        """
        Add all eccentricity and inclination secular terms in the disturbing function from
        min_order to max_order in the eccentricities and inclinations.

        Arguments
        ---------
        min_order : int, optional
            Minimum order terms in the eccentricities and inclinations to add. Defaults to 2. 
        max_order : int, optional
            Maximum order terms in the eccentricities and inclinations to add. Defaults to 2.
        l_max : int, optional
            Maximum degree of expansion in :math:`\delta = (\Lambda-\Lambda_0)/\Lambda_0` to include in cosine coefficients. Default is 0.
            to include in cosine coefficients. Default is 0.
        indexIn : int
            Index of inner planet.
        indexOut : int
            Index of outer planet.
        eccentricities: bool, optional
            By default, includes all eccentricity terms.
            Can set to False to exclude any eccentricity terms (e.g., fixed circular orbits).
        inclinations: bool, optional
            By default, includes all inclination terms.
            Can set to False to exclude any inclination terms (e.g., co-planar systems).
        """
        assert max_order>0, "max_order= {:d} not allowed,  must be non-negative.".format(max_order)
  
        terms = list_secular_terms(min_order=min_order,max_order=max_order,eccentricities=eccentricities,inclinations=inclinations)
        for k_vec, nu_vec in terms:
            self.add_cosine_term(k_vec=k_vec,l_max=l_max,nu_vecs=[nu_vec],indexIn=indexIn,indexOut=indexOut)

    def add_orbit_average_J2_terms(self,J2,Rin,max_ei_order=None,max_delta_order=None,particles = 'all',**kwargs):
        r"""
        Add Hamiltonian terms that capture the orbit-averaged effect of 
        a central body's oblateness parameterized by the :math:`J_2`
        gravitational harmonic.

        Arguments
        ---------
        J2 : float
            The value of the central body's J2 gravitational
            harmonic.
        Rin : float
            The central body's radius
        max_ei_order : int, optional
            Maximum order of expansion in eccentricity and inclination.
            By default, the value is set to 'None' and no expansion in 
            eccentricity and inclinaion is done.
        max_delta_order : int, optional
            Maxmimum order in :math:`\delta =(\Lambda-\Lambda_0)/\Lambda_0).
            Default is 'None' and dependence on :math:`Lambda` is exact.
        particles : list, optional
            Which particle numbers to add :math:`J_2` terms for. Default
            is set to all particles.
        """
        G = symbols('G')
        J2_s = kwargs.get("J2_symbol",symbols("J2"))
        Rin_s = kwargs.get("Rin_symbol",symbols(r"R"))
        self.H_params[J2_s] = J2
        self.H_params[Rin_s] = Rin
        GJ2RinSq = G * J2_s * Rin_s * Rin_s 
        a0_d = symbols("a0")
        # dummy variables, substitute later
        # Lambda_d = symbols("Lambda")
        Lambda0_d = symbols("Lambda0")
        delta_d = symbols("delta")
        Lambda_d = Lambda0_d * (1 + delta_d)
        kappa_d,eta_d,sigma_d,rho_d = symbols('kappa,eta,sigma,rho')
        Gamma = (kappa_d * kappa_d + eta_d * eta_d) / 2
        G = Lambda_d - Gamma
        omesq = (G / Lambda_d) * (G / Lambda_d)
        esq = 1 - omesq
        Q = (sigma_d*sigma_d + rho_d*rho_d) / 2
        cosI = 1 - Q / G
        ssq = (1 - cosI) /2 
        a = a0_d * (1 + delta_d) * (1 + delta_d)
        rt_omesq = sqrt(1 - esq).expand()
        num = 3 * (ssq*ssq - ssq) + 1/S(2)
        denom = rt_omesq * rt_omesq * rt_omesq 
        full_exprn =  num / denom / a / a / a
        # Expand to max order if specified.
        # Otherwise the complete expression is used.
        if max_ei_order:
            eps = symbols("epsilon")
            # e/i expansion
            eps_exprn = full_exprn.subs({sym:eps*sym for sym in (kappa_d,eta_d,sigma_d,rho_d)})
            full_exprn = series(eps_exprn,eps,0,max_ei_order+1).removeO().subs({eps:1})
        if max_delta_order:
            # delta expansion
            full_exprn = series(full_exprn,delta_d,0,max_delta_order+1).removeO()
        Hpert = -1 * GJ2RinSq * full_exprn 
        if particles == 'all':
            pids = range(1,self.N)
        else:
            pids = particles
        for pid in pids:
            p = self.particles[pid]
            m,mu,M,kappa,eta,sigma,rho = symbols('m{0},mu{0},M{0},kappa{0},eta{0},sigma{0},rho{0}'.format(pid)) 
            Lambda = symbols('Lambda{0}'.format(pid)) 
            Lambda0 = _get_Lambda0_symbol(pid)
            delta = (Lambda - Lambda0)/Lambda0
            a0 = _get_a0_symbol(pid)
            delta = (Lambda - Lambda0)/Lambda0
            self.H += M * mu * Hpert.subs({a0_d:a0,delta_d:delta,kappa_d:kappa,eta_d:eta,sigma_d:sigma,rho_d:rho,Lambda0_d:Lambda0})

    def add_gr_potential_terms(self,c,max_e_order=None,particles = 'all'):
        r"""
        Add Hamiltonian terms that capture the orbital precession
        caused by general relativity by adding a potential term
        of the form 
        
        .. math::
             \phi_\mathrm{GR} = \frac{3G^2M_*^2}{c^2a^2\sqrt{1-e^2}}

        Arguments
        ---------
        c : float
            The speed of light in the appropriate simulation units.
            harmonic.
        max_e_order : int, optional
            Maximum order of expansion in eccentricity.
            By default, the value is set to 'None' and no expansion in 
            eccentricity is done.
        particles : list, optional
            Which particle numbers to add :math:`J_2` terms for. Default
            is set to all particles.
        """
        G,c_s = symbols('G,c')
        G_by_c = G / c_s
        self.H_params[c_s] = c
        
        # dummy variables, substitute later
        # Lambda_d = symbols("Lambda")
        a0_d = symbols("a0")
        Lambda0_d = symbols("Lambda0")
        kappa_d,eta_d,sigma_d,rho_d = symbols('kappa,eta,sigma,rho')
        Gamma = (kappa_d * kappa_d + eta_d * eta_d) / 2
        G = Lambda0_d - Gamma
        omesq = (G / Lambda0_d) * (G / Lambda0_d)
        esq = 1 - omesq
        full_exprn =  -3 * G_by_c * G_by_c / a0_d / a0_d / sqrt(omesq)
        # Expand to max order if specified.
        # Otherwise the complete expression is used.
        if max_e_order:
            eps = symbols("epsilon")
            # e expansion
            eps_exprn = full_exprn.subs({sym:eps*sym for sym in (kappa_d,eta_d)})
            full_exprn = series(eps_exprn,eps,0,max_e_order+1).removeO().subs({eps:1})
        if particles == 'all':
            pids = range(1,self.N)
        for pid in pids:
            p = self.particles[pid]
            m,mu,M,kappa,eta = symbols('m{0},mu{0},M{0},kappa{0},eta{0}'.format(pid)) 
            Lambda0 = _get_Lambda0_symbol(pid)
            a0 = _get_a0_symbol(pid)
            self.H += M * M * mu * full_exprn.subs({a0_d:a0,kappa_d:kappa,eta_d:eta,Lambda0_d:Lambda0})

    def _get_laplace_lagrange_matrices(self):
        set_e_and_inc_zero_rule = {
            S('{0}{1}'.format(var,i)):0
           for i in range(1,self.N)
            for var in ['eta','kappa','rho','sigma']
        }
        mtrx = []
        for s1 in [S('eta{}'.format(i)) for i in range(1,self.N)]:
            row = []
            for s2 in [S('kappa{}'.format(i)) for i in range(1,self.N)]:
                entry= diff(self.derivs[s1],s2)
                row.append(entry.subs(set_e_and_inc_zero_rule))
            mtrx.append(row)
        ecc_mtrx = Matrix(mtrx)
        mtrx = []
        for s1 in [S('rho{}'.format(i)) for i in range(1,self.N)]:
            row = []
            for s2 in [S('sigma{}'.format(i)) for i in range(1,self.N)]:
                entry= diff(self.derivs[s1],s2)
                row.append(entry.subs(set_e_and_inc_zero_rule))
            mtrx.append(row)
        inc_mtrx = Matrix(mtrx)
        return ecc_mtrx,inc_mtrx
