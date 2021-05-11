import numpy as np
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig,diff,Matrix
from .hamiltonian import Hamiltonian
from .disturbing_function import get_fg_coeffs , laplace_b
from .disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
from .nbody_simulation_utilities import reb_add_poincare_particle, reb_calculate_orbits
from itertools import combinations
import rebound
import warnings
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
    def __init__(self, coordinates='canonical heliocentric', G=1., m=None, Mstar=None, mu=None, M=None, sLambda=None, l=None, sGamma=None, gamma=None, sQ=None, q=None, Lambda=None, Gamma=None, Q=None, a=None, e=None, inc=None, pomega=None, Omega=None):
        """
        We store Cartesian components of specific actions to support test particles
        """
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
       
        self.G = G  
        self.l = l
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

class Poincare(object):
    """
    A class representing a collection of Poincare particles constituting a planetary system.
    """
    def __init__(self, G, poincareparticles=[], coordinates="canonical heliocentric"):
        self.G = G
        self.coordinates = coordinates
        self.particles = [PoincareParticle(coordinates=coordinates, G=G, m=np.nan, Mstar=np.nan, l=np.nan, gamma=np.nan,q=np.nan, sLambda=np.nan, sGamma=np.nan, sQ=np.nan)] # dummy particle for primary
        try:
            for p in poincareparticles:
                self.add(m=p.m, Mstar=p.Mstar, sLambda=p.sLambda, l=p.l, sGamma=p.sGamma, gamma=p.gamma, sQ=p.sQ, q=p.q)
        except TypeError:
            raise TypeError("poincareparticles must be a list of PoincareParticle objects")
    
    def add(self, **kwargs):
        self.particles.append(PoincareParticle(G=self.G, coordinates=self.coordinates, **kwargs))
        # TODO: update 0th particle for the remaining COM coordinate

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

        pvars = Poincare(G=sim.G, coordinates=coordinates)
        ps = sim.particles
        Mstar = ps[0].m
        o = reb_calculate_orbits(sim, coordinates=coordinates)
        for i in range(1,sim.N_real):
            orb = o[i-1]
            if orb.a <= 0. or orb.e >= 1.:
                raise AttributeError("Celmech error: Poincare.from_Simulation only support elliptical orbits. Particle {0}'s (heliocentric) a={1}, e={2}".format(i, orb.a, orb.e))
            # always pass physical masses, PoincareParticle will calculate appropriate canonical mass based on coord
            pvars.add(m=ps[i].m, Mstar=Mstar, a=orb.a, l=orb.l, e=orb.e, pomega=orb.pomega, inc=orb.inc, Omega=orb.Omega)
        return pvars

    def to_Simulation(self):
        """ 
        Convert Poincare object to a REBOUND simulation in COM frame.

        Returns
        -------
        sim : rebound.Simulation
        """ 

        sim = rebound.Simulation()
        sim.G = self.G
        ps = self.particles
        Mstar = ps[1].Mstar # use first Poincare particle to extract Mstar
        sim.add(m=Mstar)
        for i in range(1, self.N):
            reb_add_poincare_particle(ps[i], sim)
        return sim
    
    def copy(self):
        return Poincare(G=self.G, coordinates=self.coordinates, poincareparticles=self.particles[1:self.N])

    @property
    def N(self):
        return len(self.particles)

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
        Hparams = {symbols('G'):pvars.G}
        pqpairs = []
        ps = pvars.particles
        H = S(0) 
        for i in range(1, pvars.N):
            pqpairs.append(symbols("kappa{0}, eta{0}".format(i))) 
            pqpairs.append(symbols("Lambda{0}, lambda{0}".format(i))) 
            pqpairs.append(symbols("sigma{0}, rho{0}".format(i))) 
            Hparams[symbols("mu{0}".format(i))] = ps[i].mu
            Hparams[symbols("m{0}".format(i))] = ps[i].m
            Hparams[symbols("M{0}".format(i))] = ps[i].M
            H = self.add_Hkep_term(H, i)
        self.resonance_indices = []
        super(PoincareHamiltonian, self).__init__(H, pqpairs, Hparams, pvars) 
    
    @property
    def particles(self):
        return self.state.particles

    @property
    def N(self):
        return len(self.particles)
    
    def state_to_list(self, state):
        ps = state.particles
        vpp = 6 # vars per particle
        y = np.zeros(vpp*(state.N-1)) # remove padded 0th element in ps for y
        for i in range(1, state.N):
            y[vpp*(i-1)] = ps[i].kappa
            y[vpp*(i-1)+1] = ps[i].eta
            y[vpp*(i-1)+2] = ps[i].Lambda
            y[vpp*(i-1)+3] = ps[i].l 
            y[vpp*(i-1)+4] = ps[i].sigma
            y[vpp*(i-1)+5] = ps[i].rho
        return y
    def set_secular_mode(self):
        # 
        state = self.state
        for i in range(1,state.N):
            Lambda0,Lambda = symbols("Lambda{0}0 Lambda{0}".format(i))
            self.H = self.H.subs(Lambda,Lambda0)
            self.Hparams[Lambda0] = state.particles[i].Lambda
        self._update()

    def set_planar_mode(self):
        state = self.state
        ps = state.particles
        for i in xrange(1,state.N):
            rho,sigma = symbols("rho{0} sigma{0}".format(i))
            self.H = self.H.subs({rho:0,sigma:0})
            ps[i].srho = 0
            ps[i].ssigma = 0
        self._update()   

    def update_state_from_list(self, state, y):
        ps = state.particles
        vpp = 6 # vars per particle
        for i in range(1, state.N):
            ps[i].skappa = y[vpp*(i-1)]/np.sqrt(ps[i].mu)
            ps[i].seta = y[vpp*(i-1)+1]/np.sqrt(ps[i].mu)
            ps[i].sLambda = y[vpp*(i-1)+2]/ps[i].mu
            ps[i].l = y[vpp*(i-1)+3]
            ps[i].ssigma = y[vpp*(i-1)+4] / np.sqrt(ps[i].mu) 
            ps[i].srho = y[vpp*(i-1)+5] / np.sqrt(ps[i].mu) 
            
    
    def add_Hkep_term(self, H, index):
        """
        Add the Keplerian component of the Hamiltonian for planet ''.
        """
        G, M, mu, Lambda = symbols('G, M{0}, mu{0}, Lambda{0}'.format(index))
        #m, M, mu, Lambda, lam, Gamma, gamma = self._get_symbols(index)
        H +=  -G**2*M**2*mu**3 / (2 * Lambda**2)
        return H
    def add_monomial_term(self,kvec,zvec,indexIn=1,indexOut=2,update=True):
        """
        Add individual monomial term to Hamiltonian. The term 
        is specified by 'kvec', which specifies the cosine argument
        and 'zvec', which specfies the order of inclination and
        eccentricities in the Taylor expansion of the 
        cosine coefficient. 
        """
        if (indexIn,indexOut,(kvec,zvec)) in self.resonance_indices:
            warnings.warn("Monomial term alread included Hamiltonian; no new term added.")
            return
        G = symbols('G')
        mIn,muIn,MIn,LambdaIn,lambdaIn,kappaIn,etaIn,sigmaIn,rhoIn = symbols('m{0},mu{0},M{0},Lambda{0},lambda{0},kappa{0},eta{0},sigma{0},rho{0}'.format(indexIn)) 
        mOut,muOut,MOut,LambdaOut,lambdaOut,kappaOut,etaOut,sigmaOut,rhoOut = symbols('m{0},mu{0},M{0},Lambda{0},lambda{0},kappa{0},eta{0},sigma{0},rho{0}'.format(indexOut)) 
        
        alpha = self.particles[indexIn].a/self.state.particles[indexOut].a
	# aIn = LambdaIn * LambdaIn / mIn / mIn / G / MIn
	# aOut = LambdaOut * LambdaOut / mOut / mOut / G / MOut
        # alpha = aIn/aOut
        # Resonance components
        #
        k1,k2,k3,k4,k5,k6 = kvec
        z1,z2,z3,z4 = zvec
        C = get_DFCoeff_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4,indexIn,indexOut)
        C_dict = DFCoeff_C(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4)
        C_val = eval_DFCoeff_dict(C_dict,alpha)
        self.Hparams[C] = C_val
        rtLIn = sqrt(LambdaIn)
        rtLOut = sqrt(LambdaOut)
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
        
        eIn_sq_term = (2 * GammaIn / LambdaIn )**z3
        eOut_sq_term = (2 * GammaOut / LambdaOut )**z4
        incIn_sq_term = ( QIn / LambdaIn / 2 )**z1
        incOut_sq_term = ( QOut / LambdaOut / 2 )**z2
        
        # Update internal Hamiltonian
        aOut_inv = G*MOut*muOut*muOut / LambdaOut / LambdaOut  
        prefactor1 = -G * mIn * mOut * aOut_inv
        prefactor2 = eIn_sq_term * eOut_sq_term * incIn_sq_term * incOut_sq_term 
        trig_term = re * cos(k1 * lambdaOut + k2 * lambdaIn) - im * sin(k1 * lambdaOut + k2 * lambdaIn) 
        
        # Keep track of resonances
        self.resonance_indices.append((indexIn,indexOut,(kvec,zvec)))
        
        self.H += prefactor1 * C * prefactor2 * trig_term
        if update:
            self._update()
        
    def add_all_MMR_and_secular_terms(self,p,q,max_order,indexIn = 1, indexOut = 2):
        """
        Add all disturbing function terms associated with a p:p-q mean
        motion resonance along with secular terms up to a given order.

        Arguments
        ---------
        p : int
            Coefficient of lambdaOut in resonant argument
                j*lambdaOut - (j-k)*lambdaIn
        q : int
            Order of the mean motion resonance.

        """
        assert max_order>=0, "max_order= {:d} not allowed,  must be non-negative.".format(max_order)
        if p<q or q<0:
            warnings.warn("""
            MMRs with j<k or k<0 are not supported. 
            If you really want to include these terms, 
            they may be added individually with the 
            'add_monomial_term' method.
            """)
        if max_order < q:
            warnings.warn("""Maxmium order is lower than order of the resonance!""")
        if abs(p) % q == 0 and q != 1:
            warnings.warn("p and q share a common divisor. Some important terms may be omitted!")
        max_order_by_2 = max_order // 2
        for h in range(0,max_order_by_2+1):
            if h==0:
                k_lo = 0
            else:
                k_lo = -2 * max_order_by_2
            for k in range(k_lo,2 * max_order_by_2 + 1):
                s_hi = max_order-abs(h+k)-abs(h-k)
                if h==0 and k==0:
                    s_lo = 0
                else:
                    s_lo = -s_hi
                for s in range(s_lo,s_hi+1):
                    s1_hi = max_order - abs(h+k) - abs(h-k) - abs(s)
                    if h==0 and k==0 and s==0:
                        s1_lo = 0
                    else:
                        s1_lo = -s1_hi
                    for s1 in range(s1_lo,s1_hi+1):
                        k3 = -s
                        k5 = -h-k
                        k6 = k-h
                        k4 = -s1
                        tot = k3+k4+k5+k6
                        if -p * tot % q is 0:
                            k1 = -p * tot // (q)
                            k2 = (p-q) * tot // (q)
                            kvec = np.array([k1,k2,k3,k4,k5,k6],dtype=int)
                            if k1 < 0:
                                kvec *= -1
                            self.add_cos_term_to_max_order(kvec.tolist(),max_order,indexIn,indexOut,update=False)
        # Finish with update
        self._update()

    def add_eccentricity_MMR_terms(self,p,q,max_order,indexIn = 1, indexOut = 2,update=True):
        """
        Add all eccentricity-type disturbing function terms associated with a p:p-q mean
        motion resonance up to a given order.

        Arguments
        ---------
        p : int
            Coefficient of lambdaOut in resonant argument
                j*lambdaOut - (j-k)*lambdaIn
        q : int
            Order of the mean motion resonance.
        """
        assert max_order>=0, "max_order= {:d} not allowed,  must be non-negative.".format(max_order)
        if p<q or q<0:
            warnings.warn("""
            MMRs with j<k or k<0 are not supported. 
            If you really want to include these terms, 
            they may be added individually with the 
            'add_monomial_term' method.
            """)
        if max_order < q:
            warnings.warn("""Maxmium order is lower than order of the resonance!""")
        if abs(p) % q == 0 and q != 1:
            warnings.warn("p and q share a common divisor. Some important terms may be omitted!")
        for n in range(1,int(max_order//q) + 1):
            k1 = n * p
            k2 = n * (q-p)
            for l in range(0, n * q + 1):
                k3 = -l
                k4 = l - n*q
                kvec = [k1,k2,k3,k4,0,0]
                self.add_cos_term_to_max_order(kvec,max_order,indexIn,indexOut,update=False)
        # Finish with update
        if update:
            self._update()


    def add_all_secular_terms(self,max_order,indexIn = 1, indexOut = 2):
        """
        Add all secular disturbing function terms up to a given order.

        Arguments
        ---------
        max_order : int
            Maximum order of terms to include
        indexIn : int, optional
            Integer index of inner planet
        indexOut : int, optional
            Integer index of outer planet
        """
        assert max_order>=0, "max_order= {:d} not allowed,  must be non-negative.".format(max_order)
        raise RuntimeError("THIS METHOD NEEDS TO BE FIXED!!!")
        max_order_by_2 = max_order//2
        max_order_by_4 = max_order//4
        for a in range(0,max_order_by_4+1):
            b_hi = max_order_by_2 - 2 * a
            if a==0:
                b_lo = 0
            else:
                b_lo = -b_hi
            for b in range(b_lo,b_hi+1):
                c_hi = max_order_by_2 - abs(b) - 2 * a
                if a == 0 and b ==0:
                    c_lo = 0
                else:
                    c_lo = -c_hi
                for c in range(c_lo,c_hi+1):
                    k3 = a-b
                    k4 = a+b
                    k5 = -c-a
                    k6 = c-a
                    self.add_cos_term_to_max_order([0,0,k3,k4,k5,k6],max_order,indexIn,indexOut,update=False)

        # finish with update
        self._update()

    def add_cos_term_to_max_order(self,jvec,max_order,indexIn=1,indexOut=2,update = True):
        """
        Add disturbing function term 
           c(alpha,e1,e2,s1,s2) * cos(j1 * lambda + j2 * lambda1 + j3 * pomega1 + j4 * pomega2 + j5 * Omega1 + j6 * Omega2)
        approximating c up to order 'max_order' in eccentricity and inclination.

        Arguments
        ---------
        jvec : array-like
            Vector of integers specifying cosine argument.
        max_order : int
            Maximum order of terms in include in the expansion of c
        indexIn : int, optional
            Integer index of inner planet.
        indexOut : anit, optional
            Intgeger index of outer planet.
        """
        _,_,j3,j4,j5,j6 = jvec
        order = max_order - abs(j3) - abs(j4) - abs(j5) - abs(j6)
        orderBy2 = order // 2
        N = orderBy2+1
        for z1 in range(0,N):
            for z2 in range(0,N - z1):
                for z3 in range(0,N - z1 - z2):
                    for z4 in range(0,N - z1 - z2 - z3):
                        zvec  = [z1,z2,z3,z4]
                        self.add_monomial_term(jvec,zvec,indexIn,indexOut,update=False)
        if update:
            self._update() 

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

