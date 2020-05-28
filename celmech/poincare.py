import numpy as np
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig,diff,Matrix
from celmech.hamiltonian import Hamiltonian
from celmech.disturbing_function import get_fg_coeffs, general_order_coefficient, secular_DF,laplace_B, laplace_coefficient
from celmech.disturbing_function import DFCoeff_C,eval_DFCoeff_dict
from celmech.transformations import masses_to_jacobi, masses_from_jacobi
from celmech.resonances import resonance_jk_list
from itertools import combinations
import rebound
import warnings
def get_DFCoeff_symbol(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4,indexIn,indexOut):
    return symbols("C_{0}\,{1}\,{2}\,{3}\,{4}\,{5}^{6}\,{7}\,{8}\,{9};({10}\,{11})".format(
        k1,k2,k3,k4,k5,k6,z1,z2,z3,z4,indexIn,indexOut)
    )
def partitions_of_length_4(N):
    answer = set()
    for j1 in range(N+1):
        for j2 in range(N+1-j1):
            for j3 in range(N+1-j1-j2):
                j4 = N-j1-j2-j3
                answer.add((j1,j2,j3,j4))
    return answer
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

def single_true(iterable): # Returns true if only one element in the iterable is set 
    # make generator from iterable setting any zeros as valid entries (otherwise they evaluate to False)
    i = iter([item if item != 0 else True for item in iterable]) # make generator and set zeros to valid inputs
    return any(i) and not any(i) # any(i) True once first valid item found. not any(i) ensures no additional ones exist

class PoincareParticle(object):
    def __init__(self, m, M, l, gamma, q, G=1., sLambda=None, sGamma=None, sQ=None, Lambda=None, Gamma=None, Q=None,  a=None, e=None, inc=None):
        """
        We store the specific Lambda = sqrt(G*M*a) and specific Gamma = sLambda*(1-sqrt(1-e**2)) to support test particles
        """
        if not single_true([sLambda, Lambda, a]):
            raise AttributeError("Can only pass one of Lambda, sLambda (specific Lambda, i.e. per unit mass), or a (semimajor axis)")
        if not single_true([sGamma, Gamma, e]):
            raise AttributeError("Can only pass one of Gamma, sGamma (specific Gamma, i.e. per unit mass), or e (eccentricity)")
        if not single_true([sQ, Q, inc]):
            raise AttributeError("Can only pass one of Q, sQ (specific Q, i.e. per unit mass), or inc (inclination)")
        
        if sLambda:
            self.sLambda = sLambda
        elif Lambda:
            try:
                self.sLambda = Lambda/m
            except:
                raise AttributeError("Need to pass specific actions (sLambda, sGamma, and sQ) or a, e, and inc for test particles")
        elif a:
            self.sLambda = np.sqrt(G*M*a)

        if Gamma:
            try:
                sGamma = Gamma/m
            except:
                raise AttributeError("Need to pass specific actions (sLambda, sGamma, and sQ) or a, e, and inc for test particles")
        elif e:
            sGamma = self.sLambda*(1.-np.sqrt(1.-e**2))

        if Q:
            try:
                sQ = Q/m
            except:
                raise AttributeError("Need to pass specific actions (sLambda, sGamma, and sQ) or a, e, and inc for test particles")
        elif inc:
            sQ = (self.sLambda - self.sGamma) * (1 - np.cos(inc))

        self.skappa = np.sqrt(2.*sGamma)*np.cos(gamma) # X per unit sqrt(mass)
        self.seta = np.sqrt(2.*sGamma)*np.sin(gamma)

        self.ssigma = np.sqrt(2.*sQ)*np.cos(q) # Xinc per unit sqrt(mass)
        self.srho = np.sqrt(2.*sQ)*np.sin(q)

        self.m = m 
        self.M = M
        self.G = G
        self.l = l
        
   
    @property
    def kappa(self):
        return np.sqrt(self.m)*self.skappa
    @kappa.setter
    def kappa(self, value):
        self.skappa = value/np.sqrt(self.m)
    @property
    def eta(self):
        return np.sqrt(self.m)*self.seta
    @eta.setter
    def eta(self, value):
        self.seta = value/np.sqrt(self.m)

    @property
    def sigma(self):
        return np.sqrt(self.m)*self.ssigma
    @sigma.setter
    def sigma(self, value):
        self.ssigma = value/np.sqrt(self.m)

    @property
    def rho(self):
        return np.sqrt(self.m)*self.srho
    @rho.setter
    def rho(self, value):
        self.srho = value/np.sqrt(self.m)

    @property
    def Lambda(self):
        return self.m*self.sLambda
    @Lambda.setter
    def Lambda(self, value):
        self.sLambda = value/self.m

    @property
    def Gamma(self):
        return self.m*(self.skappa**2+self.seta**2)/2.
    @Gamma.setter
    def Gamma(self, value):
        self.sGamma = value/self.m

    @property
    def Q(self):
        return self.m*(self.ssigma**2+self.srho**2)/2.
    @Q.setter
    def Q(self, value):
        self.sQ = value/self.m

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
    @property
    def n(self):
        return np.sqrt(self.G*self.M/self.a**3)

class Poincare(object):
    def __init__(self, G, poincareparticles=[]):
        self.G = G
        self.particles = [PoincareParticle(m=np.nan, M=np.nan, G=np.nan, l=np.nan, gamma=np.nan,q=np.nan, sLambda=np.nan, sGamma=np.nan, sQ=np.nan)] # dummy particle for primary
        try:
            for p in poincareparticles:
                self.add(m=p.m, sLambda=p.sLambda, l=p.l, sGamma=p.sGamma, gamma=p.gamma, sQ = p.sQ,q=p.q, M=p.M)
        except TypeError:
            raise TypeError("poincareparticles must be a list of PoincareParticle objects")

    @classmethod
    def from_Simulation(cls, sim, average=True):
        masses = [p.m for p in sim.particles]
        mjac, Mjac = masses_to_jacobi(masses)
        pvars = Poincare(sim.G)
        ps = sim.particles
        o = sim.calculate_orbits(jacobi_masses=True)
        for i in range(1,sim.N-sim.N_var):
            M = Mjac[i]
            m = mjac[i]
            orb = o[i-1]
            if orb.a <= 0. or orb.e >= 1.:
                raise AttributeError("Celmech error: Poincare.from_Simulation only support elliptical orbits. Particle {0}'s (jacobi) a={1}, e={2}".format(i, orb.a, orb.e))
            sLambda = np.sqrt(sim.G*M*orb.a)
            sGamma = sLambda*(1.-np.sqrt(1.-orb.e**2))
            sQ = sLambda*np.sqrt(1.-orb.e**2) * (1 - np.cos(orb.inc))
            pvars.add(m=m, sLambda=sLambda, l=orb.l, sGamma=sGamma, sQ = sQ, gamma=-orb.pomega,q=-orb.Omega ,M=M)
        if average is True:
            pvars.average_synodic_terms()
        return pvars

    def to_Simulation(self, masses=None, average=True):
        ''' 
        if masses is None, will calculate physical masses from the jacobi ones.
        if masses is a list, will use those as the physical masses.
        '''

        if average is True:
            self.average_synodic_terms(inverse=True)

        if not masses:
            mjac = [p.m for p in self.particles]
            Mjac = [p.M for p in self.particles]
            masses = masses_from_jacobi(mjac, Mjac)

        sim = rebound.Simulation()
        sim.G = self.G
        sim.add(m=masses[0])
        ps = self.particles
        #print(mjac, Mjac, masses)
        for i in range(1, self.N):
            sim.add(m=masses[i], a=ps[i].a, e=ps[i].e,inc=ps[i].inc, pomega=-ps[i].gamma, l=ps[i].l,Omega=ps[i].Omega, jacobi_masses=True)
        sim.move_to_com()
        return sim
    
    def add(self, **kwargs):
        self.particles.append(PoincareParticle(G=self.G, **kwargs))

    def copy(self):
        return Poincare(self.G, self.particles[1:self.N])

    def average_resonant_terms(self, i1=1, i2=2, deltaP=0.03, exclude=[], order=2, inverse=False):
        """
        Do a canonical transformation to correct the Lambdas for the fact that we have implicitly
        averaged over all the resonant terms we do not include in the Hamiltonian.
        """
        ps = self.particles
        m1 = ps[i1].m
        m2 = ps[i2].m
        n1 = ps[i1].n
        n2 = ps[i2].n
        Pratio = n2/n1 # P1/P2
        alpha = ps[i1].a/ps[i2].a
        e1 = ps[1].e
        e2 = ps[2].e
        l1 = ps[i1].l
        l2 = ps[i2].l
        gamma1 = ps[i1].gamma
        gamma2 = ps[i2].gamma
        G = self.G
        prefac = G/ps[i2].a

        sum1 = 0.
        sum2 = 0.
        prevsum=0.
        jklist = resonance_jk_list(Pratio-deltaP, min(Pratio+deltaP, 0.995), order)
        for j,k in jklist:
            if [j,k] in exclude:
                continue
            prefac1 = (j-k)/(j*n2 - (j-k)*n1)
            prefac2 = j/(j*n2 - (j-k)*n1)
            theta = j*l2 - (j-k)*l1
            for l in range(k+1): # 0 to k inclusive
                Cjkl = general_order_coefficient(j,k,l,alpha)
                Bjkl = Cjkl*e1**l*e2**(k-l) 
                cosine = np.cos(theta + l*gamma1 - (l-k)*gamma2)
                sum1 += prefac1*Bjkl*cosine
                sum2 += prefac2*Bjkl*cosine
                denom = 1.-float(j-k)/j*n1/n2
                #print(j,k,(sum1-prevsum)*prefac*m2/ps[i1].sLambda, denom, 1./j*m2/ps[i2].M/denom)
            prevsum=sum1

        sum1 *= prefac
        sum2 *= prefac
        print(sum1*m2/ps[i1].sLambda, sum2*m1/ps[i2].sLambda)
        if inverse is True:
            sum1 *= -1
            sum2 *= -1
        ps[i1].sLambda += m2*sum1
        ps[i2].sLambda -= m1*sum2

    def average_synodic_terms(self, inverse=False):
        """
        Do a canonical transformation to correct the Lambdas for the fact that we have implicitly
        averaged over all the synodic terms we do not include in the Hamiltonian.
        """
        corrpvars = self.copy() # so we use original values when planet appears in more than one pair
        pairs = combinations(range(1,self.N), 2)
        #TODO assumes particles ordered going outward so a1 < a2 always. Sort first?
        for i1, i2 in pairs:
            ps = self.particles
            m1 = ps[i1].m
            m2 = ps[i2].m
            deltalambda = ps[i1].l-ps[i2].l
            G = self.G

            prefac = G/ps[i2].a/(ps[i1].n-ps[i2].n) 
            alpha = ps[i1].a/ps[i2].a
            summation = (1. + alpha**2 - 2*alpha*np.cos(deltalambda))**(-0.5)
            s = prefac*(alpha*np.cos(deltalambda)-summation+laplace_coefficient(0.5, 0, 0, alpha)/2.)
            if inverse is True:
                s *= -1
            corrpvars.particles[i1].sLambda += m2*s # prefac*m1*m2*s/m1 (sLambda=Lambda/m)
            corrpvars.particles[i2].sLambda -= m1*s
        
        for i, p in enumerate(self.particles):
            p.sLambda = corrpvars.particles[i].sLambda

    @property
    def N(self):
        return len(self.particles)

class PoincareHamiltonian(Hamiltonian):
    def __init__(self, pvars):
        Hparams = {symbols('G'):pvars.G}
        pqpairs = []
        ps = pvars.particles
        H = S(0) 
        for i in range(1, pvars.N):
            pqpairs.append(symbols("kappa{0}, eta{0}".format(i))) 
            pqpairs.append(symbols("Lambda{0}, lambda{0}".format(i))) 
            pqpairs.append(symbols("sigma{0}, rho{0}".format(i))) 
            Hparams[symbols("m{0}".format(i))] = ps[i].m
            Hparams[symbols("M{0}".format(i))] = ps[i].M
            H = self.add_Hkep_term(H, i)
        self.resonance_indices = []
        super(PoincareHamiltonian, self).__init__(H, pqpairs, Hparams, pvars) 
        # Create polar hamiltonian for display/familiarity for user
        self.Hpolar = S(0)
        for i in range(1, pvars.N):
            self.Hpolar = self.add_Hkep_term(self.Hpolar, i)

        # Don't re-compute secular DF terms, save a dictionary
        # of secular DF expansions indexed by order:
        self.secular_terms = {0:S(0)}
    
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
            ps[i].skappa = y[vpp*(i-1)]/np.sqrt(ps[i].m)
            ps[i].seta = y[vpp*(i-1)+1]/np.sqrt(ps[i].m)
            ps[i].sLambda = y[vpp*(i-1)+2]/ps[i].m
            ps[i].l = y[vpp*(i-1)+3]
            ps[i].ssigma = y[vpp*(i-1)+4] / np.sqrt(ps[i].m) 
            ps[i].srho = y[vpp*(i-1)+5] / np.sqrt(ps[i].m) 
            
    
    def add_Hkep_term(self, H, index):
        """
        Add the Keplerian component of the Hamiltonian for planet ''.
        """
        G, M, m, Lambda = symbols('G, M{0}, m{0}, Lambda{0}'.format(index))
        #m, M, mu, Lambda, lam, Gamma, gamma = self._get_symbols(index)
        H +=  -G**2*M**2*m**3 / (2 * Lambda**2)
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
        mIn,MIn,LambdaIn,lambdaIn,kappaIn,etaIn,sigmaIn,rhoIn = symbols('m{0},M{0},Lambda{0},lambda{0},kappa{0},eta{0},sigma{0},rho{0}'.format(indexIn)) 
        mOut,MOut,LambdaOut,lambdaOut,kappaOut,etaOut,sigmaOut,rhoOut = symbols('m{0},M{0},Lambda{0},lambda{0},kappa{0},eta{0},sigma{0},rho{0}'.format(indexOut)) 
        
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
        prefactor1 = -G**2*MOut**2*mOut**3 *( mIn / MIn) / (LambdaOut**2)
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

    def add_eccentricity_MMR_terms(self,p,q,max_order,indexIn = 1, indexOut = 2):
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
                k3 = l
                k4 = n*q-l
                kvec = [k1,k2,k3,k4,0,0]
                self.add_cos_term_to_max_order(kvec.tolist(),max_order,indexIn,indexOut,update=False)
        # Finish with update
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
