import numpy as np
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig
from celmech.hamiltonian import Hamiltonian

from celmech.disturbing_function import get_fg_coeffs, general_order_coefficient, secular_DF,laplace_B, laplace_coefficient
from celmech.disturbing_function import DFCoeff_Cbar,eval_DFCoeff_dict
from celmech.transformations import masses_to_jacobi, masses_from_jacobi
from celmech.resonances import resonance_jk_list
from itertools import combinations
import rebound
import warnings

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
                raise AttributeError("Need to pass specific actions (sLambda and sGamma) or a and e for test particles")
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

    def update_state_from_list(self, state, y):
        ps = state.particles
        vpp = 4 # vars per particle
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
	
        # Resonance components
        #
        k1,k2,k3,k4,k5,k6 = kvec
        z1,z2,z3,z4 = zvec
        Cbar = symbols( "C_{0}\,{1}\,{2}\,{3}\,{4}\,{5}^{6}\,{7}\,{8}\,{9}".format(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4) )
        Cbar_dict = DFCoeff_Cbar(k1,k2,k3,k4,k5,k6,z1,z2,z3,z4)
        Cbar_val = eval_DFCoeff_dict(Cbar_dict,alpha)
        self.Hparams[Cbar] = Cbar_val
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
        
        self.H += prefactor1 * Cbar * prefactor2 * trig_term
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
    ###
    # Old methods
    ###
    def add_secular_terms(self, order=2,fixed_Lambdas=True, indexIn=1, indexOut=2):
        G = symbols('G')
        mOut,MOut,LambdaOut,lambdaOut,GammaOut,gammaOut,XOut,YOut = symbols('m{0},M{0},Lambda{0},lambda{0},Gamma{0},gamma{0},X{0},Y{0}'.format(indexOut)) 
        mIn,MIn,LambdaIn,lambdaIn,GammaIn,gammaIn,XIn,YIn = symbols('m{0},M{0},Lambda{0},lambda{0},Gamma{0},gamma{0},X{0},Y{0}'.format(indexIn)) 

        
        eIn,eOut,gammaIn,gammaOut = symbols("e e' gamma gamma'") 
        hIn,kIn,hOut,kOut=symbols("h,k,h',k'")
        # Work smarter not harder! Use an expression it is already available...
        if order not in self.secular_terms.keys():
            print("Computing secular expansion to order %d..."%order)
            subdict={eIn:sqrt(hIn*hIn + kIn*kIn), eOut:sqrt(hOut*hOut + kOut*kOut),gammaIn:atan2(-1*kIn,hIn),gammaOut:atan2(-1*kOut,hOut),}
            self.secular_terms[order] = secular_DF(eIn,eOut,gammaIn,gammaOut,order).subs(subdict,simultaneous=True)

        exprn = self.secular_terms[order]
        salpha = S("alpha{0}{1}".format(indexIn,indexOut))
        if order==2:
            subdict = { hIn:XIn/sqrt(LambdaIn) , kIn:(-1)*YIn/sqrt(LambdaIn), hOut:XOut/sqrt(LambdaOut) , kOut:(-1)*YOut/sqrt(LambdaOut), S("alpha"):salpha}
        else:
            # fix this to include higher order terms
            subdict = { hIn:XIn/sqrt(LambdaIn) , kIn:(-1)*YIn/sqrt(LambdaIn), hOut:XOut/sqrt(LambdaOut) , kOut:(-1)*YOut/sqrt(LambdaOut), S("alpha"):salpha}

        alpha = self.particles[indexIn].a/self.particles[indexOut].a
        self.Hparams[salpha] = alpha
        self.Hparams[Function('b')] = laplace_B
        exprn = exprn.subs(subdict)
        # substitute a fixed value for Lambdas in DF terms

        prefactor = -G**2*MOut**2*mOut**3 *( mIn / MIn) / (LambdaOut**2)
        exprn = prefactor * exprn
        if fixed_Lambdas:
            LambdaIn0,LambdaOut0=symbols("Lambda{0}0 Lambda{1}0".format(indexIn,indexOut))
            self.Hparams[LambdaIn0]=self.state.particles[indexIn].Lambda
            self.Hparams[LambdaOut0]=self.state.particles[indexOut].Lambda
            exprn = exprn.subs([(LambdaIn,LambdaIn0),(LambdaOut,LambdaOut0)])
        self.H += exprn
        self._update()

    def add_all_resonance_subterms(self, j, k, indexIn=1, indexOut=2):
        """
        Add all the terms associated the j:j-k MMR between planets 'indexIn' and 'indexOut'.
        Inputs:
        indexIn     -    index of the inner planet
        indexOut    -    index of the outer planet
        j           -    together with k specifies the MMR j:j-k
        k           -    order of the resonance
        """
        for l in range(k+1):
            self.add_single_resonance(j=j, k=k, l=l, indexIn=indexIn, indexOut=indexOut) 

    def add_single_resonance(self, j, k, l, indexIn=1, indexOut=2):
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
        G = symbols('G')
        mIn,MIn,LambdaIn,lambdaIn,XIn,YIn = symbols('m{0},M{0},Lambda{0},lambda{0},X{0},Y{0}'.format(indexIn)) 
        mOut,MOut,LambdaOut,lambdaOut,XOut,YOut = symbols('m{0},M{0},Lambda{0},lambda{0},X{0},Y{0}'.format(indexOut)) 
        #mIn, MIn, muIn, LambdaIn,lambdaIn,XIn,YIn = self._get_symbols(indexIn)
        #mOut, MOut, muOut, LambdaOut,lambdaOut,XOut,YOut = self._get_symbols(indexOut)
        
        # Resonance index
        assert l<=k, "Invalid resonance term, l must be less than or equal to k."
        alpha = self.particles[indexIn].a/self.state.particles[indexOut].a
	
        # Resonance components
        #
        Cjkl = symbols( "C_{0}\,{1}\,{2}".format(j,k,l) )
        self.Hparams[Cjkl] = general_order_coefficient(j,k,l,alpha)
        #
        
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
        # Keep track of resonances
        self.resonance_indices.append((indexIn,indexOut,(j,k,l)))
        # Update internal Hamiltonian
        prefactor = -G**2*MOut**2*mOut**3 *( mIn / MIn) / (LambdaOut**2)
        costerm = cos(j * lambdaOut  - (j-k) * lambdaIn )
        sinterm = sin(j * lambdaOut  - (j-k) * lambdaIn )
        
        self.H += prefactor * Cjkl * (reFactor * costerm - imFactor * sinterm)
        self._update()
        
        # update polar Hamiltonian
        GammaIn,gammaIn,GammaOut,gammaOut = symbols('Gamma{0},gamma{0},Gamma{1},gamma{1}'.format(indexIn, indexOut))
        eccIn = sqrt(2*GammaIn/LambdaIn)
        eccOut = sqrt(2*GammaOut/LambdaOut)
        #
        costerm = cos( j * lambdaOut - (j-k) * lambdaIn + l * gammaIn + (k-l) * gammaOut )
        #
        self.Hpolar += prefactor * Cjkl * (eccIn**l) * (eccOut**(k-l)) * costerm
