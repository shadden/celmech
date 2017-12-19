import numpy as np
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig
from celmech.hamiltonian import Hamiltonian
from celmech.disturbing_function import get_fg_coeffs, general_order_coefficient, secular_DF,laplace_B, laplace_coefficient
from celmech.transformations import masses_to_jacobi, masses_from_jacobi
from itertools import combinations
import rebound

class PoincareParticle(object):
    def __init__(self, m, Lambda, l, Gamma, gamma, M):
        X = np.sqrt(2.*Gamma)*np.cos(gamma)
        Y = np.sqrt(2.*Gamma)*np.sin(gamma)
        self.m = m
        self.M = M
        self.Lambda = Lambda
        self.l = l
        self.X = X
        self.Y = Y
    
    @property
    def Gamma(self):
        return (self.X**2+self.Y**2)/2.
    @property
    def gamma(self):
        return np.arctan2(self.Y, self.X)
    @property
    def eccentricity(self):
        GbyL = self.Gamma / self.Lambda
        return np.sqrt(1 - (1-GbyL)*(1-GbyL))

class Poincare(object):
    def __init__(self, G, M, poincareparticles=[]):
        self.G = G
        self.particles = [PoincareParticle(M, 0., 0., 0., 0., 0.)]
        try:
            for p in poincareparticles:
                self.particles.append(PoincareParticle(p.m, p.Lambda, p.l, p.Gamma, p.gamma, p.M))
        except TypeError:
            raise TypeError("poincareparticles must be a list of PoincareParticle objects")
    '''
    @classmethod
    def from_Simulation(cls, sim, average_synodic_terms=False):
        pvars = Poincare(sim.G, sim.particles[0].m)
        ps = sim.particles
        for i in range(1,sim.N):
            M = ps[0].m # TODO: add jacobi option ?
            m = ps[i].m
            orb = ps[i].calculate_orbit(primary=ps[0])
            Lambda = m*np.sqrt(sim.G*M*orb.a)
            Gamma = Lambda*(1.-np.sqrt(1.-orb.e**2))
            pvars.add(m, Lambda, orb.l, Gamma, -orb.pomega, M)
        if average_synodic_terms is True:
            pvars = pvars.synodic_Lambda_correction(inverse=False)
        return pvars

    def to_Simulation(self, average_synodic_terms=False):
        if average_synodic_terms is True:
            pvars = self.synodic_Lambda_correction(inverse=True)
        else:
            pvars = self

        sim = rebound.Simulation()
        sim.G = pvars.G
        sim.add(m=pvars.particles[0].m)
        ps = pvars.particles
        for p in ps[1:pvars.N]:#i in range(1, pvars.N):
            a = p.Lambda**2/p.m**2/pvars.G/p.M
            e = np.sqrt(1.-(1.-p.Gamma/p.Lambda)**2)
            sim.add(m=p.m, a=a, e=e, pomega=-p.gamma, l=p.l, primary=sim.particles[0])
        sim.move_to_com()
        return sim
    '''
    @classmethod
    def from_Simulation(cls, sim, average_synodic_terms=False):
        masses = [p.m for p in sim.particles]
        mjac, Mjac = masses_to_jacobi(masses)
        pvars = Poincare(sim.G, sim.particles[0].m)
        ps = sim.particles
        for i in range(1,sim.N):
            M = Mjac[i]#ps[0].m # TODO: add jacobi option ?
            m = mjac[i]#ps[i].m
            orb = ps[i].calculate_orbit()#primary=ps[0])
            Lambda = m*np.sqrt(sim.G*M*orb.a)
            Gamma = Lambda*(1.-np.sqrt(1.-orb.e**2))
            pvars.add(m, Lambda, orb.l, Gamma, -orb.pomega, M)
        if average_synodic_terms is True:
            pvars = pvars.synodic_Lambda_correction(inverse=False)
        return pvars

    def to_Simulation(self, average_synodic_terms=False):
        if average_synodic_terms is True:
            pvars = self.synodic_Lambda_correction(inverse=True)
        else:
            pvars = self

        mjac = [p.m for p in pvars.particles]
        Mjac = [p.M for p in pvars.particles]
        masses = masses_from_jacobi(mjac, Mjac)
        sim = rebound.Simulation()
        sim.G = pvars.G
        sim.add(m=masses[0])
        ps = pvars.particles
        for i in range(1, pvars.N):
            a = ps[i].Lambda**2/ps[i].m**2/pvars.G/ps[i].M
            e = np.sqrt(1.-(1.-ps[i].Gamma/ps[i].Lambda)**2)
            sim.add(m=masses[i], a=a, e=e, pomega=-ps[i].gamma, l=ps[i].l)#, primary=sim.particles[0])
        sim.move_to_com()
        return sim

    def copy(self):
        return Poincare(self.G, self.particles[0].m, self.particles[1:self.N])

    def synodic_Lambda_correction(pvars, inverse=False):
        """
        Do a canonical transformation to correct the Lambdas for the fact that we have implicitly
        averaged over all the synodic terms we do not include in the Hamiltonian.
        """
        corrpvars = pvars.copy()
        pairs = combinations(range(1,pvars.N), 2)
        for i1, i2 in pairs:
            s=0
            ps = pvars.particles
            L1 = ps[i1].Lambda
            L2 = ps[i2].Lambda
            m1 = ps[i1].m
            m2 = ps[i2].m
            M = ps[0].m 
            deltalambda = ps[i1].l-ps[i2].l
            G = pvars.G

            n1 = G**2*M**2*m1**3/L1**3
            n2 = G**2*M**2*m2**3/L2**3
            deltan = n1-n2
            alpha = (m2/m1*L1/L2)**2
            prefac = G**2*M**2*m2**3/L2**2*m1/M/deltan

            summation = (1. + alpha**2 - 2*alpha*np.cos(deltalambda))**(-0.5)
            s = prefac*(summation - 0.5*laplace_coefficient(0.5, 0, 0, alpha) - alpha*np.cos(deltalambda))
            if inverse is True:
                s *= -1
            print(0.5*m1/M*n2/n1*L2*laplace_coefficient(0.5, 0, 0, alpha)/s)
            corrpvars.particles[i1].Lambda -= s #+ 0.5*m1/M*n2/n1*L2*laplace_coefficient(0.5, 0, 0, alpha)
            corrpvars.particles[i2].Lambda += s

        return corrpvars
    
    '''

    def synodic_Lambda_correction(self, Ls, i1, i2, inverse=False):
        s=0
        ps = self.particles
        L1 = ps[i1].Lambda
        L2 = ps[i2].Lambda
        m1 = ps[i1].m
        m2 = ps[i2].m
        G = self.G
        M = self.M

        n1 = G**2*M**2*m1**3/L1**3
        n2 = G**2*M**2*m2**3/L2**3
        alpha = (m2/m1*L1/L2)**2
        deltan = n1-n2
        prefac = G**2*M**2*m2**3/L2**2*m1/M/deltan
        deltalambda = ps[i1].l-ps[i2].l

        summation = (1. + alpha**2 - 2*alpha*np.cos(deltalambda))**(-0.5)
        s = prefac*(summation - 0.5*laplace_coefficient(0.5, 0, 0, alpha) - alpha*np.cos(deltalambda))
        if inverse is True:
            s *= -1
        #print('correction', s/L1)
        Ls[i1] -= s
        Ls[i2] += s
    def synodic_Lambda_correction(self, Ls, i1, i2, inverse=False):
        """
        Do a canonical transformation to correct the Lambdas for the fact that we have implicitly
        averaged over all the synodic terms we do not include in the Hamiltonian.
        """
        s=0
        ps = self.particles
        m1 = ps[i1].m
        m2 = ps[i2].m
        G = self.G
        M = self.M

        n1 = G**2*M**2*m1**3/Ls[i1]**3
        n2 = G**2*M**2*m2**3/Ls[i2]**3
        alpha = (m2/m1*Ls[i1]/Ls[i2])**2
        deltan = n1-n2
        prefac = G**2*M**2*m2**3/Ls[i2]**2*m1/M/deltan
        deltalambda = ps[i1].l-ps[i2].l

        summation = (1. + alpha**2 - 2*alpha*np.cos(deltalambda))**(-0.5)
        s = prefac*(summation - 0.5*laplace_coefficient(0.5, 0, 0, alpha) - alpha*np.cos(deltalambda))
        if inverse is True:
            s *= -1
        print('correction', s/Ls[i1])
        return [Ls[i1]-s, Ls[i2]+s]  
    '''
    def add(self, m, Lambda, l, Gamma, gamma, M=None):
        if M is None:
            M = self.particles[0].m
        self.particles.append(PoincareParticle(m, Lambda, l, Gamma, gamma, M))
    

    def get_a(self, index):
        p = self.particles[index]
        return p.Lambda**2/p.m**2/self.G/p.M

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
            pqpairs.append(symbols("X{0}, Y{0}".format(i))) 
            pqpairs.append(symbols("Lambda{0}, lambda{0}".format(i))) 
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

    def state_to_list(self, state):
        ps = state.particles
        vpp = 4 # vars per particle
        y = np.zeros(vpp*(state.N-1)) # remove padded 0th element in ps for y
        for i in range(1, state.N):
            y[vpp*(i-1)] = ps[i].X  
            y[vpp*(i-1)+1] = ps[i].Y
            y[vpp*(i-1)+2] = ps[i].Lambda
            y[vpp*(i-1)+3] = ps[i].l 
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
            ps[i].X = y[vpp*(i-1)]
            ps[i].Y = y[vpp*(i-1)+1]
            ps[i].Lambda = y[vpp*(i-1)+2]
            ps[i].l = y[vpp*(i-1)+3]
    
    def add_Hkep_term(self, H, index):
        """
        Add the Keplerian component of the Hamiltonian for planet ''.
        """
        G, M, m, Lambda = symbols('G, M{0}, m{0}, Lambda{0}'.format(index))
        #m, M, mu, Lambda, lam, Gamma, gamma = self._get_symbols(index)
        H +=  -G**2*M**2*m**3 / (2 * Lambda**2)
        return H
    
    def add_secular_terms(self, indexIn, indexOut,order=2,fixed_Lambdas=True):

        
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

        alpha = self.state.get_a(indexIn)/self.state.get_a(indexOut)
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
        G = symbols('G')
        mIn,MIn,LambdaIn,lambdaIn,XIn,YIn = symbols('m{0},M{0},Lambda{0},lambda{0},X{0},Y{0}'.format(indexIn)) 
        mOut,MOut,LambdaOut,lambdaOut,XOut,YOut = symbols('m{0},M{0},Lambda{0},lambda{0},X{0},Y{0}'.format(indexOut)) 
        #mIn, MIn, muIn, LambdaIn,lambdaIn,XIn,YIn = self._get_symbols(indexIn)
        #mOut, MOut, muOut, LambdaOut,lambdaOut,XOut,YOut = self._get_symbols(indexOut)
        
        # Resonance index
        assert l<=k, "Invalid resonance term, l must be less than or equal to k."
        alpha = self.state.get_a(indexIn)/self.state.get_a(indexOut)
	
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
