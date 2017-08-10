import numpy as np
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig
from celmech.hamiltonian import Hamiltonian
from celmech.disturbing_function import get_fg_coeffs, general_order_coefficient, secular_DF,laplace_B
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
        self.M = M
        self.particles = [None]
        try:
            for p in poincareparticles:
                self.particles.append(p)
        except TypeError:
            raise TypeError("poincareparticles must be a list of PoincareParticle objects")

    def add(self, m, Lambda, l, Gamma, gamma, M=None):
        if M is None:
            M = self.M
        self.particles.append(PoincareParticle(m, Lambda, l, Gamma, gamma, M))
    
    @classmethod
    def from_Simulation(cls, sim, average_synodic_terms=False):
        pvars = Poincare(sim.G, sim.particles[0].m)
        ps = sim.particles
        for i in range(1,sim.N):
            M = ps[0].m # TODO: add jacobi option ?
            m = ps[i].m
            Lambda = m*np.sqrt(sim.G*M*ps[i].a)
            Gamma = Lambda*(1.-np.sqrt(1.-ps[i].e**2))
            pvars.add(ps[i].m, Lambda, ps[i].l, Gamma, -ps[i].pomega, M)
        # TODO: if average_synodic_terms is True:    
        return pvars

    def to_Simulation(self):
        sim = rebound.Simulation()
        sim.G = self.G
        sim.add(m=self.M)
        ps = self.particles
        for i in range(1, self.N):
            a = ps[i].Lambda**2/ps[i].m**2/self.G/ps[i].M
            e = np.sqrt(1.-(1.-ps[i].Gamma/ps[i].Lambda)**2)
            sim.add(m=ps[i].m, a=a, e=e, pomega=-ps[i].gamma, l=ps[i].l)
        sim.move_to_com()
        return sim

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
