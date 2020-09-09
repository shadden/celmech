from .poincare import PoincareHamiltonian
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig,diff,Matrix
from sympy.functions import elliptic_f,elliptic_k
from sympy.core import pi

class FirstOrderGeneratingFunction(PoincareHamiltonian):
    def __init__(self,pvars):
        super(FirstOrderGeneratingFunction,self).__init__(pvars)
        self.H = S(0)

    @property
    def chi(self):
        return self.H
    @property
    def Nchi(self):
        return self.NH

    def add_zeroth_order_term(self,indexIn,indexOut,update=True):
        G = symbols('G')
        mIn,muIn,MIn,LambdaIn,lambdaIn = symbols('m{0},mu{0},M{0},Lambda{0},lambda{0}'.format(indexIn)) 
        mOut,muOut,MOut,LambdaOut,lambdaOut = symbols('m{0},mu{0},M{0},Lambda{0},lambda{0}'.format(indexOut)) 
        aOut_inv = G*MOut*muOut*muOut / LambdaOut / LambdaOut  
        prefactor = -G * mIn * mOut * aOut_inv
        psi = lambdaOut - lambdaIn
        aIn = (LambdaIn/muIn)**2 / G / MIn
        aOut = (LambdaOut/muOut)**2 / G / MOut
        alpha = aIn/aOut
        omega_syn = self.kvec_to_omega((1,-1,0,0,0,0),indexIn,indexOut)
        m = alpha*alpha
        psi_integral = elliptic_f(psi / 2, m ) - psi * elliptic_k(m) / pi - sin(psi) / sqrt(alpha)
        term = prefactor * psi_integral / omega_syn
        self.H += term
        if update:
            self._update()
        
    def get_mean_motion(self,index):
        G,mu,M,Lambda = symbols('G,mu{0},M{0},Lambda{0}'.format(index)) 
        return G*G*M*M*mu*mu*mu / Lambda / Lambda / Lambda
        
    def kvec_to_omega(self,kvec,indexIn,indexOut):
        nIn = self.get_mean_motion(indexIn)
        nOut = self.get_mean_motion(indexOut)
        return kvec[0] * nOut + kvec[1] * nIn

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
        omega = self.kvec_to_omega(kvec,indexIn,indexOut)
        omega_inv = 1/omega
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
        trig_term = re * sin(k1 * lambdaOut + k2 * lambdaIn) + im * cos(k1 * lambdaOut + k2 * lambdaIn) 
        
        
        # Keep track of resonances
        self.resonance_indices.append((indexIn,indexOut,(kvec,zvec)))
        
        self.H += prefactor1 * C * prefactor2 * trig_term * omega_inv

        if update:
            self._update()
