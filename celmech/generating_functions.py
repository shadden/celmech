import numpy as np
from .poincare import PoincareHamiltonian
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig,diff,Matrix
from sympy.functions import elliptic_f,elliptic_k
from sympy.core import pi
from .disturbing_function import DFCoeff_C,eval_DFCoeff_dict,get_DFCoeff_symbol
from .poincare import get_re_im_components
import warnings

class FirstOrderGeneratingFunction(PoincareHamiltonian):
    """
    A class representing a generating function that maps 
    from the 'osculating' canonical variables of the full 
    :math:`N`-body problem to 'mean' variables of an 
    'averaged' Hamiltonian or normal form.  

    The generating function is constructed to first order in 
    planet-star mass ratios by specifying indivdual monomial
    terms to be eliminated from the full Hamiltonian.

    This class is a sub-class of 
    :class:`celmech.poincare.PoincareHamiltonian`
    and disturbing function terms to be eliminated are added 
    in the same manner that disturbing function terms can be
    added to 
    :class:`celmech.poincare.PoincareHamiltonian`.

    Attributes
    ----------
    chi : sympy expression
        Symbolic expression for the generating function.
    Nchi : sympy expression
        Symbolic expression for the generating function 
        with numerical values of parameters substituted
        where applicable.
    state : :class:`celmech.poincare.Poincare`
        A set of Poincare variables to which 
        transformations are applied.
    N : int
        Number of particles
    particles : list
        List of :class:`celmech.poincare.PoincareParticle`s 
        making up the system.
    """
    def __init__(self,pvars):
        super(FirstOrderGeneratingFunction,self).__init__(pvars)
        self.H = S(0)

    @property
    def chi(self):
        return self.H
    @property
    def Nchi(self):
        return self.NH

    def _get_approximate_corrections(self,y):
        corrections = np.zeros(y.shape)
        for i,deriv_func in enumerate(self.Nderivs):
            corrections[i] = deriv_func(*y)
        return corrections

    def osculating_to_mean_state_vector(self,y,approximate=True,**integrator_kwargs):
        r"""
        Convert a state vector of canonical variables of the 
        the un-averaged :math:`N`-body Hamiltonian to a state
        vector of mean variables used by a normal form.

        Arguments
        ---------
        y : array-like
          State vector of un-averaged variables.

        approximate : bool, optional
          If True, Lie transformation is computed approximately
          to first order.  In other words, the approximation

          .. math::
            \exp[{\cal L}_{\chi}]f \approx f + \left[f,\chi \right]

        is used. If False, the exact Lie transformation is computed
        numerically.

        Returns
        -------
        ymean : array-like
          State vector of transformed (averaged) variables.
        """
        if approximate:
            yarr = np.atleast_1d(y)
            corrections = self._get_approximate_corrections(yarr)
            return yarr - corrections
        else:
            self.integrator.set_initial_value(y,t=0)
            self.integrator.integrate(-1,**integrator_kwargs)
            return self.integrator.y

    def mean_to_osculating_state_vector(self,y,approximate=True,**integrator_kwargs):
        r"""
        Convert a state vector of canonical variables of mean 
        variables used by a normal form to un-averaged variables
        of the full :math:`N`-body Hamiltonian.

        Arguments
        ---------
        y : array-like
          State vector of 'averaged' canonical variables.

        approximate : bool, optional
          If True, Lie transformation is computed approximately
          to first order.  In other words, the approximation

          .. math::
            \exp[{\cal L}_{\chi}]f \approx f + \left[f,\chi \right]

        is used. If False, the exact Lie transformation is computed
        numerically.

        Returns
        -------
        yosc : array-like
          State vector of osculating canonical variables.
        """
        if approximate:
            yarr = np.atleast_1d(y)
            corrections = self._get_approximate_corrections(yarr)
            return yarr + corrections
        else:
            self.integrator.set_initial_value(y,t=0)
            self.integrator.integrate(+1,**integrator_kwargs)
            return self.integrator.y

    def osculating_to_mean(self,**integrator_kwargs):
        """
        Convert the :attr:`state <celmech.generating_functions.FirstOrderGeneratingFunction.state>`'s
        variables from osculating
        to mean canonical variables.
        """
        y_osc = self.state_to_list(self.state)
        y_mean = self.osculating_to_mean_state_vector(y_osc)
        self.update_state_from_list(self.state,y_mean)

    def mean_to_osculating(self,**integrator_kwargs):
        """
        Convert the :attr:`state <celmech.generating_functions.FirstOrderGeneratingFunction.state>`'s
        variables from osculating
        to mean canonical variables.
        """
        y_mean = self.state_to_list(self.state)
        y_osc = self.mean_to_osculating_state_vector(y_mean)
        self.update_state_from_list(self.state,y_osc)

    def add_zeroth_order_term(self,indexIn=1,indexOut=2,update=True):
        r"""
        Add generating function term that elimiates 
        planet-planet interactions to 0th order in 
        inclinations and eccentricities.
        
        The added generating function term cancels the term

        .. math:: 
            -\frac{Gm_im_j}{a_j}\left(\frac{1}{\sqrt{1+\alpha^2-2\cos(\lambda_j-\lambda_i)}} - \frac{1}{2}b_{1/2}^{(0)}(\alpha) -\frac{\cos(\lambda_j-\lambda_i)}{\sqrt{\alpha}} \right)

        from the Hamiltonian to first order in planet-star mass ratios.

        Arguments
        ---------
        indexIn : int, optional
          Index of inner planet
        indexOut : int, optional
          Index of outer planet
        update : bool, optional
          Whether the numerical values of the generating function
          should be updated. It may be desirable to this option
          to :code:`False` when numerous terms are being added.
        """
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
        om_alpha = 1-alpha
        om_alpha_sq = om_alpha * om_alpha
        F = elliptic_f(psi/2,-4 * alpha / om_alpha_sq) / om_alpha
        psi_integral = 2 * F - 2 * psi * elliptic_k(m) / pi - sin(psi) / sqrt(alpha)
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
        Add individual monomial term to generating function. The term 
        is specified by 'kvec', which specifies the cosine argument
        and 'zvec', which specfies the order of inclination and
        eccentricities in the Taylor expansion of the 
        cosine coefficient. 
        """
        if (indexIn,indexOut,(kvec,zvec)) in self.resonance_indices:
            warnings.warn("Monomial term k=({},{},{},{},{},{}) , z = ({},{},{},{}) already included Hamiltonian; no new term added.".format(*kvec,*zvec))
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
    
