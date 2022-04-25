import numpy as np
from .poincare import PoincareHamiltonian
from sympy import symbols, S, binomial, summation, sqrt, cos, sin, Function,atan2,expand_trig,diff,Matrix
from sympy import lambdify as sym_lambdify
from sympy.functions import elliptic_f,elliptic_k
from sympy.core import pi
from .disturbing_function import  _p1_p2_from_k_nu, evaluate_df_coefficient_delta_expansion
from .disturbing_function import df_coefficient_C,evaluate_df_coefficient_dict,get_df_coefficient_symbol
from .disturbing_function import  list_resonance_terms, _nucombos, _lcombos
from .poincare import get_re_im_components, _get_Lambda0_symbol, _get_a0_symbol
from .hamiltonian import _my_elliptic_e, _lambdify_kwargs
import warnings

class FirstOrderGeneratingFunction(PoincareHamiltonian):
    """
    A class representing a generating function that maps from the 'osculating'
    canonical variables of the full :math:`N`-body problem to 'mean' variables
    of an 'averaged' Hamiltonian or normal form.  

    The generating function is constructed to first order in planet-star mass
    ratios by specifying indivdual cosine terms to be eliminated from the full
    Hamiltonian.

    This class is a sub-class of :class:`celmech.poincare.PoincareHamiltonian`
    and disturbing function terms to be eliminated are added in the same manner
    that disturbing function terms can be added to
    :class:`celmech.poincare.PoincareHamiltonian`.

    Attributes
    ----------
    chi : sympy expression
        Symbolic expression for the generating function.
    N_chi : sympy expression
        Symbolic expression for the generating function with numerical values
        of paramteters substituted where applicable.
    state : :class:`celmech.poincare.Poincare`
        A set of Poincare variables to which transformations are applied.
    N : int
        Number of particles
    particles : list
        List of :class:`~celmech.poincare.PoincareParticle` objects making up
        the system.
    """
    def __init__(self,pvars):
        super(FirstOrderGeneratingFunction,self).__init__(pvars)
        self.H = S(0)

    @property
    def chi(self):
        return self.H
    @property
    def N_chi(self):
        return self.N_H

    def _get_approximate_corrections(self,y):
        return self.flow_func(*y).reshape(-1)
        #corrections = np.zeros(y.shape)
        #for i,deriv_func in enumerate(self.N_derivs):
        #    corrections[i] = deriv_func(*y)
        #return corrections

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

            is used. Otherwise, the exact Lie transformation is computed
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
        y_osc = self.state.values
        y_mean = self.osculating_to_mean_state_vector(y_osc)
        self.state.values = y_mean

    def mean_to_osculating(self,**integrator_kwargs):
        """
        Convert the :attr:`state <celmech.generating_functions.FirstOrderGeneratingFunction.state>`'s
        variables from osculating
        to mean canonical variables.
        """
        y_mean = self.state.values
        y_osc = self.mean_to_osculating_state_vector(y_mean)
        self.state.values = y_osc

    def add_zeroth_order_term(self,indexIn=1,indexOut=2):
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

    def get_mean_motion(self,index):
        G,mu,M,Lambda = symbols('G,mu{0},M{0},Lambda{0}'.format(index)) 
        return G*G*M*M*mu*mu*mu / Lambda / Lambda / Lambda

    def kvec_to_omega(self,kvec,indexIn,indexOut):
        nIn = self.get_mean_motion(indexIn)
        nOut = self.get_mean_motion(indexOut)
        return kvec[0] * nOut + kvec[1] * nIn

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
        if np.sum(k_vec) != 0:
            raise AttributeError("Invalid k_vec={0}. The coefficients must sum to zero to satisfy the d'Alembert relation.".format(k_vec))
        k1,k2,k3,k4,k5,k6 = k_vec
        k_vec = tuple(k_vec) # ensure k_vec is a tuple
        if nu_vecs:
            nu_vecs = [tuple(nu_vec) for nu_vec in nu_vecs]
            if max_order:
                raise AttributeError('Must pass EITHER max_order OR nu_vecs to add_cos_term, but not both. See docstring.')
        else:
            # this is the leading order for the cosine term chosen (k3+k4+k5+k6 = -(k1+k2) by d'Alembert)
            min_order = abs(k3) + abs(k4) + abs(k5) + abs(k6)
            if not max_order:
                max_order = min_order
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

        try:
            if np.array(nu_vecs).shape[1] != 4:
                raise
        except:
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
        
        omega = self.kvec_to_omega(k_vec,indexIn,indexOut)
        omega_inv = 1/omega
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
                    warnings.warn("Cosine term k_vec={0}, nu_vec={1}, l_vec={2} already included Hamiltonian; no new term added.".format(k_vec, nu_vec, l_vec))
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
            trig_term = re * sin(k1 * lambdaOut + k2 * lambdaIn) + im * cos(k1 * lambdaOut + k2 * lambdaIn)
            self.H += prefactor1 * Ctot * prefactor2 * trig_term * omega_inv

    def get_perturbative_solution(self,expression,free_variables=[],correction_only=False,lambdify=False,time_symbol=None):
        r"""
        Calculate a solution for the time evolution of an expression using
        first-order perturbation theory.

        Arguments
        ---------
        expression : sympy expression
            Expression in terms of canonical variables for which 
            to compute a perturbative solution.
        free_variables : list, optional
            List of canonical variables to leave undetermined in the 
            derived expression. By default, `free_variables` is empty
            and numerical values are substituted for all canonical 
            variables
        correction_only : bool, optional
            If `True`, only return the perturbative correction to 
            the `expression`. I.e., if `expression` is 
            :math:`f(q,p)` then only return :math:`[f(q,p),\chi(q,p)]`.
            Otherwise, return :math:`f(p,q) + [f(q,p),\chi(q,p)].
            Default is `False`.
        lambdify : bool, optional
            If `True`, return a function using sympy.lambdify.
            The list of arguments accepted by the returned function
            will the list `free_variables`, followed by the 
            time at which to evaluate the expression.
        time_symbol : sympy.Symbol, optional
            Symbol to use for denoting time. If no symbol is given,
            :math:`t` is used.
            
        
        Returns
        -------
        result: sympy expression or function
            An expression for the solution as a function of time.
        """
        subsrule = {k:v for k,v in self.qp.items() if k not in free_variables}

        if time_symbol is None:
            t = symbols('t')
        else:
            t = time_symbol

        for i in range(1,self.N):
            n=self.get_mean_motion(i).subs(self.H_params)
            lsymb = symbols("lambda{}".format(i))
            exprn = n * t + lsymb
            subsrule[lsymb] = exprn.subs(subsrule)

        if correction_only:
            pt_solution = self.N_Lie_deriv(expression)
        else:
            pt_solution = expression + self.N_Lie_deriv(expression)
        result = pt_solution.subs(subsrule)
        if lambdify:
            args = list(free_variables) + [t]
            return sym_lambdify(args,result , **_lambdify_kwargs)
        else:
            return result
