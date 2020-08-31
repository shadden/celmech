import rebound as rb
import reboundx
import numpy as np
import theano
import theano.tensor as T
from warnings import warn
from scipy.optimize import lsq_linear
from scipy.integrate import odeint
import warnings
from tqdm import tqdm

from .utils import planar_els2xv,calc_Hint_components_spatial
from ..nbody_simulation_utilities import get_canonical_heliocentric_orbits, add_canonical_heliocentric_elements_particle, align_simulation
from ..miscellaneous import getOmegaMatrix
# set to fast compile, for testing at least...


def _get_compiled_theano_functions(N_QUAD_PTS):
    # Planet masses: m1,m2
    m1,m2 = T.dscalars(2)
    mstar = 1
    mu1  = m1 * mstar / (mstar  + m1) 
    mu2  = m2 * mstar / (mstar  + m2) 
    Mstar1 = mstar + m1
    Mstar2 = mstar + m2
    beta1 = mu1 * T.sqrt(Mstar1/mstar) / (mu1 + mu2)
    beta2 = mu2 * T.sqrt(Mstar2/mstar) / (mu1 + mu2)
    j,k = T.lscalars('jk')
    s = (j-k) / k

    # Angle variable for averaging over
    psi = T.dvector()

    # Quadrature weights
    quad_weights = T.dvector('w')

    # Dynamical variables:
    Ndof = 3
    Nconst = 1
    dyvars = T.vector()
    y1, y2, y_inc, x1, x2, x_inc, amd  = [dyvars[i] for i in range(2*Ndof + Nconst)]
    
    a20 = T.constant(1.)
    a10 = ((j-k)/j)**(2/3) * (Mstar1 / Mstar2)**(1/3)
    L10 = beta1 * T.sqrt(a10)
    L20 = beta2 * T.sqrt(a20)
    Ltot = L10 + L20
    f = L10/L20
    L2res = (Ltot + amd) / (1+f)
    Psi = -k * (s * L2res + (1+s) * f * L2res) 
    ###
    # actions
    ###
    I1 = 0.5 * (x1*x1 + y1*y1)
    I2 = 0.5 * (x2*x2 + y2*y2)
    Phi = 0.5 * (x_inc*x_inc + y_inc*y_inc)
    L1 = -s*Ltot - Psi/k - s * (I1 + I2 + Phi)
    L2 = (1+s)*Ltot + Psi/k + (1+s) * (I1 + I2 + Phi)

    # Set lambda2=0
    l2 = T.constant(0.)
    l1 = -1 * k * psi 
    theta_res = (1+s) * l2 - s * l1
    cos_theta_res = T.cos(theta_res)
    sin_theta_res = T.sin(theta_res)
    
    kappa1 = x1 * cos_theta_res + y1 * sin_theta_res
    eta1   = y1 * cos_theta_res - x1 * sin_theta_res
    
    kappa2 = x2 * cos_theta_res + y2 * sin_theta_res
    eta2   = y2 * cos_theta_res - x2 * sin_theta_res
    
    sigma = x_inc * cos_theta_res + y_inc * sin_theta_res
    rho   = y_inc * cos_theta_res - x_inc * sin_theta_res
    # y = (sigma-i*rho)/sqrt(2)
    #   = sqrt(Phi) * exp[i (Omega1+Omega2) / 2]
    # Malige+ 2002,  Eqs 20 and 21
    r2byr1 = (L2 - L1 - I2 + I1) / Ltot
    sigma1 = rho * T.sqrt( 1 + r2byr1) / T.sqrt(2)
    sigma2 = -rho * T.sqrt( 1 - r2byr1) / T.sqrt(2)
    rho1 = -sigma * T.sqrt( 1 + r2byr1) / T.sqrt(2)
    rho2 = sigma * T.sqrt( 1 - r2byr1) / T.sqrt(2)

    Xre1 = kappa1 / T.sqrt(L1)
    Xim1 = -eta1 / T.sqrt(L1)
    Yre1 = 0.5 * sigma1 / T.sqrt(L1)
    Yim1 = -0.5 * rho1 / T.sqrt(L1)

    Xre2 = kappa2 / T.sqrt(L2)
    Xim2 = -eta2 / T.sqrt(L2)
    Yre2 = 0.5 * sigma2 / T.sqrt(L2)
    Yim2 = -0.5 * rho2 / T.sqrt(L2)

    absX1_sq = 2 * I1 / L1
    absX2_sq = 2 * I2 / L2
    X_to_z1 = T.sqrt(1 - absX1_sq / 4 )
    X_to_z2 = T.sqrt(1 - absX2_sq / 4 )
    Y_to_zeta1 =  1 / T.sqrt(1 - absX1_sq/2) 
    Y_to_zeta2 =  1 / T.sqrt(1 - absX2_sq/2) 

    a1 = (L1 / beta1 )**2 
    k1 = Xre1 * X_to_z1
    h1 = Xim1 * X_to_z1
    q1 = Yre1 * Y_to_zeta1
    p1 = Yim1 * Y_to_zeta1
    e1 = T.sqrt( absX1_sq ) * X_to_z1
    inc1 = 2*T.arcsin(T.sqrt(p1*p1+q1*q1))

    a2 = (L2 / beta2 )**2 
    k2 = Xre2 * X_to_z2
    h2 = Xim2 * X_to_z2
    q2 = Yre2 * Y_to_zeta2
    p2 = Yim2 * Y_to_zeta2
    e2 = T.sqrt( absX2_sq ) * X_to_z2
    inc2 = 2*T.arcsin(T.sqrt(p2*p2+q2*q2))

    beta1p = T.sqrt(Mstar1) * beta1
    beta2p = T.sqrt(Mstar2) * beta2
    Hkep = -0.5 * beta1p / a1 - 0.5 * beta2p / a2

    Hdir,Hind = calc_Hint_components_spatial(
            a1,a2,l1,l2,h1,k1,h2,k2,p1,q1,p2,q2,Mstar1,Mstar2
    )
    eps = m1*m2/ (mu1 + mu2) / T.sqrt(mstar)
    Hpert = (Hdir + Hind/mstar)
    Hpert_av = Hpert.dot(quad_weights)
    Htot = Hkep + eps * Hpert_av

    #####################################################
    # Set parameters for compiling functions with Theano
    #####################################################
    
    # Get numerical quadrature nodes and weights
    nodes,weights = np.polynomial.legendre.leggauss(N_QUAD_PTS)
    
    # Rescale for integration interval from [-1,1] to [-pi,pi]
    nodes = nodes * np.pi
    weights = weights * 0.5
    
    # 'givens' will fix some parameters of Theano functions compiled below
    givens = [(psi,nodes),(quad_weights,weights)]

    # 'ins' will set the inputs of Theano functions compiled below
    #   Note: 'extra_ins' will be passed as values of object attributes
    #   of the 'ResonanceEquations' class 'defined below
    extra_ins = [m1,m2,j,k]
    ins = [dyvars] + extra_ins

    Stilde = Phi * (L2-I2 - L1 + I1) / (Ltot)
    Q1 = 0.5 * (Phi+Stilde)
    Q2 = 0.5 * (Phi-Stilde)
    inc1 = T.arccos(1-Q1/(L1-I1))
    inc2 = T.arccos(1-Q2/(L2-I2))

    orbels = [a1,e1,inc1,k*T.arctan2(y1,x1),a2,e2,inc2,k*T.arctan2(y2,x2),T.arctan2(y_inc,x_inc)]
    orbels_dict = dict(zip(
            ['a1','e1','inc1','theta1','a2','e2','inc2','theta2','phi'],
            orbels
        )
    )

    actions = [L1,L2,I1,I2,Q1,Q2]
    actions_dict = dict(
            zip(
                ['L1','L2','Gamma1','Gamma2','Q1','Q2'],
                actions
                )
            )

    #  Conservative flow
    gradHtot = T.grad(Htot,wrt=dyvars)
    gradHpert = T.grad(Hpert_av,wrt=dyvars)
    gradHkep = T.grad(Hkep,wrt=dyvars)

    hessHtot = theano.gradient.hessian(Htot,wrt=dyvars)
    hessHpert = theano.gradient.hessian(Hpert_av,wrt=dyvars)
    hessHkep = theano.gradient.hessian(Hkep,wrt=dyvars)

    Jtens = T.as_tensor(np.pad(getOmegaMatrix(Ndof),(0,Nconst),'constant'))
    H_flow_vec = Jtens.dot(gradHtot)
    Hpert_flow_vec = Jtens.dot(gradHpert)
    Hkep_flow_vec = Jtens.dot(gradHkep)

    H_flow_jac = Jtens.dot(hessHtot)
    Hpert_flow_jac = Jtens.dot(hessHpert)
    Hkep_flow_jac = Jtens.dot(hessHkep)

    ##########################
    # Compile Theano functions
    ##########################
    func_dict={
     # Hamiltonians
     'H':Htot,
     #'Hpert':Hpert_av,
     #'Hkep':Hkep,
     ## Hamiltonian flows
     'H_flow':H_flow_vec,
     #'Hpert_flow':Hpert_flow_vec,
     #'Hkep_flow':Hkep_flow_vec,
     ## Hamiltonian flow Jacobians
     'H_flow_jac':H_flow_jac,
     #'Hpert_flow_jac':Hpert_flow_jac,
     #'Hkep_flow_jac':Hkep_flow_jac,
     ## Extras
     'orbital_elements':orbels_dict,
     'actions':actions_dict
    }
    compiled_func_dict=dict()
    with tqdm(func_dict.items()) as t:
        for key,val in t:
            t.set_description("Compiling '{}'".format(key))
            if key is 'timescales':
                inputs = extra_ins
            else:
                inputs = ins 
            cf = theano.function(
                inputs=inputs,
                outputs=val,
                givens=givens,
                on_unused_input='ignore'
            )
            compiled_func_dict[key]=cf
    return compiled_func_dict


class SpatialResonanceEquations():

    """
    A class for equations describing the dynamics of a pair of 
    planets in/near a mean motion resonance.

    Attributes
    ----------
    j : int
        Together with k specifies j:j-k resonance
    
    k : int
        Order of resonance.
    
    alpha : float
        Semi-major axis ratio a_1/a_2

    eps : float
        Mass parameter m1*m2 / (mu1+mu2)

    m1 : float
        Inner planet mass

    m2 : float
        Outer planet mass

    """
    def __init__(self,j,k, n_quad_pts = 40, m1 = 1e-5 , m2 = 1e-5):
        self.j = j
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.mstar = 1
        self.n_quad_pts = n_quad_pts
        self._funcs = _get_compiled_theano_functions(n_quad_pts)

    @property
    def extra_args(self):
        return [self.m1,self.m2,self.j,self.k]
    @property
    def mu1(self):
        return self.m1 / (1 + self.m1)
    @property
    def mu2(self):
        return self.m2 / (1 + self.m2)
    @property
    def beta1(self):
        return self.mu1 * np.sqrt(1+self.m1) / (self.mu1 + self.mu2)
    @property
    def beta2(self):
        return self.mu2 * np.sqrt(1+self.m2) / (self.mu1 + self.mu2)
    @property
    def eps(self):
        return self.m1 * self.m2 / (self.mu1 + self.mu2)
    @property
    def alpha(self):
        alpha0 = ((self.j-self.k)/self.j)**(2/3)
        return alpha0 * ((1 + self.m1) / (1+self.m2))**(1/3)
    def H(self,z):
        """
        Calculate the value of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the Hamiltonian evaluated at z.
        """
        return self._funcs['H'](z,*self.extra_args)

    def H_kep(self,z):
        """
        Calculate the value of the Keplerian component of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the Keplerian part of the Hamiltonian evaluated at z.
        """
        return self._funcs['Hkep'](z,*self.extra_args)

    def H_pert(self,z):
        r"""
        Calculate the value of the perturbation component of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the perturbation part of the Hamiltonian evaluated at z.
        """
        return self._funcs['Hpert'](z,*self.extra_args)

    def H_flow(self,z):
        r"""
        Calculate flow induced by the Hamiltonian
        .. math:
            \dot{z} = \Omega \cdot \nablda_{z}H(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow vector
        """
        return self._funcs['H_flow'](z,*self.extra_args)


    def H_flow_jac(self,z):
        r"""
        Calculate the Jacobian of the flow induced by the Hamiltonian
        .. math:
             \Omega \cdot \Delta H(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Jacobian matrix
        """
        return self._funcs['H_flow_jac'](z,*self.extra_args)

    def dyvars_to_orbital_elements(self,z):
        r"""
        Convert dynamical variables
        .. math:
            z = (y_1,y_2,y_inc,x_1,x_2,y_inc,{\cal D})
        to orbital elements
        .. math:
            (a_1,e_1,inc_1,\theta_1,a_2,e_2,inc_2,\theta_2)
        """

        return self._funcs['orbital_elements'](z,*self.extra_args)

    def dyvars_to_Poincare_actions(self,z):
        r"""
        Convert dynamical variables
        .. math:
            z = (y1,y2,yinc,x1,x2,xinc,amd) 
        to canonical Poincare action variables
        .. math:
            (L1,L2,Gamma1,Gamma2,Q1,Q2)
        
        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        dict : 
            Dictionary of canonical actions with strings as keys.
        """

        return self._funcs['actions'](z,*self.extra_args)

    def dyvars_from_rebound_simulation(self,sim,iIn=1,iOut=2,osculating_correction=False,full_output=False):
        r"""
        Convert rebound simulation to dynamical variables.

        Arguments
        ---------
        sim : rebound.Simulation
            Simulation object
        iIn : int, optional 
            Index of inner planet in simulation particles
            list. Default is 1.
        iOut : int, optional 
            Index of outer planet in simulation particles
            list. Default is 2.
        osculating_correction : boole, optional
            If True, correct the orbital elements
            of the rebound simulation by transforming
            to the mean variables of the resonance model.
            Default is False.
        full_output : boole, optional
            Return a dictionary containing the re-scaling 
            factors from time and distance units of the
            input simulation to the time and distance units
            used by SpatialResonanceEquations. 
        Returns
        -------
        z : ndarray
            Dynamical variables for resonance equations of motion.
        scales : dict, optional
            Returned if 'full_output' is True. 
            'scales' contains conversion factors from 
            simulation to resonance equation units such that:
               distance_sim = scales['distance'] * distance_eqs
               time_sim = scales['time'] * time_eqs
        """
        if osculating_correction:
            warnings.warn("Osculating corrections are currently not implemented.")
        align_simulation(sim)
        star = sim.particles[0]
        p1 = sim.particles[iIn]
        p2 = sim.particles[iOut]
        
        self.m1 = p1.m/star.m
        self.m2 = p2.m/star.m
        orbits = get_canonical_heliocentric_orbits(sim)
        o1 = orbits[iIn-1]
        o2 = orbits[iOut-1]
        
        L1 = self.beta1 * np.sqrt(o1.a)
        L2 = self.beta2 * np.sqrt(o2.a)
        Gamma1 = L1 * (1 - np.sqrt(1-o1.e**2))
        Gamma2 = L2 * (1 - np.sqrt(1-o2.e**2))
        Q1 = (L1 - Gamma1) * (1 - np.cos(o1.inc))
        Q2 = (L2 - Gamma2) * (1 - np.cos(o2.inc))
        
        k = self.k
        j = self.j
        Mstar = self.mstar
        m1 = self.m1
        m2 = self.m2
        alpha_res = self.alpha
        s = (j - k) / k
        
        Ltot =  L1 + L2 - Gamma1 - Gamma2 - Q1 - Q2
        factor =  self.beta2 + self.beta1 * np.sqrt(alpha_res)
        action_scale = (factor/Ltot)
        L1 *= action_scale
        L2 *= action_scale
        Gamma1 *= action_scale
        Gamma2 *= action_scale
        Q1 *= action_scale
        Q2 *= action_scale
        Ltot *= action_scale
        dscale = 1 / action_scale**2
        tscale = dscale**(1.5)
        scales = {'action':action_scale,'distance':dscale, 'time':tscale}
        
        I1 = Gamma1
        I2 = Gamma2
        Phi = Q1 + Q2
        Psi = -k * ( s * L2 + (1+s) * L1) 

        s1 = (1 + s) * o2.l - s * o1.l - o1.pomega
        s2 = (1 + s) * o2.l - s * o1.l - o2.pomega
        phi = (1 + s) * o2.l - s * o1.l - 0.5 * (o1.Omega + o2.Omega)
        
        y1 =  np.sqrt(2*I1)*np.sin(s1)
        x1 =  np.sqrt(2*I1)*np.cos(s1)
        y2 =  np.sqrt(2*I2)*np.sin(s2)
        x2 =  np.sqrt(2*I2)*np.cos(s2)
        y_inc =  np.sqrt(2*Phi)*np.sin(phi)
        x_inc =  np.sqrt(2*Phi)*np.cos(phi)

        L1byL2res = self.beta1 * np.sqrt(alpha_res) / self.beta2 
        denom = L1byL2res + s * (1 + L1byL2res)
        amd = (-Psi/k)  * (1+L1byL2res) / denom - Ltot 

        if full_output:
            return np.array([y1,y2,y_inc,x1,x2,x_inc,amd]),scales
        else:
            return np.array([y1,y2,y_inc,x1,x2,x_inc,amd])

    def mean_to_osculating_dyvars(self,psi,z,N = 256):
        r"""
        Apply perturbation theory to transfrom from the phase space coordiantes of the 
        averaged model to osculating phase space coordintates of the full phase space.

        Assumes that the transformation is being applied at the fixed point of the 
        averaged model.

        Arguemnts
        ---------
        Q : float or ndarry
            Value(s) of angle (lambda2-lambda1)/k at which to apply transformation.
            Equilibrium points of the averaged model correspond to orbits that are 
            periodic in Q in the full phase space.
        z : ndarray
            Dynamical variables of the averaged model:
                $\sigma_1,\sigma_2,I_1,I_2,AMD$
        N : int, optional 
            Number of Q points to evaluate functions at when performing Fourier 
            transformation. Default is 
                N=256
        Returns
        -------
        zosc : ndarray, (5,) or (M,5) 
            The osculating values of the dynamical varaibles for the
            input Q values. The dimension of the returned variables
            is set by the dimension of the input 'Q'. If Q is a 
            float, then z is an array of length 5. If Q is an 
            array of length M then zosc is an (M,5) array.
        """
        warnings.warn("Correction from mean to osculating variables is not currently implemented!")
        return z

    def dyvars_to_rebound_simulation(self,z,psi=0,Omega=0,osculating_correction = True,include_dissipation = False,**kwargs):
        r"""
        Convert dynamical variables to a Rebound simulation.

        Arguments
        ---------
        z : ndarray
            Dynamical variables
        Omega : float, optional
            Value of cyclic coordinate Omega   
        psi : float, optional
            Angle variable psi = (lambda1-lambda2) / k. 
            Default is psi=0. Resonant periodic orbits
            are 2pi-periodic in Q.
        include_dissipation : bool, optional
            Include dissipative effects through 
            reboundx's external forces.
            Default is False

        Keyword Arguments
        -----------------
        Mstar : float
            Default=1.0
            Stellar mass.
        inc : float, default = 0.0
            Inclination of planets' orbits.
        period1 : float
            Default = 1.0
            Orbital period of inner planet
        units : tuple
            Default = ('AU','Msun','days')
            Units of rebound simulation.

        Returns
        -------
        tuple :
            Returns a tuple. The first item
            of the tuple is a rebound simulation. 
            The second item is a reboundx.Extras object
            if 'include_dissipation' is True, otherwise
            the second item is None.
        """
        mean_orbels = self.dyvars_to_orbital_elements(z)
        if osculating_correction:
            zosc = self.mean_to_osculating_dyvars(psi,z)
            orbels = self.dyvars_to_orbital_elements(zosc)
        else:
            orbels = mean_orbels
        j,k = self.j, self.k
        s = (j-k) / k
        sim = rb.Simulation()
        Mstar = self.mstar
        mpl1 = self.m1 
        mpl2 = self.m2 
        orbels['lmbda1'] = orbels['phi'] + k * (1 + s) * psi + Omega
        orbels['lmbda2'] = orbels['phi'] + k * s * psi + Omega
        orbels['pomega1'] = orbels['phi'] + Omega - orbels['theta1']/k
        orbels['pomega2'] = orbels['phi'] + Omega - orbels['theta2']/k
        orbels['Omega1'] = np.pi/2 - Omega
        orbels['Omega2'] = -np.pi/2 - Omega
        for i in range(1,3):
            orbels['omega{}'.format(i)] = orbels['pomega{}'.format(i)] - orbels['Omega{}'.format(i)]
        sim.add(m=Mstar)
        for i,mpl in enumerate([mpl1,mpl2]):
            elements = {key:orbels[key + str(i+1)] for key in ['a','e','inc','lmbda','omega','Omega']}
            add_canonical_heliocentric_elements_particle(mpl,elements,sim)
        sim.move_to_com()
        rebx = None
        if include_dissipation:
            pass
        return sim

    def integrate_initial_conditions(self,dyvars0,times):
        """
        Integrate initial conditions and calculate dynamical
        variables and orbital elements at specified times.

        Integration is done with the scipy.integrate.odeint
        method. 

        Arguments
        ---------
        dyvars0 : ndarray
            Initial conditions in the form of an array
            of dynamical variables.
        times : ndarray
            Times at which to calculate output.
        """
        f = lambda y,t: self.H_flow(y)
        Df = lambda y,t: self.H_flow_jac(y)
        soln_dyvars = odeint(
                f,
                dyvars0,
                times,
                Dfun = Df
        )
        el_dict0=self.dyvars_to_orbital_elements(dyvars0)
        els_dict = {key:np.zeros(len(soln_dyvars)) for key in el_dict0};
        els_dict['times'] = times
        for i,sol in enumerate(soln_dyvars):
            els = self.dyvars_to_orbital_elements(sol)
            for key,val in els.items():
                els_dict[key][i]=val
        return {'times':times,'dynamical_variables':soln_dyvars,'orbital_elements':els_dict}
            
