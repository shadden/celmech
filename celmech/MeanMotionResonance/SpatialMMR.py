import rebound as rb
import reboundx
import numpy as np
import theano
import theano.tensor as T
from exoplanet.theano_ops.kepler import KeplerOp
from warnings import warn
from scipy.optimize import lsq_linear
from NbodySimulationUtilities import add_canonical_heliocentric_elements_particle, align_simulation, get_canonical_heliocentric_orbits
from scipy.integrate import odeint
import warnings
DEBUG = False

def EulerAnglesTransform(x,y,z,Omega,I,omega):
    
    s1,c1 = T.sin(omega),T.cos(omega)
    x1 = c1 * x - s1 * y
    y1 = s1 * x + c1 * y
    z1 = z
    
    s2,c2 = T.sin(I),T.cos(I)
    x2 = x1
    y2 = c2 * y1 - s2 * z1
    z2 = s2 * y1 + c2 * z1

    s3,c3 = T.sin(Omega),T.cos(Omega)
    x3 = c3 * x2 - s3 * y2
    y3 = s3 * x2 + c3 * y2
    z3 = z2

    return x3,y3,z3

def _get_Omega_matrix(n):
    """
    Get the 2n x 2n skew-symmetric block matrix:
          [0 , I_n]
          [-I_n, 0 ]
    that appears in Hamilton's equations.

    Arguments
    ---------
    n : int
        Determines matrix dimension

    Returns
    -------
    numpy.array
    """
    return np.vstack(
        (
         np.concatenate([np.zeros((n,n)),np.eye(n)]).T,
         np.concatenate([-np.eye(n),np.zeros((n,n))]).T
        )
    )

def _get_compiled_theano_functions(N_QUAD_PTS):
    # Planet masses: m1,m2
    m1,m2 = T.dscalars(2)
    mstar = 1
    mu1  = m1 * mstar / (mstar  + m1) 
    mu2  = m2 * mstar / (mstar  + m2) 
    eta1 = mstar + m1
    eta2 = mstar + m2
    beta1 = mu1 * T.sqrt(eta1/mstar) / (mu1 + mu2)
    beta2 = mu2 * T.sqrt(eta2/mstar) / (mu1 + mu2)
    j,k = T.lscalars('jk')
    s = (j-k) / k

    # Angle variable for averaging over
    psi = T.dvector('psi')

    # Quadrature weights
    quad_weights = T.dvector('w')

    # Dynamical variables:
    Ndof = 3
    Nconst = 1
    dyvars = T.vector()
    s1, s2, phi,  I1, I2, Phi, dRtilde  = [dyvars[i] for i in range(2*Ndof + Nconst)]

    
    a20 = T.constant(1.)
    a10 = ((j-k)/j)**(2/3) * (eta1 / eta2)**(1/3) * a20
    L10 = beta1 * T.sqrt(a10)
    L20 = beta2 * T.sqrt(a20)
    Psi = s * L20 + (1 + s) * L10
    Rtilde = dRtilde - L10 - L20
    ####
    # angles
    ####
    rtilde = T.constant(0.)
    Omega = -1 * rtilde
    l1 = phi + k * (1 + s) * psi + Omega 
    l2 = phi + k * s * psi + Omega 
    gamma1 = s1 - phi - Omega
    gamma2 = s2 - phi - Omega
    q1 = 0.5 * np.pi - Omega
    q2 = -0.5 * np.pi - Omega

    pomega1 = -1 * gamma1
    pomega2 = -1 * gamma2
    Omega1 = -1 * q1
    Omega2 = -1 * q2
    omega1 = pomega1 - Omega1
    omega2 = pomega2 - Omega2

    ###
    # actions
    ###
    Gamma1 = I1 
    Gamma2 = I2 
    L1 = Psi / k - s * (I1 + I2) - s * Phi
    L2 = -1*Psi / k + (1 + s) * (I1 + I2) + (1+s) * Phi
    Cz = -1 * Rtilde

    R = L1+L2-Gamma1-Gamma2-Cz
    G1 = L1 - Gamma1
    G2 = L2 - Gamma2
    
    r2_by_r1 = (L2 - L1 - Gamma2 + Gamma1) / (L1 + L2 - Gamma1 - Gamma2 - R)
    rho1 = 0.5 * R * (1 + r2_by_r1)
    rho2 = 0.5 * R * (1 - r2_by_r1)


    a1 = (L1 / beta1 )**2 
    e1 = T.sqrt(1-(1-(Gamma1 / L1))**2)
    
    a2 = (L2 / beta2 )**2 
    e2 = T.sqrt(1-(1-(Gamma2 / L2))**2)
    
    cos_inc1 = 1 - rho1 / G1 
    cos_inc2 = 1 - rho2 / G2
    inc1 = T.arccos(cos_inc1)
    inc2 = T.arccos(cos_inc2)
    


    Hkep = -0.5 * T.sqrt(eta1) * beta1 / a1 - 0.5 * T.sqrt(eta2) * beta2 / a2

    ko = KeplerOp()
    M1 = l1 - pomega1
    M2 = l2 - pomega2
    sinf1,cosf1 =  ko( M1, e1 + T.zeros_like(M1) )
    sinf2,cosf2 =  ko( M2, e2 + T.zeros_like(M2) )
    # 
    n1 = T.sqrt(eta1 / mstar ) * a1**(-3/2)
    n2 = T.sqrt(eta2 / mstar ) * a2**(-3/2)
    Hint_dir,Hint_ind,r1,r2,v1,v2 = calc_Hint_components_sinf_cosf(
            a1,a2,e1,e2,inc1,inc2,omega1,omega2,Omega1,Omega2,n1,n2,sinf1,cosf1,sinf2,cosf2
    )
    eps = m1*m2/(mu1 + mu2) / T.sqrt(mstar)
    Hpert = (Hint_dir + Hint_ind / mstar)
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
    orbels = [a1,e1,inc1,k*s1,a2,e2,inc2,k*s2,phi,Omega]
    orbels_dict = dict(zip(
            ['a1','e1','inc1','theta1','a2','e2','inc2','theta2','phi'],
            orbels
        )
    )
    actions = [L1,L2,Gamma1,Gamma2,rho1,rho2,Rtilde,Psi]
    actions_dict = dict(
            zip(
                ['L1','L2','Gamma1','Gamma2','Q1','Q2','Rtilde','Psi'],
                actions
                )
            )

    #  Conservative flow
    gradHtot = T.grad(Htot,wrt=dyvars)
    hessHtot = theano.gradient.hessian(Htot,wrt=dyvars)
    Jtens = T.as_tensor(np.pad(_get_Omega_matrix(Ndof),(0,Nconst),'constant'))
    H_flow_vec = Jtens.dot(gradHtot)
    H_flow_jac = Jtens.dot(hessHtot)

    ##########################
    # Compile Theano functions
    ##########################
    orbels_fn = theano.function(
        inputs=ins,
        outputs=orbels_dict,
        givens=givens,
        on_unused_input='ignore'
    )

    actions_fn = theano.function(
        inputs=ins,
        outputs=actions_dict,
        givens=givens,
        on_unused_input='ignore'
    )
    Rtilde_fn = theano.function(
        inputs=ins,
        outputs=Rtilde,
        givens=givens,
        on_unused_input='ignore'
    )

    Htot_fn = theano.function(
        inputs=ins,
        outputs=Htot,
        givens=givens,
        on_unused_input='ignore'
    )

    Hpert_fn = theano.function(
        inputs=ins,
        outputs=Hpert_av,
        givens=givens,
        on_unused_input='ignore'
    )

    H_flow_vec_fn = theano.function(
        inputs=ins,
        outputs=H_flow_vec,
        givens=givens,
        on_unused_input='ignore'
    )
    
    H_flow_jac_fn = theano.function(
        inputs=ins,
        outputs=H_flow_jac,
        givens=givens,
        on_unused_input='ignore'
    )

    return dict({
        'orbital_elements':orbels_fn,
        'actions':actions_fn,
        'Rtilde':Rtilde_fn,
        'Hamiltonian':Htot_fn,
        'Hpert':Hpert_fn,
        'Hpert_components':Hpert_components_fn,
        'Hamiltonian_flow':H_flow_vec_fn,
        'Hamiltonian_flow_jacobian':H_flow_jac_fn
        })
def calc_Hint_components_sinf_cosf(a1,a2,e1,e2,inc1,inc2,omega1,omega2,Omega1,Omega2,n1,n2,sinf1,cosf1,sinf2,cosf2):
    """
    Compute the value of the disturbing function
    .. math::
        \frac{1}{|r-r'|} - ??? v.v'
    from a set of input orbital elements for coplanar planets.

    Arguments
    ---------
    a1 : float
        inner semi-major axis 
    a2 : float
        outer semi-major axis 
    e1 : float
        inner eccentricity
    e2 : float
        outer eccentricity
    I1 : float
        inner inclination
    I2 : float
        outer inclination
    omega1 : float
        inner argument of periapse
    omega2 : float
        outer argument of periapse
    dOmega : float
        difference in long. of nodes, Omega2-Omega1
    n1 : float
        inner mean motion
    n2 : float
        outer mean motion
    sinf1 : float
        sine of inner planet true anomaly
    cosf1 : float
        cosine of inner planet true anomaly
    sinf2 : float
        sine of outer planet true anomaly
    cosf2 : float
        cosine of outer planet true anomaly

    Returns
    -------
    (direct,indirect) : tuple
        Returns a tuple containing the direct and indirect parts
        of the interaction Hamiltonian
    """
    r1 = a1 * (1-e1*e1) /(1 + e1 * cosf1)
    _x1 = r1 * cosf1
    _y1 = r1 * sinf1
    _z1 = 0.
    x1,y1,z1 = EulerAnglesTransform(_x1,_y1,_z1,Omega1,inc1,omega1)

    vel1 = n1 * a1 / T.sqrt(1-e1*e1) 
    _u1 = -1 * vel1 * sinf1
    _v1 = vel1 * (e1 + cosf1)
    _w1 = 0.
    u1,v1,w1 = EulerAnglesTransform(_u1,_v1,_w1,Omega1,inc1,omega1)

    r2 = a2 * (1-e2*e2) /(1 + e2 * cosf2)
    _x2 = r2 * cosf2
    _y2 = r2 * sinf2
    _z2 = 0.
    x2,y2,z2 = EulerAnglesTransform(_x2,_y2,_z2,Omega2,inc2,omega2)
    vel2 = n2 * a2 / T.sqrt(2-e2*e2) 
    _u2 = -1 * vel2 * sinf2
    _v2 = vel2 * (e2 + cosf2)
    _w2 = 0.
    u2,v2,w2 = EulerAnglesTransform(_u2,_v2,_w2,Omega2,inc2,omega2)

    # direct term
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1 
    dr2 = dx*dx + dy*dy + dz*dz
    direct = -1 / T.sqrt(dr2)
    # indirect terms
    indirect = u1*u2 + v1*v2 + w1*w2
    return direct,indirect,[x1,y1,z1],[x2,y2,z2],[u1,v1,w1],[u2,v2,w2]

class ResonanceEquations():

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
        return self.m1 * self.mstar / (self.mstar + self.m1)
    @property
    def mu2(self):
        return self.m2 * self.mstar / (self.mstar + self.m2)
    @property
    def eta1(self):
        return self.m1 + self.mstar
    @property
    def eta2(self):
        return self.m2 + self.mstar
    @property
    def beta1(self):
        return self.mu1 * np.sqrt(self.eta1 / self.mstar) / (self.mu1 + self.mu2)
    @property
    def beta2(self):
        return self.mu2 * np.sqrt(self.eta2 / self.mstar) / (self.mu1 + self.mu2)
    @property
    def eps(self):
        return self.m1 * self.m2 / (self.mu1 + self.mu2)
    @property
    def alpha0(self):
        return ((self.j-self.k)/self.j)**(2/3) * (self.eta1 / self.eta2)**(1/3) 

    def Hpert(self,z):
        """
        Calculate the value of the Q-averaged disturbing function

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of Hpert evaluated at z.
        """
        return self._funcs['Hpert'](z,*self.extra_args)

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
        return self._funcs['Hamiltonian'](z,*self.extra_args)

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
        Htot = self.H(z)
        Hpert = self.Hpert(z)
        return Htot - Hpert

    def H_flow(self,z):
        """
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
        return self._funcs['Hamiltonian_flow'](z,*self.extra_args)


    def H_flow_jac(self,z):
        """
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
        return self._funcs['Hamiltonian_flow_jacobian'](z,*self.extra_args)


    def dyvars_to_orbital_elements(self,z):
        r"""
        Convert dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,phi,I_1,I_2,Phi,dRtilde)
        to orbital elements
        .. math:
            (a_1,e_1,inc_1,\\theta_1,a_2,e_2,inc_2,\\theta_2,\\phi)
        
        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        dict : 
            Dictionary of orbital elements with strings as keys.
        """

        return self._funcs['orbital_elements'](z,*self.extra_args)

    def dyvars_to_Poincare_actions(self,z):
        r"""
        Convert dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,phi,I_1,I_2,Phi,dRtilde)
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

    def simulation_to_dyvars(self,sim,i1=1,i2=2):
        r"""
        Convert rebound simulation to dynamical variables.

        Arguments
        ---------
        sim : rebound.Simulation
            Simulation object
        i1 : int, optional 
            Index of inner planet in simulation particles
            list. Default is 1.
        i2 : int, optional 
            Index of outer planet in simulation particles
            list. Default is 2.

        Returns
        -------
        z : ndarray
            Dynamical variables for resonance equations of motion.
        """
        align_simulation(sim)
        star = sim.particles[0]
        p1 = sim.particles[i1]
        p2 = sim.particles[i2]
        
        self.m1 = p1.m/star.m
        self.m2 = p2.m/star.m
        orbits = get_canonical_heliocentric_orbits(sim)
        o1 = orbits[i1-1]
        o2 = orbits[i2-1]
        
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
        alpha_res = ((j-k)/j)**(2/3) * ((Mstar + m1) / (Mstar+m2))**(1/3)
        s = (j - k) / k
        
        Psi =  s * L2 + (1 + s) * L1
        factor =  s * self.beta2 + (1 + s) * self.beta1 * np.sqrt(alpha_res)
        action_scale = (factor/Psi)
        L1 *= action_scale
        L2 *= action_scale
        Gamma1 *= action_scale
        Gamma2 *= action_scale
        Q1 *= action_scale
        Q2 *= action_scale
        dscale = 1 / action_scale**2
        tscale = dscale**(1.5)
        scales = {'action':action_scale,'distance':dscale, 'time':tscale}
        
        I1 = Gamma1
        I2 = Gamma2
        Phi = L1 - Gamma1 + L2 - Gamma2
        Rtilde = Q1 + Q2 - Phi
        dRtilde = Rtilde + self.beta2 + self.beta1 * np.sqrt(alpha_res)

        s1 = (1 + s) * o2.l - s * o1.l - o1.pomega
        s2 = (1 + s) * o2.l - s * o1.l - o2.pomega
        phi = (1 + s) * o2.l - s * o1.l - 0.5 * (o1.Omega + o2.Omega)

        return np.array([s1,s2,phi,I1,I2,Phi,dRtilde]),scales

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

    from scipy.optimize import lsq_linear

    def find_equilibrium(self,guess,dissipation=False,tolerance=1e-9,max_iter=10):
        r"""
        Use Newton's method to locate an equilibrium solution of 
        the equations of motion. 
    
        By default, an equilibrium of the dissipation-free equations
        is sought. In this case, the $\Delta R$ value of the equilibrium 
        solution will be equal to the value of the initially supplied 
        guess of dynamical variables. In this case, the search identifies 
        the equilibrium values of (simga1,sigma2,psi,I1,I2,Psi) at the 
        fixed value of $\Delta R$.
        
        Dissipative dynamics are currently not implemented.

        Arguments
        ---------
        guess : ndarray
            Initial guess for the dynamical variables at the equilibrium.
        dissipation : bool, optional
            Whether dissipative terms are considered in the equations of
            motion. Default is False.
        tolerance : float, optional
            Tolerance for root-finding such that solution satisfies |f(z)| < tolerance
            Default value is 1E-9.
        max_iter : int, optional
            Maximum number of Newton's method iterations.
            Default is 10.
        include_dissipation : bool, optional
            Include dissipative terms in the equations of motion.
            Default is False
    
        Returns
        -------
        zeq : ndarray
            Equilibrium value of dynamical variables.
    
        Raises
        ------
        RuntimeError : Raises error if maximum number of Newton iterations is exceeded.
        """
        if dissipation:
            warnings.warn(
                    "Dissipative equations not currently implemented.\n"+
                    "Proceeding with search for conservative equilibrium."
                    )
        return self._find_conservative_equilibrium(guess,tolerance,max_iter)

    def _find_dissipative_equilibrium(self,guess,tolerance=1e-9,max_iter=10):
        pass

    def _find_conservative_equilibrium(self,guess,tolerance=1e-9,max_iter=10):
        y = guess
        lb = np.array([-np.pi,-np.pi,-np.pi, -1* y[2], -1 * y[3],-1*y[4]])
        ub = np.array([np.pi,np.pi,np.pi, np.inf,np.inf,np.inf])
        fun = self.H_flow
        jac = self.H_flow_jac
        f = fun(y)[:-1]
        J = jac(y)[:-1,:-1]
        it=0
            # Newton method iteration
        while np.linalg.norm(f)>tolerance and it < max_iter:
            # Note-- using constrained least-squares
            # to avoid setting actions to negative
            # values.
            
            # The lower bounds ensure that  I1, I2, and Phi are positive quantities
            lb[3:] =  -1* y[3:-1]
            dy = lsq_linear(J,-f,bounds=(lb,ub)).x
            y[:-1] = y[:-1] + dy
            f = fun(y)[:-1]
            J = jac(y)[:-1,:-1]
            it+=1
            if it==max_iter:
                raise RuntimeError("Max iterations reached!")
        return y

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
            
