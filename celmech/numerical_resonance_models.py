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

from .numerical_resonance_utils import planar_els2xv,calc_Hint_components_spatial,calc_Hint_components_planar
from .nbody_simulation_utilities import get_canonical_heliocentric_orbits, add_canonical_heliocentric_elements_particle, align_simulation
from .miscellaneous import getOmegaMatrix
# set to fast compile, for testing at least...
######################
####### PLANAR #######
######################
def _get_planar_compiled_Hpert_full():
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

        # Dynamical variables:
        Ndof = 2
        Nconst = 1
        psi = T.scalar()
        dyvars = T.vector()
        y1, y2, x1, x2, amd = [dyvars[i] for i in range(2*Ndof + Nconst)]

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
        L1 = -s*Ltot - Psi/k - s * (I1 + I2)
        L2 = (1+s)*Ltot + Psi/k + (1+s) * (I1 + I2)

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
        
        
        Xre1 = kappa1 / T.sqrt(L1)
        Xim1 = -eta1 / T.sqrt(L1)

        Xre2 = kappa2 / T.sqrt(L2)
        Xim2 = -eta2 / T.sqrt(L2)

        absX1_sq = 2 * I1 / L1
        absX2_sq = 2 * I2 / L2
        X_to_z1 = T.sqrt(1 - absX1_sq / 4 )
        X_to_z2 = T.sqrt(1 - absX2_sq / 4 )

        a1 = (L1 / beta1 )**2 
        k1 = Xre1 * X_to_z1
        h1 = Xim1 * X_to_z1
        e1 = T.sqrt( absX1_sq ) * X_to_z1

        a2 = (L2 / beta2 )**2 
        k2 = Xre2 * X_to_z2
        h2 = Xim2 * X_to_z2

        Hdir,Hind = calc_Hint_components_planar(
                a1,a2,l1,l2,h1,k1,h2,k2,Mstar1/mstar,Mstar2/mstar
        )
        eps = m1*m2/ (mu1 + mu2) / T.sqrt(mstar)
        extra_ins = [m1,m2,j,k]
        ins = [psi] + [dyvars] + extra_ins
        ################################
        Hpert = eps * (Hdir + Hind/mstar)
        gradHpert = T.grad(Hpert,wrt=dyvars)
        ################################
        # Scalars
        Hpert_fn = theano.function(
            inputs=ins,
            outputs=Hpert,
            givens=None,
            on_unused_input='ignore'
        )
        # Gradients
        gradHpert_fn = theano.function(
            inputs=ins,
            outputs=gradHpert,
            givens=None,
            on_unused_input='ignore'
        )
        return Hpert_fn, gradHpert_fn

def _get_planar_compiled_theano_functions(N_QUAD_PTS):
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
        Ndof = 2
        Nconst = 1
        dyvars = T.vector()
        y1, y2, x1, x2, amd = [dyvars[i] for i in range(2*Ndof + Nconst)]

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
        L1 = -s*Ltot - Psi/k - s * (I1 + I2)
        L2 = (1+s)*Ltot + Psi/k + (1+s) * (I1 + I2)

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
        
        
        Xre1 = kappa1 / T.sqrt(L1)
        Xim1 = -eta1 / T.sqrt(L1)

        Xre2 = kappa2 / T.sqrt(L2)
        Xim2 = -eta2 / T.sqrt(L2)

        absX1_sq = 2 * I1 / L1
        absX2_sq = 2 * I2 / L2
        X_to_z1 = T.sqrt(1 - absX1_sq / 4 )
        X_to_z2 = T.sqrt(1 - absX2_sq / 4 )

        a1 = (L1 / beta1 )**2 
        k1 = Xre1 * X_to_z1
        h1 = Xim1 * X_to_z1
        e1 = T.sqrt( absX1_sq ) * X_to_z1

        a2 = (L2 / beta2 )**2 
        k2 = Xre2 * X_to_z2
        h2 = Xim2 * X_to_z2
        e2 = T.sqrt( absX2_sq ) * X_to_z2

        beta1p = T.sqrt(Mstar1) * beta1
        beta2p = T.sqrt(Mstar2) * beta2
        Hkep = -0.5 * beta1p / a1 - 0.5 * beta2p / a2
        
        Hdir,Hind = calc_Hint_components_planar(
                a1,a2,l1,l2,h1,k1,h2,k2,Mstar1,Mstar2
        )
        eps = m1*m2/ (mu1 + mu2) / T.sqrt(mstar)
        Hpert = eps * (Hdir + Hind/mstar)
        Hpert_av = Hpert.dot(quad_weights)
        Htot = Hkep + Hpert_av

        ######################
        # Dissipative dynamics
        ######################
        tau_alpha_0, K1, K2, p = T.dscalars(4)
        y1dot_dis,y2dot_dis,x1dot_dis,x2dot_dis,amddot_dis = T.dscalars(5)
        tau_m_inv = 1/tau_alpha_0
        alpha_res = a10/a20
        # Define timescales
        tau_e1 = tau_alpha_0 / K1
        tau_e2 = tau_alpha_0 / K2
        tau_a1_0_inv = -beta2p * alpha_res * tau_m_inv / (beta1p + alpha_res * beta2p)
        tau_a2_0_inv = beta1p * tau_m_inv / (beta1p + alpha_res * beta2p)
        tau_a1 = 1 / (tau_a1_0_inv + 2 * p * e1*e1 / tau_e1 )
        tau_a2 = 1 / (tau_a2_0_inv + 2 * p * e2*e2 / tau_e2 )
        
        tau_L1 = 2 * tau_a1
        tau_L2 = 2 * tau_a2
        Gamma1=I1
        Gamma2=I2
        tau_Gamma1_inv = 1/tau_L1 + (Gamma1-2*L1) / (Gamma1-L1) / tau_e1 
        tau_Gamma2_inv = 1/tau_L2 + (Gamma2-2*L2) / (Gamma2-L2) / tau_e2 
        # Time derivatives of canonical variables
        x1dot_dis = -0.5 * x1 * tau_Gamma1_inv
        x2dot_dis = -0.5 * x2 * tau_Gamma2_inv
        y1dot_dis = -0.5 * y1 * tau_Gamma1_inv
        y2dot_dis = -0.5 * y2 * tau_Gamma2_inv

        L1dot_dis = -1 * L1/tau_L1
        L2dot_dis = -1 * L2/tau_L2
        Gamma1dot_dis = -1 * I1 * tau_Gamma1_inv 
        Gamma2dot_dis = -1 * I2 * tau_Gamma2_inv 
        Ltot_dot_dis = L1dot_dis + L2dot_dis - Gamma1dot_dis - Gamma2dot_dis
        Psi_dot_dis = -k * (s * L2dot_dis + (1+s) * L1dot_dis) 
        amddot_dis = -1 * Ltot_dot_dis - Psi_dot_dis * (1+f) /(s + f*(1+s)) / k
        
        #####################################################
        # Set parameters for compiling functions with Theano
        #####################################################
        
        # Get numerical quadrature nodes and weight
        nodes,weights = np.polynomial.legendre.leggauss(N_QUAD_PTS)
        
        # Rescale for integration interval from [-1,1] to [-pi,pi]
        nodes = nodes * np.pi
        weights = weights * 0.5
        
        # 'givens' will fix some parameters of Theano functions compiled below
        givens = [(psi,nodes),(quad_weights,weights)]

        # 'ins' will set the inputs of Theano functions compiled below
        #   Note: 'extra_ins' will be passed as values of object attributes
        #   of the 'ResonanceEquations' class defined below
        extra_ins = [m1,m2,j,k,tau_alpha_0,K1,K2,p]
        ins = [dyvars] + extra_ins
        
        # Define flows and jacobians.

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
        
        # Dissipative flow
        dis_flow_vec = T.stack(y1dot_dis,y2dot_dis,x1dot_dis,x2dot_dis,amddot_dis)
        dis_flow_jac = theano.gradient.jacobian(dis_flow_vec,dyvars)

        # Extras
        sigma1 = T.arctan2(y1,x1)
        sigma2 = T.arctan2(y2,x2)
        orbels = [a1,e1,k*sigma1,a2,e2,k*sigma2]
        dis_timescales = [1/tau_a1_0_inv,1/tau_a2_0_inv,tau_e1,tau_e2]

        orbels_dict = dict(zip(
                ['a1','e1','theta1','a2','e2','theta2'],
                orbels
            )
        )

        actions = [L1,L2,I1,I2]
        actions_dict = dict(
            zip(
                ['L1','L2','Gamma1','Gamma2'],
                actions
                )
        )
        
        timescales_dict = dict(zip(
            ['tau_m1','tau_m2','tau_e1','tau_e2'],
            dis_timescales
            )
        )
        ##########################
        # Compile Theano functions
        ##########################
        func_dict={
         # Hamiltonians
         'H':Htot,
         'Hpert':Hpert_av,
         'Hkep':Hkep,
         # Hamiltonian flows
         'H_flow':H_flow_vec,
         'Hpert_flow':Hpert_flow_vec,
         'Hkep_flow':Hkep_flow_vec,
         # Hamiltonian flow Jacobians
         'H_flow_jac':H_flow_jac,
         'Hpert_flow_jac':Hpert_flow_jac,
         'Hkep_flow_jac':Hkep_flow_jac,
         # Dissipative flow and Jacobian
         'dissipative_flow':dis_flow_vec,
         'dissipative_flow_jac':dis_flow_jac,
         # Extras
         'orbital_elements':orbels_dict,
         'actions':actions_dict,
         'timescales':timescales_dict
        }
        compiled_func_dict=dict()
        for key,val in func_dict.items():
            if key == 'timescales':
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
        

class PlanarResonanceEquations():
    r"""
    A class for the model describing the dynamics of a pair of planar planets
    in/near a mean motion resonance.

    Includes the effects of dissipation.

    Attributes
    ----------
    j : int
        Together with k specifies j:j-k resonance
    
    k : int
        Order of resonance.
    
    alpha : float
        Semi-major axis ratio :math:`a_1/a_2`

    eps : float
        Mass parameter m1*mu2 / (mu1+mu2)

    m1 : float
        Inner planet mass

    m2 : float
        Outer planet mass

    tau_alpha : float
        Migration timescale defined by :math:`\tau_\alpha^{-1} =\tau_{m,2}^{-1} - \tau_{m,1}^{-1}`

    K1 : float
        Ratio of inner planet eccentricity damping timescale to migration timescale, tau_alpha.

    K2 : float
        Ratio of outer planet eccentricity damping timescale to migration timescale, tau_alpha.

    p : float
        Sets coupling between eccentricity damping and semi-major axis damping so that

        .. math::

            \frac{d}{dt}a_i = -a_i/\tau_{m,i} - 2pe_i^2/\tau_{e,i} 
        
        A value of p=1 corresponds to eccentricity damping at constant angular momentum.

    timescales : dict
        Dictionary containing the migration and eccentricity damping time-scales of 
        the inner and outer planets based on the damping parameters of the resonance
        model.

    """
    def __init__(self,j,k, n_quad_pts = 40, m1 = 1e-5 , m2 = 1e-5,K1=100, K2=100, tau_alpha = 1e5, p = 1):
        self.j = j
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.K1 = K1
        self.K2 = K2
        self.tau_alpha = tau_alpha
        self.p = p 
        self.n_quad_pts = n_quad_pts
        self._Hpert_full,self._grad_Hpert_full = _get_planar_compiled_Hpert_full()
        self._funcs = _get_planar_compiled_theano_functions(n_quad_pts)

    @property
    def extra_args(self):
        return [self.m1,self.m2,self.j,self.k,self.tau_alpha,self.K1,self.K2,self.p]

    @property
    def extra_args_short(self):
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
    @property
    def timescales(self):
        return self._funcs['timescales'](*self.extra_args)

    
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

    def H_kep_flow(self,z):
        """
        Calculate the flow vector generated by the Keplerian component of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            The flow vector of the Keplerian component of the Hamiltonian
        """
        return self._funcs['Hkep_flow'](z,*self.extra_args)

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

    def H_pert_flow(self,z):
        r"""
        Calculate the flow vector generated by the perturbation component of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            The flow vector generated by the perturbation part of the Hamiltonian evaluated at z.
        """
        return self._funcs['Hpert_flow'](z,*self.extra_args)

    def H_pert_full(self,psi,z):
        r"""
        Calculate the value of the perturbation component of the full, unaveraged
        Hamiltonian.

        Arguments
        ---------
        psi : float
            Angle variable 

            .. math::
                (\lambda_2-\lambda_1)/k  

        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the perturbation part of the Hamiltonian evaluated at :math:`(\psi,z)`.
        """
        return self._Hpert_full(psi,z,*self.extra_args_short)

    def H_pert_osc(self,psi,z):
        r"""
        Calculate the oscillating part of value of the perturbation component of the unaveraged Hamiltonian.

        Arguments
        ---------
        psi : float
            Angle variable 

            .. math::
                (\lambda_2-\lambda_1)/k  

        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the perturbation part of the Hamiltonian evaluated at :math:`(\psi,z)`.
        """
        return self.H_pert_full(psi,z) - self.H_pert(z)

    def grad_H_pert_full(self,psi,z):
        r"""
        Calculate the gradient of the perturbation component of the full, unaveraged
        Hamiltonian with respect to variables :math:`z`.

        Arguments
        ---------
        psi : float
            Angle variable 

            .. math::
                (\lambda_2-\lambda_1)/k  

        z : ndarray
            Dynamical variables

        Returns
        -------
        grad : ndarray
            The gradient the perturbation part of the Hamiltonian with respect to 
            :math:`z` evaluated at :math:`(\psi,z)`.
        """
        return self._grad_Hpert_full(psi,z,*self.extra_args_short)

    def H_flow(self,z):
        r"""
        Calculate flow induced by the Hamiltonian

        .. math::
            \dot{z} = \Omega \cdot \nabla_{z}H(z)

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

        .. math::
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

    def flow(self,z):
        r"""
        Calculate the flow vector of the equations
        of motion

        .. math::
            \dot{z} = \Omega \cdot \nabla_{z}H(z) + f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow vector
        """
        return self._funcs['H_flow'](z,*self.extra_args) + self._funcs['dissipative_flow'](z,*self.extra_args)

    def flow_jac(self,z):
        r"""
        Calculate the Jacobian of the equations of motion
        given by 

        .. math::
             \Omega \cdot \Delta H(z) + \nabla f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Jacobian matrix
        """
        return self._funcs['H_flow_jac'](z,*self.extra_args) + self._funcs['dissipative_flow_jac'](z,*self.extra_args)

    def omega_syn(self,z):
        r"""
        Calculate the synodic frequency,

        .. math::
            \omega_{syn} = n_2 - n_1

        as a function of dynamical variables :math:`z`

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            Synodic frequency
        """
        actions = self.dyvars_to_Poincare_actions(z)
        L1,L2 = actions['L1'],actions['L2']
        n1 = np.sqrt(1 + self.m1) * self.beta1**3 / L1**3
        n2 = np.sqrt(1 + self.m2) * self.beta2**3 / L2**3
        return n2 - n1

    def grad_omega_syn(self,z):
        r"""
        Calculate the gradient of the synodic frequency,

        .. math::
            \nabla_{\pmb{z}} \omega_{syn} 

        as a function of dynamical variables :math:`\pmb{z}`

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float  
            Synodic frequency
        """
        s  = (self.j - self.k) / self.k
        y1,y2,x1,x2,amd = z
        actions = self.dyvars_to_Poincare_actions(z)
        L1,L2 = actions['L1'],actions['L2']
        n1 = np.sqrt(1 + self.m1) * self.beta1**3 / L1**3
        n2 = np.sqrt(1 + self.m2) * self.beta2**3 / L2**3
        dn1_dL1 = -3 * n1 / L1
        dn2_dL2 = -3 * n2 / L2

        dL1dx1 = -1 * s * x1
        dL1dy1 = -1 * s * y1
        dL1dx2 = -1 * s * x2
        dL1dy2 = -1 * s * y2

        dL2dx1 = (1+s) *  x1
        dL2dy1 = (1+s) *  y1
        dL2dx2 = (1+s) *  x2
        dL2dy2 = (1+s) *  y2
        
        return np.array((
            dn2_dL2 *  dL2dy1 - dn1_dL1 * dL1dy1,
            dn2_dL2 *  dL2dy2 - dn1_dL1 * dL1dy2,
            dn2_dL2 *  dL2dx1 - dn1_dL1 * dL1dx1,
            dn2_dL2 *  dL2dx2 - dn1_dL1 * dL1dx2
            ))

    def dyvars_to_orbital_elements(self,z):
        r"""
        Convert dynamical variables

        .. math::
            z = (y_1,y_2,x_1,x_2,{\cal D})

        to orbital elements

        .. math::
            (a_1,e_1,\theta_1,a_2,e_2,\theta_2)
        """

        return self._funcs['orbital_elements'](z,*self.extra_args)

    def orbital_elements_to_dyvars(self,orbels):
        r"""
        Convert orbital elements

        .. math::
            (a_1,e_1,\theta_1,a_2,e_2,\theta_2)

        to dynamical variables

        .. math::
            z = (y_1,y_2,x_1,x_2,{\cal D})
        """
        # Total angular momentum constrained by
        # Ltot = beta1 * sqrt(alpha_res) + beta2 - amd
        Mstar = 1
        m1 = self.m1
        m2 = self.m2
        k = self.k
        j = self.j
        alpha_res = self.alpha
    
        s = (j - k) / k
        
        keys=['a1','e1','theta1','a2','e2','theta2']
        a1,e1,theta1,a2,e2,theta2 = [orbels[key] for key in keys]
        
        L1 = self.beta1 * np.sqrt(a1)
        L2 = self.beta2 * np.sqrt(a2)
        Gamma1 = L1 * (1 - np.sqrt(1-e1**2))
        Gamma2 = L2 * (1 - np.sqrt(1-e2**2))
        I1 = Gamma1
        I2 = Gamma2
        
        Ltot = L1 + L2 - I1 - I2
        Psi = -k * ( s * L2 + (1+s) * L1) 
        rt_a20 = Ltot / (self.beta2 + self.beta1 * np.sqrt(alpha_res)) 

        I1 /= rt_a20
        I2 /= rt_a20
        Psi  /= rt_a20
        Ltot /= rt_a20
        L1byL2res = self.beta1 * np.sqrt(alpha_res) / self.beta2 
        denom = L1byL2res + s * (1 + L1byL2res)
        amd = (-Psi/k)  * (1+L1byL2res) / denom - Ltot 
        sigma1 = theta1 / k
        sigma2 = theta2 / k
        x1 = np.sqrt(2 * I1) * np.cos(sigma1)
        y1 = np.sqrt(2 * I1) * np.sin(sigma1)
        x2 = np.sqrt(2 * I2) * np.cos(sigma2) 
        y2 = np.sqrt(2 * I2) * np.sin(sigma2)

        #dscale = 1 / action_scale**2
        #tscale = dscale**(1.5)
        #scales = {'action':action_scale,'distance':dscale, 'time':tscale}

        return np.array((y1,y2,x1,x2,amd))
    
    def find_equilibrium(self,guess,dissipation=False,tolerance=1e-9,max_iter=10):
        """
        Use Newton's method to locate an equilibrium solution of 
        the equations of motion. 
    
        By default, an equilibrium of the dissipation-free equations
        is sought. In this case, the AMD value of the equilibrium 
        solution will be equal to the value of the initially supplied 
        guess of dynamical variables. If dissipative terms are included
        then the equilibrium will depend on the parameters K1 and K2
        as well as tau_alpha.
    
        Arguments
        ---------
        guess : ndarray
            Initial guess for the dynamical variables at the equilibrium.
        dissipation : bool, optional
            Whether dissipative terms are considered in the equations of
            motion. Default is False.
        tolerance : float, optional
            Tolerance for root-finding such that solution satisfies :math:`|f(z)|` < tolerance
            Default value is 1E-9.
        max_iter : int, optional
            Maximum number of Newton's method iterations.
            Default is 10. If maximum is reached, result will be 
            returned with a warning.
        include_dissipation : bool, optional
            Include dissipative terms in the equations of motion.
            Default is False
    
        Returns
        -------
        zeq : ndarray
            Equilibrium value of dynamical variables.
    
        """
        if dissipation:
            return self._find_dissipative_equilibrium(guess,tolerance,max_iter)
        else:
            return self._find_conservative_equilibrium(guess,tolerance,max_iter)

    def _find_dissipative_equilibrium(self,guess,tolerance=1e-9,max_iter=10):
        y = guess
        fun = self.flow
        jac = self.flow_jac
        f = fun(y)
        J = jac(y)
        it=0
        # Newton method iteration
        while np.linalg.norm(f)>tolerance and it < max_iter:
            dy = lsq_linear(J,-f).x
            y = y + dy
            f = fun(y)
            J = jac(y)
            it+=1
            if it==max_iter:
                warn("Newton's method failed to converge before the maximum of {} iterations were completed. Try re-running with a higher value of 'max_iter' or choose a better initial guess for the equilibrium configuration.".format(max_iter))
        return y
    def _find_conservative_equilibrium(self,guess,tolerance=1e-9,max_iter=10):
        y = guess
        fun = self.H_flow
        jac = self.H_flow_jac
        f = fun(y)[:-1]
        J = jac(y)[:-1,:-1]
        it=0
        # Newton method iteration
        while np.linalg.norm(f)>tolerance and it < max_iter:
            dy = lsq_linear(J,-f).x
            y[:-1] = y[:-1] + dy
            f = fun(y)[:-1]
            J = jac(y)[:-1,:-1]
            it+=1
            if it==max_iter:
                raise RuntimeError("Max iterations reached!")
        return y

    def dyvars_to_rebound_simulation(self,z,Q=0,pomega1=0,osculating_correction = True,include_dissipation = False,**kwargs):
        r"""
        Convert dynamical variables

        .. math::
            z = (y_1,y_2,x_1,x_2,{\cal D}

        to a Rebound simulation.

        Arguments
        ---------
        z : ndarray
            Dynamical variables
        pomega1 : float, optional
            Planet 1's longitude of periapse. 
            Default is 0
        Q : float, optional
            Angle variable Q = (lambda2-lambda1) / k. 
            Default is Q=0. Resonant periodic orbits
            are 2pi-periodic in Q.
        osculating_correction : bool, optional
            If True, apply correction from the 'averaged' variables employed by
            the resonance model to osculating variables used in the un-averaged
            three-body problem. Correction is done to first order in planet masses
            and assumes that the system is near an equilibrium configuration of 
            the resonance model.
            Default is True.
        include_dissipation : bool, optional
            Include dissipative effects through 
            reboundx's external forces.
            Default is False

        Keyword Arguments
        -----------------
        inc : float, default = 0.0
            Inclination of planets' orbits.

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
            zosc = self.mean_to_osculating_dyvars(Q,z)
            orbels = self.dyvars_to_orbital_elements(zosc)
        else:
            orbels = mean_orbels
        j,k = self.j, self.k
        keys=['a1','e1','theta1','a2','e2','theta2']
        a1,e1,theta1,a2,e2,theta2 = [orbels[key] for key in keys]

        syn_angle = self.k * Q
        pomega2 = np.mod( pomega1 -  (theta2 - theta1) / k, 2*np.pi)
        M1 = np.mod( (theta1 - j*syn_angle) / k ,2*np.pi )
        l1 = np.mod(M1 + pomega1,2*np.pi)
        l2 = np.mod( syn_angle + l1,2*np.pi)
        inc = kwargs.pop('inc',0.0)

        sim = rb.Simulation()
        sim.add(m=1)
        for i in range(1,3):
            els={'a':orbels['a{}'.format(i)],'e':orbels['e{}'.format(i)],'inc':inc}
            els['l']=[l1,l2][i-1]
            els['omega']=[pomega1,pomega2][i-1]
            els['Omega']=0.
            m = [self.m1,self.m2][i-1]
            add_canonical_heliocentric_elements_particle(m,els,sim)
        sim.move_to_com()
        
        if include_dissipation:
            ps = sim.particles
            rebx = reboundx.Extras(sim)
            mod = rebx.load_operator("modify_orbits_direct")
            rebx.add_operator(mod)
            mod.params["p"] = self.p
            timescales = self.timescales
            ps[1].params["tau_a"]=-1*timescales['tau_m1'] 
            ps[2].params["tau_a"]=-1*timescales['tau_m2'] 
            ps[1].params["tau_e"]=-1*timescales['tau_e1'] 
            ps[2].params["tau_e"]=-1*timescales['tau_e2'] 
            return sim,rebx
        else:
            return sim
    def dyvars_from_rebound_simulation(self,sim,iIn=1,iOut=2, osculating_correction = False):
        r"""
        Convert dynamical variables

        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}

        to a Rebound simulation.

        Arguments
        ---------
        sim : rebound.Simulation
            Simulation object to 
        iIn : int, optional
            Integer index of inner planet to use
            for computing dynamical variables.
            Default is 1
        iOut : int, optional 
            Integer index of outer planet to use
            for computing dynamical variables.
            Default is 2
        osculating_correction : boole, optional
            If True, correct the orbital elements
            of the rebound simulation by transforming
            to the mean variables of the resonance model.
        Returns
        ---------
        z : ndarray
            Dynamical variables
        """
        els=dict()
        orbits = get_canonical_heliocentric_orbits(sim)
        o1 = orbits[iIn-1]
        o2 = orbits[iOut-1]
        for i,o in enumerate([o1,o2]):
            els['a{}'.format(i+1)] = o.a
            els['e{}'.format(i+1)] = o.e
            els['pomega{}'.format(i+1)] = o.pomega
            els['l{}'.format(i+1)] = o.l
        theta = self.j * els['l2'] - (self.j-self.k) * els['l1']
        for i in range(1,3):
            els['theta{}'.format(i)] = np.mod(theta-self.k * els['pomega{}'.format(i)],2*np.pi)
        if osculating_correction:
            warnings.warn("Osculating corrections are currently not implemented.")
        return self.orbital_elements_to_dyvars(els)

    def dyvars_to_Poincare_actions(self,dyvars):
        """
        Convert a set of dynamical variables to Poincare actions

        Arguments
        ---------
        dyvars : ndarray
          Dynamical variables to convert

        Returns
        -------
        actions : dict
          Poincare actions stored as dictionary entries.
        """
        return self._funcs['actions'](dyvars,*self.extra_args)

    def mean_to_osculating_dyvars(self,Q,z,N = 256):
        r"""
        Apply perturbation theory to transfrom from the phase space coordiantes of the
        averaged model to osculating phase space coordintates of the full phase space.
        Assumes that the transformation is being applied at the fixed point of the
        averaged model. Details are described in the appendix of 
        `Hadden & Payne (2020) <https://ui.adsabs.harvard.edu/abs/2020AJ....160..106H/abstract>`_

        Arguemnts
        ---------
        Q : float or ndarry
            Value(s) of angle :math:`Q=(lambda2-lambda1)/k` at which to apply transformation.
            Equilibrium points of the averaged model correspond to orbits that are
            periodic in Q in the full phase space.
        z : ndarray
            Dynamical variables of the averaged model, math::`z =
            (y_1,y_2,x_1,x_2,{\cal D})`

        N : int, optional
            Number of Q points to evaluate functions at when performing Fourier
            transformation. Default is N=256

        Returns
        -------
        zosc : ndarray, (5,) or (M,5)
            The osculating values of the dynamical varaibles for the
            input Q values. The dimension of the returned variables
            is set by the dimension of the input 'Q'. If Q is a
            float, then z is an array of length 5. If Q is an
            array of length M then zosc is an (M,5) array.
        """
        omega_syn = self.omega_syn(z)
        OmegaMtrx = getOmegaMatrix(2)
        Omega_del2H = self.H_flow_jac(z)[:-1,:-1]
        vals,S = np.linalg.eig(Omega_del2H)
        S = np.transpose([S.T[i] for i in (0,2,1,3)])
        s1 = (S.T @ OmegaMtrx @ S)[0,2]
        s2 = (S.T @ OmegaMtrx @ S)[1,3]
        S.T[0]*=1/np.sqrt(s1)
        S.T[2]*=1/np.sqrt(s1)
        S.T[1]*=1/np.sqrt(s2)
        S.T[3]*=1/np.sqrt(s2)
        Sinv = np.linalg.inv(S)

        Qarr = np.atleast_1d(Q)
        dchi = np.zeros((4,len(Qarr)),dtype=np.complex128)

        # Fill arrays for FFT
        Qs = np.linspace(0,2*np.pi,N)
        X = np.zeros((N,4),dtype=np.complex128)

        gradHkep = -1 * OmegaMtrx @ self.H_kep_flow(z)[:-1]
        domega_syn_dz = self.grad_omega_syn(z)
        domega_syn_dw = S.T @ domega_syn_dz

        for i,q in enumerate(Qs):
            gradHosc = self.grad_H_pert_full(q,z)[:-1] + gradHkep
            X[i] = S.T @ (gradHosc)
            X[i] -= self.H_pert_osc(q,z) * domega_syn_dw/omega_syn

        omegas = +1*np.imag(np.diag(Sinv @ Omega_del2H @ S))[:2]
        for I in range(2):
            A = np.fft.fft(X[:,I])
            freqs = np.fft.fftshift(np.fft.fftfreq(N)*N)
            amps = np.fft.fftshift(A)/N
            for l in range(1,N//2 - 1):
                sig = -1j * self.k * amps[N//2+l] * np.exp(1j * freqs[N//2+l] * Qarr)  / (-l*omega_syn - self.k * omegas[I])
                sig +=-1j * self.k * amps[N//2-l] * np.exp(1j * freqs[N//2-l] * Qarr)  / (l*omega_syn - self.k * omegas[I])
                dchi[I] += sig

        dchi[2] = -1j * np.conjugate(dchi[0])
        dchi[3] = -1j * np.conjugate(dchi[1])

        # Get AMD correction
        j,k = self.j,self.k
        L20 = self.beta2
        L10 = self.beta1 * np.sqrt(self.alpha)
        f = L10/L20
        s = (j-k)/k
        prefactor = (1+f) / (f*(1+s) + s) / k
        dAMD = prefactor * np.array([self.H_pert_osc(q,z) for q in Qarr]) / omega_syn 

        dz = np.transpose(-1 * (S @ OmegaMtrx @ dchi).T)
        result = np.transpose(z + np.vstack((dz,dAMD)).T)
        result = np.real(result) # trim small imaginary parts cause by numerical errors
        if result.shape[1] == 1:
            return result.reshape(-1)
        return result

    def integrate_initial_conditions(self,dyvars0,times,dissipation=False):
        r"""
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

        Returns
        -------
        soln : dict
            Dictionary containing both the dynamical variables
            and the orbital elements of the integrated solution.
            The dictionary keys are
            - 'times': Times at which solutin is computed.
            - 'dynamical_variables': ndarray of dynamical at each of the times in 'times'.
            - 'orbital_elements': dict containing arrays of the various orbital elements.
        """
        if dissipation:
            f = lambda y,t: self.flow(y)
            Df = lambda y,t: self.flow_jac(y)
        else:
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

#######################
####### Spatial #######
#######################
def _get_spatial_compiled_theano_functions(N_QUAD_PTS):
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
    Hpert = eps * (Hdir + Hind/mstar)
    Hpert_av = Hpert.dot(quad_weights)
    Htot = Hkep + Hpert_av

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
            if key == 'timescales':
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
        self._funcs = _get_spatial_compiled_theano_functions(n_quad_pts)

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

        .. math::
            \dot{z} = \Omega \cdot \nabla_{z}H(z)

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
      
        .. math::
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
            Dynamical variables of the averaged model, :math:`z =
            (y_1,y_2,y_I,x_1,x_2,x_I,\mathcal{D})`
        N : int, optional
            Number of Q points to evaluate functions at when performing Fourier
            transformation. Default is N=256

        Returns
        -------
        zosc : ndarray, (7,) or (M,7) 
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
        orbels['l1'] = orbels['phi'] + k * (1 + s) * psi + Omega
        orbels['l2'] = orbels['phi'] + k * s * psi + Omega
        orbels['pomega1'] = orbels['phi'] + Omega - orbels['theta1']/k
        orbels['pomega2'] = orbels['phi'] + Omega - orbels['theta2']/k
        orbels['Omega1'] = np.pi/2 - Omega
        orbels['Omega2'] = -np.pi/2 - Omega
        for i in range(1,3):
            orbels['omega{}'.format(i)] = orbels['pomega{}'.format(i)] - orbels['Omega{}'.format(i)]
        sim.add(m=Mstar)
        for i,mpl in enumerate([mpl1,mpl2]):
            elements = {key:orbels[key + str(i+1)] for key in ['a','e','inc','l','omega','Omega']}
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
