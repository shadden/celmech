import numpy as np
from .miscellaneous import sk,Dsk

class StandardMap():
    r"""
    A class representing the Chirikov standard map.
    The map depends on a single parameter, :math:`K` 
    and is defind by

    .. math::
        \begin{align}
            p' &=& p + K  \sin\theta
            \theta' &=& \theta + p'
        \end{align}

    By default, the map is defined on the cylinder with
    the :math:`\theta` coordinate taken mod :math:`2\pi`.
    The parameter `mod_p=True` can be set to take the 
    :math:`p` coordinate modulo :math:`2\pi` as well.

    Parameters
    ----------
    K : float
        Map non-linearity parameter.
    mod_theta : bool, optional
        If True, the :math:`\theta` coordinate
        is taken modulo :math:`2\pi`.
        Default is `True`
    mod_p : bool, optional
        If True, the :math:`p` coordinate
        is taken modulo :math:`2\pi`.
        Default is `False`.
    """
    def __init__(self,K,mod_theta = True,mod_p = False):
        self.K = K 
        self._mod_theta = mod_theta
        self._mod_p = mod_p
        self._set_modfn()

    @property
    def mod_theta(self):
        return self._mod_theta
    @property
    def mod_p(self):
        return self._mod_p
    @mod_theta.setter
    def mod_theta(self,value):
        self._mod_theta = value
        self._set_modfn()
    @property
    def mod_p(self):
        return self._mod_p
    @mod_p.setter
    def mod_p(self,value):
        self._mod_p = value
        self._set_modfn()


    def _set_modfn(self):
        iden = lambda x: x
        mod2pi = lambda x: np.mod(x,2*np.pi)
        
        pfn = mod2pi if self._mod_p else iden
        thetafn = mod2pi if self._mod_theta else iden
        self._modfn = lambda x: np.array([thetafn(x[0]),pfn(x[1])])
    
    def __call__(self,x):
        theta,p = x
        p1 = p + self.K * np.sin(theta)
        theta1 = theta + p1
        x1 = np.array([theta1,p1])
        x1 = self._modfn(x1)
        return x1

    def jac(self,x):
        r"""
        Evaluate the Jacobian map at :math:`x=(\theta,p)`,
        given by 
            
            .. math::
            DT(x) = \begin{pmatrix}
                        1 + K\cos\theta & 1 \\
                        K\cos\theta     & 1
                    \end{pmatrix}
        
        Aruments
        --------
        x : array
            Point(s) at which to evaluate the
            Jacobian.

        Returns
        -------
        DT : array
            Value of the Jacobain at point x.
        """
        theta,p = x
        K = self.K
        Kcos = K*np.cos(theta)
        jac = np.array([
            [1 + Kcos, 1],
            [Kcos,1]
        ])
        return jac

class EncounterMap():
    def __init__(self,m,J,y0, Nmax=7, mod = True):
        self.m = m
        self.J = J
        self._Nmax = Nmax
        self._y0 = y0
        self._update_amplitudes()
        self.mod = mod

    @property
    def mod(self):
        return self._mod

    @mod.setter
    def mod(self,val):
        self._mod = val
        if val:
            self._modfn = lambda x: np.mod(x,2*np.pi)
        else:
            self._modfn = lambda x: x

    @property
    def J(self):
        return self._J

    @property
    def eps(self):
        da = self.da0
        m = self.m
        da4 = da*da*da*da
        return 2 * m /da4 / 3

    @J.setter
    def J(self,value):
        self._J = value
        self.da0 = 2/3/self._J

    @property
    def Nmax(self):
        return self._Nmax
    @property
    def y0(self):
        return self._y0

    @y0.setter
    def y0(self,value):
        self._y0 = value
        self._update_amplitudes()

    @Nmax.setter
    def Nmax(self,value):
        self._Nmax = value
        self._update_amplitudes()

    def _update_amplitudes(self):
        y0 = self._y0
        Nmax = self._Nmax
        self.amps = np.array([-4 * np.pi * k * sk(k,y0) / 3 for k in np.arange(1,Nmax+1)])

    def f(self,theta):
        sin_ktheta = np.array([np.sin(k*theta) for k in range(1,self.Nmax+1)])
        return self.amps @ sin_ktheta

    def dfdtheta_n(self,theta,n):
        trig = np.array([k**(n) * np.sin(k*theta + 0.5 * n * np.pi) for k in range(1,self.Nmax+1)])
        return self.amps @ trig

    def __call__(self,X):
        theta,x = X
        eps = self.eps
        x1 = x + eps * self.f(theta)
        theta1 = theta + 2*np.pi * x1
        theta1 = self._modfn(theta1)
        return np.array([theta1,x1])

    def jac(self,X):
        theta,x = X
        dx1_dx = 1
        dx1_dtheta = self.eps * self.dfdtheta_n(theta,1)
        dtheta1_dx = 2*np.pi * dx1_dx
        dtheta1_dtheta = 1 + 2 * np.pi * dx1_dtheta
        return np.array([[dtheta1_dtheta,dtheta1_dx],[dx1_dtheta,dx1_dx]])

    def inv(self,X):
        theta1,x1 = X
        eps = self.eps
        theta = theta1 - 2 * np.pi * x1
        x = x1 - eps * self.f(theta)
        return (theta,x)

    def partial_derivs(self,x0,Nmax):
        """
        Get the partial derivatives of the map up 
        to order `Nmax` evaluated at point `x0`.
        """
        theta,x = x0
        T = np.zeros((2,Nmax+1,Nmax+1))
        eps = self.eps
        T[:,0,0] = self.__call__(x0)
        T[0][0,1] = 2 * np.pi
        T[1][0,1] = 1
        n=1
        eps_fn = eps * self.dfdtheta_n(theta,n)
        T[0][1,0] = 1 + 2 * np.pi * eps_fn
        T[1][1,0] = eps_fn
        for n in range(2,Nmax+1):
            eps_fn = eps * self.dfdtheta_n(theta,n)
            T[0][n,0] = 2 * np.pi * eps_fn
            T[1][n,0] = eps_fn
        return T

    def inv_partial_derivs(self,x0,Nmax):
        """
        Get the partial derivatives of the map up 
        to order `Nmax` evaluated at point `x0`.
        """
        theta1,x1 = x0
        T = np.zeros((2,Nmax+1,Nmax+1))
        eps = self.eps
        T[:,0,0] = self.inv(x0)
        T[0][1,0] = 1
        T[0][0,1] = -2 * np.pi
        theta,x = T[:,0,0]
        for n in range(1,Nmax+1):
            eps_fn = eps * self.dfdtheta_n(theta,n)
            for l in range(n+1):
                T[1][l,n-l] = -1 * (-2*np.pi)**(n-l) * eps_fn
        T[1][0,1] += 1
        return T



