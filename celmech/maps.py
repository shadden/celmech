import numpy as np

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
