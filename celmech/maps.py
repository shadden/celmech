import numpy as np
from .miscellaneous import sk,Dsk

class StandardMap():
    r"""
    A class representing the `Chirikov standard map`_.
    The map depends on a single parameter, :math:`K` 
    and is defind by

    .. math::
        \begin{align}
            p' &=& p + K  \sin\theta\\
            \theta' &=& \theta + p'
        \end{align}

    By default, the map is defined on the cylinder with
    the :math:`\theta` coordinate taken mod :math:`2\pi`.
    The parameter `mod_p=True` can be set to take the 
    :math:`p` coordinate modulo :math:`2\pi` as well.

    .. _Chirikov standard map: https://en.wikipedia.org/wiki/Standard_map

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
        r"""
        Is the coordinate :math:`\theta` calculated
        modulo :math:`2\pi`?
        """
        return self._mod_theta
    @property
    def mod_p(self):
        r"""
        Is the coordinate :math:`p` calculated
        modulo :math:`2\pi`?
        """
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

    def inv(self,x):
        r"""
        The inverse mapping

        .. math::
            \begin{align}
            \theta &=& p' - \theta' \\
            p &=& p' - K \sin\theta
            \end{align}


        Arguments
        ---------
        x : array-like
            The point :math:`(\theta',p')`

        Returns
        -------
        array-like
            The point :math:`(\theta,p)`
        """
        theta1,p1 = x
        theta = theta1 - p1
        p = p1 - self.K * np.sin(theta)

    def partial_derivs(self,x,Nmax):
        r"""
        Get the partial derivatives of
        the map evaluated at the point
        `x0` up to order `Nmax`.

        Arguments
        ---------
        x : array-like
            The point at which derivatives
            are to be evaluated
        Nmax : int
            Maximum order of the partial
            derivatives to return

        Returns
        -------
        T : array, shape (2,Nmax+1,Nmax+1)
            The partial derivatives of the map.
            Writing the value of the map at a point
            :math:`(x_1,x_2)` as 
            :math:`T(x_1,x_2) = (T_1(x_1,x_2),T_2(x_1,x_2))`,
            the entry T[i,n,m] stores
            .. math::
                \frac{\partial^{(n+m)}}{\partial x_1^n \partial x_2^m} T_i

            Note that ``T[:,0,0]`` give the value of the map.
        """
        theta,p = x
        c,s = np.cos(theta),np.sin(theta)
        K = self.K
        Ksin_derivs = K*np.array([s,c,-s,-c])
        T = np.zeros((2,Nmax+1,Nmax+1))
        T[:,0,0] = self.__call__(x)
        for n in range(1,Nmax+1):
            T[:,n,0] = Ksin_derivs[n%4]
        T[0,1,0]+=1
        T[0,0,1]+=1
        T[1,0,1]+=1
        return T

    def inv_partial_derivs(self,x,Nmax):
        r"""
        Get the partial derivatives of the inverse map evaluated at the point
        `x0` up to order `Nmax`.

        Arguments
        ---------
        x : array-like
            The point at which derivatives are to be evaluated
        Nmax : int
            Maximum order of the partial derivatives

        Returns
        -------
        T : array, shape (2,Nmax+1,Nmax+1)
            The partial derivatives of the map.  Writing the value of the map
            at a point :math:`(x_1,x_2)` as :math:`T(x_1,x_2) =
            (T_1(x_1,x_2),T_2(x_1,x_2))`, the entry T[i,n,m] stores

            .. math::
                \frac{\partial^{(n+m)}}{\partial x_1^n \partial x_2^m} T_i

            Note that T[:,0,0] give the value of the map.
        """
        theta1,p1 = x
        theta,p = self.inv(x)
        c,s = np.cos(theta),np.sin(theta)
        K = self.K
        Ksin_derivs = K*np.array([s,c,-s,-c])
        T = np.zeros((2,Nmax+1,Nmax+1))
        T[:,0,0] = theta,p
        T[0,1,0] = 1
        T[0,0,1] = -1
        for n in range(1,Nmax+1):
            for l in range(0,n+1):
                T[1,l,n-l] = -(-1)**(n-l) * Ksin_derivs[n%4]
        T[1,0,1]+=1
        return T

    def jac(self,x):
        r"""
        Evaluate the Jacobian map at :math:`x=(\theta,p)`, given by

        .. math::
            DT(x) = \begin{pmatrix}
                        1 + K\cos\theta & 1 \\
                        K\cos\theta     & 1
                    \end{pmatrix}
        
        Aruments
        --------
        x : array
            Point(s) at which to evaluate the Jacobian.

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
    r"""
    A class representing the encounter map.
    The map depends on three parameters,
    :math:`\epsilon,y`, and :math:`J`.
    The map is defined by the equations

    .. math::
        \begin{align}
        x' &= x + \epsilon f(\theta;y)
        \\
        \theta' &= \theta + 2\pi(J-x')
        \end{align}

    By default, the map is defined on the cylinder with
    the :math:`\theta` coordinate taken mod :math:`2\pi`.
    The parameter `mod_p=True` can be set to take the 
    :math:`p` coordinate modulo :math:`2\pi` as well.

    .. _Chirikov standard map: https://en.wikipedia.org/wiki/Standard_map

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
        return 8 * m /da4 / 9

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
        theta1 = theta - 2 * np.pi * x1
        theta1 = self._modfn(theta1)
        return np.array([theta1,x1])

    def jac(self,X):
        theta,x = X
        dx1_dx = 1
        dx1_dtheta = self.eps * self.dfdtheta_n(theta,1)
        dtheta1_dx = -2*np.pi * dx1_dx
        dtheta1_dtheta = 1 - 2 * np.pi * dx1_dtheta
        return np.array([[dtheta1_dtheta,dtheta1_dx],[dx1_dtheta,dx1_dx]])

    def inv(self,X):
        theta1,x1 = X
        eps = self.eps
        theta = theta1 + 2 * np.pi * x1 
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
        T[0][0,1] = -2 * np.pi
        T[1][0,1] = 1
        n=1
        eps_fn = eps * self.dfdtheta_n(theta,n)
        T[0][1,0] = 1 - 2 * np.pi * eps_fn
        T[1][1,0] = eps_fn
        for n in range(2,Nmax+1):
            eps_fn = eps * self.dfdtheta_n(theta,n)
            T[0][n,0] = -2 * np.pi * eps_fn
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
        T[0][0,1] = 2 * np.pi
        theta,x = T[:,0,0]
        for n in range(1,Nmax+1):
            eps_fn = eps * self.dfdtheta_n(theta,n)
            for l in range(n+1):
                T[1][l,n-l] = -1 * (2*np.pi)**(n-l) * eps_fn
        T[1][0,1] += 1
        return T


from sympy import bell
from scipy.special import binom
def _evaluate_chain_rule(n,farr,garr):
    r"""
    Evaluate the :math:`n`th derivative of :math:`f(g(x))` at x=0
    by specifying the Taylor series coefficients of the functions
    f and g.

    Arguments
    ---------
    farr : numpy array
        Taylor series coefficients of function f such that:
        f(u) = f[0] + f[1]*u + f[2]*u^2/2 + ... + f[n]*u^n/n! + ...
    garr : numpy array
        Taylor series coefficients of function g, such that:
        g(x) = g[0] + g[1]*x + g[2]*x^2/2 + ... + g[n]*x^n/n! + ...

    Returns
    -------
    float
        The value of :math:`\frac{d^n}{dx^n}f(g(x))\bigg|_{x=0}`
    """
    #
    return np.sum([farr[k] * bell(n,k,garr[1:n-k+2]) for k in range(1,n)])

def _to_uv_derivs(T,n,m,c,s):
    """
    Helper function for rotate_derivs_array
    """
    tot = np.zeros(2)
    R = np.array([[c,s],[-s,c]])
    for l in range(n+1):
        binom_nl = binom(n,l)
        for l1 in range(m+1):
            cfactor = c**(m+l-l1)
            sfactor = s**(n-l+l1)
            coeff = (-1)**l1 * binom_nl * binom(m,l1) * cfactor * sfactor
            tot +=  coeff * T[:,l+l1,n+m-l-l1]
    return R @ tot

def rotate_derivs_array(T,theta_rot):
    r"""
    Given the partial derivatives of a map, :math:`T(x,y)`, with respect
    to :math:`x` and :math:`y`, get the partial derivative with respect to new,
    rotated coordinates :math:`(u,v)` defined by

    .. math::
        \begin{pvec}
            u \\
            v
        \end{pvec}
        =
        R_\theta \cdot
        \begin{pvec}
            x \\
            y
        \end{pvec}

    where :math:`R_\theta` is a rotation matrix.
    """
    c,s = np.cos(theta_rot),np.sin(theta_rot)
    T1 = np.zeros(T.shape)
    for n in range(1,T.shape[1]):
        for l in range(n+1):
            T1[:,l,n-l] = _to_uv_derivs(T,l,n-l,c,s)
    return T1

### Utilities for calculating Psis ####
from collections import defaultdict
def deriv_of_coeff_and_pows(coeff,pows):
    new_pows = np.append(pows,0)
    results = []
    for m in range(len(pows)):
        if new_pows[m]>0:
            new_pows = np.append(pows,0)
            new_pows[m]-=1
            new_pows[m+1]+=1
            results.append((coeff * pows[m], new_pows))
    return results

def _consolidate_coeff_and_pows_list(cp_list):
    pows_arr = np.array([_[1] for _ in cp_list])
    coeff_arr = np.array([_[0] for _ in cp_list])
    i = 0
    tot = 0
    result = []
    while tot < len(coeff_arr):
        pows = pows_arr[i]
        msk = np.alltrue(pows_arr==pows,axis=1)
        coeff = np.sum(coeff_arr[msk])
        result+=[(coeff,pows)]
        # next index of unique powers
        tot += np.sum(msk)
        i=i+np.argmin(msk[i:])
    return result


def get_Psi_dicts(n):
    PsiOld = {
        (0,1):[(1,np.array([0,1]))],
        (1,0):[(1,np.array([0,0]))]
    }
    Psis = [0,PsiOld]
    for m in range(1,n):
        PsiNew = defaultdict(list)
        for ij,coeffs_and_pows in PsiOld.items():
            i,j = ij
            PsiNew[(i+1,j)] += coeffs_and_pows
            for coeff,pows in coeffs_and_pows:
                PsiNew[(i,j)] += deriv_of_coeff_and_pows(coeff,pows)
                add_one = pows.copy()
                add_one[1] +=1
                PsiNew[(i,j+1)] += [(coeff,add_one)]
        PsiOld = PsiNew.copy()
        for ij, lst in PsiOld.items():
            PsiOld[ij] = _consolidate_coeff_and_pows_list(lst)
        Psis.append(PsiOld)
    # Remove (0,1) entries from Psis
    [Psi.pop((0,1),None) for Psi in Psis[1:]]
    return Psis

def evaluate_Psi(Psi_dict,Tprimes_arr,farr):
    tot = 0
    for ij,coeffs_and_pows_list in Psi_dict.items():
        i,j = ij
        Tprime_ij = Tprimes_arr[i,j]
        sub_tot = 0
        for coeff,pows in coeffs_and_pows_list:
            if coeff==0:
                continue
            npows = pows.shape[0]
            sub_tot += coeff * np.product(farr[:npows]**pows)
        tot+= Tprime_ij * sub_tot
    return tot

def func_from_series(coeffs,x):
    """
    Given a set of Taylor series coefficients, (c_0,....,c_N), evalute
    the sum

    .. math::
        \sum_{n=0}^{N} c_n x^n / n!

    Arguments
    ---------
    coeffs : numpy array
        Values of Taylor series coeffieciens
    x : float
        Argument of function

    Returns
    -------
    float
    """
    return coeffs @ np.array([x**n/np.math.factorial(n) for n in range(coeffs.shape[0])])

def manifold_approx(u,n,farr,garr):
    f = lambda x: func_from_series(farr[:n+1],x)
    g = lambda x: func_from_series(garr[:n+1],x)
    p0 = np.array([u,f(u)])
    p1 = np.array([g(u),f(g(u))])
    return p0,p1

def solve_manifold_f_and_g(xunst,mapobj,Nmax,unstable=True):
    r"""
    Solve for Taylor series approximations of functions
    :math:`f` and :math:`g` satisfying

    .. math::
        T(x_u(s)) = x_* + g(s)\hat{x}_u + f(g(s))\hat{x}_\perp

    where ...

    Arguments
    ---------
    xunst : array-like, (2,)
        Unstable fixed point of map.
    mapobj : object
        A 2D map.
    Nmax : int
        Maximum order Taylor series coefficients to
        compute for f and g
    unstable : bool, optional
        If true, solve for the unstable manifold f and
        g functions, otherwise solve for the stable manifold
        functions.
    Returns
    -------
    R : ndarray, shape (2,2)
        Rotation matrix
    """
    # Array of partial derivatives up to order Nmax
    if unstable:
        T = mapobj.partial_derivs(xunst,Nmax)
    else:
        T = mapobj.inv_partial_derivs(xunst,Nmax)
    # jacobian evaluated at xunst
    jac = np.array([
        [T[0][1,0],T[0][0,1]],
        [T[1][1,0],T[1][0,1]]
    ])
    vals,vecs = np.linalg.eig(jac)
    isreal = np.logical_not(np.iscomplex(vals))
    assert np.alltrue(isreal), "Eigenvalues of map at point ({},{}) are\
    complex!".format(xunst[0],xunst[1])
    iunst = np.argmax(vals)
    # Unstable eigenvalue and direction
    lambdaU = vals[iunst]
    uvec = vecs[:,iunst]
    theta_rot = np.arctan2(uvec[1],uvec[0])
    # Rotation matrix
    s,c = np.sin(theta_rot), np.cos(theta_rot)
    R = np.array([[c,s],[-s,c]])

    Tprime = rotate_derivs_array(T,theta_rot)
    # Initialize iterative procedure to solve for 
    # f and g arrays
    TU_01 = Tprime[0][0,1]
    Tperp_01 = Tprime[1][0,1]
    farr,garr = np.zeros((2,Nmax+1))
    garr[1] = lambdaU
    Psis = get_Psi_dicts(Nmax)
    # Iteratively solve for f and g coeffs
    for n in range(2,Nmax + 1):
        Psi_dict=Psis[n]
        # Get numerical value of Psis
        PsiPerp = evaluate_Psi(Psi_dict,Tprime[1],farr)
        PsiU = evaluate_Psi(Psi_dict,Tprime[0],farr)
        denom = Tperp_01-lambdaU**n
        Bsum = _evaluate_chain_rule(n,farr,garr)
        farr[n] = (Bsum-PsiPerp)/denom
        garr[n] = TU_01 * farr[n] + PsiU
    return R, farr, garr
