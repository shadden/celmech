import numpy as np
from .miscellaneous import sk,Dsk
from sympy import totient
from scipy.special import erfc

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
    
    def with_variational(self,X,dX):
        r"""
        Apply the map along with the tangent map to a point plus variationals.

        In particular,

        .. math::
                \begin{aligned}
                (\theta', p') &= T(\theta, p) 
                \\
                (\delta \theta', \delta p') &= DT(\theta, p) \cdot (\delta \theta, \delta p)
                \end{aligned}

        where :math:`T` is the usual map and :math:`DT` is the Jacobian of the map.

        Parameters
        ----------
        X : array-like
            The point :math:`X = (\theta, w)`.
        dX : array-like
            The variational vector :math:`(\delta \theta, \delta w)`.

        Returns
        -------
        X' : array-like
            The new point :math:`(\theta', w')`.
        dX' : array-like
            The new variational vector :math:`(\delta \theta', \delta w')`.
        """
        jac = self.jac(X)
        X1 = self.__call__(X)
        dX1 = jac @ dX
        return X1,dX1

    def action(self,pt):
        r"""
        Evaluate The action zero-form,

        .. math::
            \lambda(\theta, w) = 2\pi \left( \frac{w'^2}{2} - \frac{\epsilon}{2\pi} F_\beta(\theta) \right),

        where :math:`w' = w - \epsilon \partial_\theta F_\beta(\theta)`. The action zero-form satisfies
        :math:`T^*(w d\theta) - w d\theta = d\lambda`, where :math:`T^*` is the pullback of the map.

        Parameters
        ----------
        pt : array-like
            The point :math:`(\theta, w)` at which to evaluate the action.

        Returns
        -------
        float
            The value of the action zero-form, :math:`\lambda(\theta, w)`.
        """
        theta,p = pt
        K = self.K
        p1 = p + K * np.sin(theta)
        return 0.5 * p1 * p1 - K * np.cos(theta)

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
        T : array, shape (2, Nmax+1, Nmax+1)
            The partial derivatives of the map.
            Writing the value of the map at a point
            :math:`(x_1, x_2)` as 
            :math:`T(x_1, x_2) = (T_1(x_1, x_2), T_2(x_1, x_2))`,
            the entry T[i, n, m] stores
            :math:`\frac{\partial^{(n+m)}}{\partial x_1^n \partial x_2^m} T_i`

            Note that ``T[:, 0, 0]`` gives the value of the map.
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
    def symmetry_lines(self):
        """
        Return the symmetry lines of the map.

        Returns
        -------
        tuple
            Tuple containing three functions that parameterize the symmetry lines of the map.
        """
        sline1 = lambda x: np.array((0,x))
        sline2 = lambda x: np.array((np.pi,x))
        sline3 = lambda x: np.array((0.5 * x,x))
        return (sline1,sline2,sline3)

class EncounterMap():
    r"""
    A class representing the encounter map. The map depends on three parameters,
    :math:`\epsilon`, :math:`y`, and :math:`J`. The map is defined by the equations

    .. math::
        \begin{align}
        x' &= x + \epsilon f(\theta; y) \\
        \theta' &= \theta + 2\pi(J - x')
        \end{align}

    By default, the map is defined on the cylinder with
    the :math:`\theta` coordinate taken mod :math:`2\pi`.
    The parameter ``mod_p=True`` can be set to take the 
    :math:`p` coordinate modulo :math:`2\pi` as well.

    See also the `Chirikov standard map <https://en.wikipedia.org/wiki/Standard_map>`_.

    Parameters
    ----------
    m : float
        Planet-star mass ratio.
    y : float
        The eccentricity divided by the orbit-crossing eccentricity.
    J : float
        Center the map on the :math:`J`:`J-1` MMR. For integer J, the map
        is centered on a first-order MMR. For rational :math:`J = p/q`, the map is
        centered on a :math:`q`-th order MMR.
    mod_theta : bool, optional
        If True, the :math:`\theta` coordinate is taken modulo :math:`2\pi`.
        Default is True.
    mod_p : bool, optional
        If True, the :math:`p` coordinate is taken modulo :math:`2\pi`.
        Default is False.
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

from scipy.interpolate import pade as pade_approx
def func_from_series(coeffs,x,pade=False):
    """
    Given a set of Taylor series coefficients, (c_0,....,c_N), evalute
    the sum

    .. math::
        \sum_{n=0}^{N} c_n x^n / n!

    Arguments
    ---------
    coeffs : numpy array
        Values of Taylor series coefficients
    x : float
        Argument of function

    Returns
    -------
    float
    """
    if not pade:
        return coeffs @ np.array([x**n/np.math.factorial(n) for n in range(coeffs.shape[0])])
    else:
        p,q = pade_approx(coeffs,len(coeffs)//2)
        return p(x)/q(x)

def manifold_approx(u,n,farr,garr,pade=False):
    f = lambda x: func_from_series(farr[:n+1],x,pade)
    g = lambda x: func_from_series(garr[:n+1],x,pade)
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
    f_arr : ndarray, shame (Nmax,)
        Coefficients of taylor expansion for :math:`f`
    g_arr : ndarray, shame (Nmax,)
        Coefficients of taylor expansion for :math:`g`

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

############################################### 
############# Comet Map Utilities ############# 
############################################### 
from .miscellaneous import levin_method_integrate_adaptive
from .disturbing_function import laplace_b
from warnings import warn

from scipy.special import eval_chebyt,eval_chebyu
def _comet_map_coeff_ck(tau,k,q):
    alpha = 1/q
    op_tausq = 1+tau*tau
    om_tausq = 2-op_tausq
    T = eval_chebyt(k,om_tausq/op_tausq)
    lb = laplace_b(0.5,k,0,alpha/op_tausq)
    if k==1:
        lb -=alpha/op_tausq
    return np.sqrt(2 * q) * lb * T
def _comet_map_coeff_sk(tau,k,q):
    alpha = 1/q
    op_tausq = 1+tau*tau
    om_tausq = 2-op_tausq
    U = eval_chebyu(k-1,om_tausq/op_tausq) * (2*tau/op_tausq)
    lb = laplace_b(0.5,k,0,alpha/op_tausq)
    if k==1:
        lb -=alpha/op_tausq
    return np.sqrt(2 * q) * lb * U
def _comet_map_get_osc_root(k,q,N,phase_offset = 0):
    p=np.zeros(4)
    p[0] = np.sqrt(2*q*q*q) * k / 3
    p[2] = 3 * p[0]
    p[3] = -2 * np.pi * N - phase_offset
    root = np.real(np.roots(p)[-1])
    return root

def _comet_map_get_levin_integration_funcs(k,q):
    sk = _comet_map_coeff_sk
    ck = _comet_map_coeff_ck
    fvec_fn = lambda x: [np.vectorize(sk)(x,k,q),np.vectorize(ck)(x,k,q)]
    g = lambda x: np.sqrt(2*q*q*q) * k * (1 + x*x)
    zerofn = np.vectorize(lambda x: 0)
    Amtrx_fns = [[ zerofn, np.vectorize(g)],[np.vectorize(lambda x: -1*g(x)),zerofn]]
    wvec_fn = lambda x: [np.sin(k*np.sqrt(2*q*q*q)*(x+x*x*x/3)),np.cos(k*np.sqrt(2*q*q*q)*(x+x*x*x/3))]
    return fvec_fn,wvec_fn,Amtrx_fns

def comet_map_ck(k,q,atol=1.49e-8,Nmax=10,**kwargs):
    max_intervals = kwargs.get('max_intervals',10)
    max_quad_pts = kwargs.get('max_quad_pts',128)
    interval_size =kwargs.get(
        'interval_size',
        _comet_map_get_osc_root(k,q,5,np.pi/4)
    )
    fvec_fn,wvec_fn,Amtrx_fns = _comet_map_get_levin_integration_funcs(k,q)
    i=0
    tot = 0
    while i<=max_intervals:
        last = levin_method_integrate_adaptive(
            fvec_fn,wvec_fn,Amtrx_fns,
            i*interval_size,(i+1)*interval_size,
            Nmax=max_quad_pts,
            rtol=0,atol=atol
        )
        tot += last
        i+=1
        if np.abs(last)<atol:
            break
    else:
        warn("Exceeded maximum number of iteractions")
    return 2*tot

def _asymptotic_rfunc(beta : float):
    omega = 1/np.sqrt(0.5 * beta*beta*beta)
    asin_arg = 1 - 27 / 8 / omega
    sin = np.sin(np.arcsin(asin_arg)/3)
    return np.sqrt(omega) * (2/3 - 4 * sin /3)
def _comet_map_lambda_constant(beta : float):
    R = _asymptotic_rfunc(beta)
    R2 = R*R
    R3 = R2*R
    R4 = R3*R
    omega = 1/np.sqrt(0.5 * beta*beta*beta)
    val  = 2 * omega / 3
    val += -R2
    val += R3/3 / np.sqrt(omega)
    val += 0.5 * np.log(R4 * beta / 2)
    return val
def _comet_map_A_constant(beta : float):
    R = _asymptotic_rfunc(beta)
    rt_beta3 = np.sqrt(beta*beta*beta)
    num = 2 * (4 - 2**(3/4)*R*np.sqrt(rt_beta3)) * beta**0.25 * R*R
    denom  = 2 + 2*R*R - 2**(3/4)*R*R*R*np.sqrt(rt_beta3)
    denom *= 4*np.sqrt(2)*R*R + R*R*R*R * rt_beta3 - 2 * np.sqrt(beta) - 2**(9/4) * R*R*R * np.sqrt(rt_beta3)
    denom = np.sqrt(denom)
    return num/denom

def comet_map_ck_asymptotic(k : int, beta : float):
    return _comet_map_A_constant(beta) * np.exp(- k * _comet_map_lambda_constant(beta)) / k

def _comet_map_get_ck_arrays(q : float, rtol: float, atol: float, kmax : int):
    beta = 1/q
    
    lmbda = _comet_map_lambda_constant(beta)
    A = _comet_map_A_constant(beta)
    ck_asym_arr = A * np.exp(-lmbda * np.arange(1,kmax+1)) / np.arange(1,kmax+1)
    ck_arr = np.zeros(kmax)

    k = 0
    rel_err = np.inf
    new_err = np.inf
    while rel_err > rtol:
        k += 1
        # exceeded kmax?
        if k > kmax:
            k-=1
            warn("Failed to meet relative error tolerance {} for k<{} ".format(rtol,kmax))
            break

        ck_arr[k-1] = comet_map_ck(k ,q,atol=atol)
        new_err = np.abs(ck_asym_arr[k-1]/ck_arr[k-1] - 1)

        # error is decreasing?
        if new_err < rel_err:
            rel_err = new_err
        else:
            # remove bad amps
            msg = "Error increased before meeting relative error tolerance {}.".format(rtol)
            msg += "Consider decreasing absolute accuracy parameter, atol={}".format(atol)
            warn(msg)
            k -= 1
            break

    ck_arr = np.array(ck_arr)
    ck_asym_arr = np.array(ck_asym_arr)

    return k, ck_arr[:k], ck_asym_arr[:k], rel_err


class CometMap():
    r"""
    A class representing the comet map.

    The map depends on the pericenter distance to perturber semi-major axis ratio, 
    :math:`q/a_p`, the comet-perturber semi-major axis ratio, :math:`a/a_p`, 
    and the perturber-star mass ratio, :math:`\mu`. The map is defined by the equations:

    .. math::

        \begin{align}
        w' &= w + \epsilon f(\theta; q/a_p) \\
        \theta' &= \theta + 2\pi\left(N + w'\right)
        \end{align}

    By default, the map is defined on the cylinder with the :math:`\theta` coordinate taken modulo :math:`2\pi`.  
    The parameter `mod_p=True` can be set to take the :math:`p` coordinate modulo :math:`2\pi` as well.

    Example Usage
    -------------
    To initialize the map:

    .. code-block:: python

        comet_map = CometMap(m=5e-5, q=4/3, N=20, max_kmax=32, rtol=0.05, atol=1e-8, mod=True)

    Parameters
    ----------
    m : float
        Planet-star mass ratio (:math:`\mu`).
    q : float
        Pericenter distance of the comet, measured in units of the perturber semi-major axis (:math:`q/a_p`).
    N : int
        Center the map on an :math:`N:1` mean-motion resonance (MMR).
    max_kmax : int
        Maximum order of Fourier amplitude to include before resorting to an asymptotic approximation.
    rtol : float
        Relative tolerance for Fourier amplitude calculations before using the asymptotic formula.
    atol : float
        Absolute tolerance for Fourier amplitude calculations before using the asymptotic formula.
    mod : bool, optional
        If :code:`True`, the :math:`\theta` coordinate is taken modulo :math:`2\pi`. Default is :code:`True`.

    """
    def __init__(self,m,N,q, max_kmax=32, rtol = 0.05, atol =1.49e-8, mod=True):
        self.m = m
        self.N = N
        assert type(N)==int, "Only integer N allowed"
        self._max_kmax = max_kmax
        self._rtol = rtol
        self.atol = atol
        self._q = q
        self._update_amplitudes()
        self.mod = mod
    def __repr__(self):
        return '<{0}.{1} object at {2}, m={3}, q={4}, N={5}, kmax={6}>'.format(
            self.__module__, #0
            type(self).__name__, #1
            hex(id(self)), #2
            self.m, #3
            self.q, #4
            self.N, #5
            self.kmax #6
            )
    def status(self):
        """
        Print a summary of the current status of the map.

        Returns
        -------
        None
        """
        s = ""
        s+= "---------------------------------\n"
        s+= "celmech CometMap object\n"
        s+= "Pericenter distance:\t{}\n".format(self.q)
        s+= "Planet mass:\t{}\n".format(self.m)
        s+= "N:1 Resonance:\t{}:{}\n".format(self.N,self.N-1)
        s+= "Epsilon parameter:\t{}\n".format(self.eps)
        s+= "mod 1:\t{}\n".format(self.mod)
        print(s)
    @property 
    def q(self):
        return self._q
    @q.setter
    def q(self,val):
        self._q = val
        self._update_amplitudes()

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
    def kmax(self):
        return self._kmax
    
    @property
    def max_kmax(self):
        return self._max_kmax
    
    @max_kmax.setter
    def max_kmax(self,val : int):
        self._max_kmax = val
        if self._rtol_actual > self.rtol:
            self._update_amplitudes()

    @property 
    def rtol(self):
        return self._rtol

    @rtol.setter
    def rtol(self,val):
        self._rtol = val
        if val  < self._rtol_actual:
            if self.kmax < self.max_kmax:
                self._update_amplitudes()
            else:
                warn("Cannot meet target tolerance unless max_kmax={0} is increased.".format(self.max_kmax))
    @property
    def a0(self):
        N = self.N
        return N**(2/3)
    
    @property
    def x0(self):
        return 1/self.a0
    @property
    def eps(self):
        a0 = self.a0
        m = self.m
        return 3*a0**(2.5) * m

    def _update_amplitudes(self):
        q = self._q
        beta = 1/q
        if q<=9/8:
            warn("Asymptotic formulas only valid for q>9/8. Corrections will be ignored.")
            self.lambda_const = np.inf
            self.A_const = 0
        else:
            self.lambda_const = _comet_map_lambda_constant(beta)
            self.A_const =_comet_map_A_constant(beta)

        self.cosh_lambda = np.cosh(self.lambda_const)
        max_kmax = self.max_kmax
        kmax, ck, ck_asym, rtol = _comet_map_get_ck_arrays(q,self.rtol,self.atol,max_kmax)
        self._rtol_actual = rtol
        self._kmax = kmax
        self.ck = ck
        self.delta_ck = ck - ck_asym
        self.amps = np.arange(1,kmax+1) * self.ck
        self.delta_amps = np.arange(1,kmax+1) * self.delta_ck
    def f_asym(self,theta):
        r"""
        The asymptotic kick function,

        .. math::
            -\frac{A(\beta)}{2}\frac{\sin\theta}{\cos\theta - \cosh \lambda(\beta)}


        Parameters
        ----------
        theta : float
            Angle parameter

        Returns
        -------
        float
            value of the kick function
        """
        return -0.5 * self.A_const * np.sin(theta) / (np.cos(theta) - self.cosh_lambda)
    def delta_f(self,theta):
        """
        Difference between the kick fucntion and its asymptotic value.
        """
        sin_ktheta = np.array([np.sin(k*theta) for k in range(1,self.kmax+1)])
        return self.delta_amps @ sin_ktheta
    
    def f(self,theta):
        r"""
        The kick function of the map, :math:`-\partial_\theta F_\beta(\theta)`.

        Parameters
        ----------
        theta : float
            Angle varaible at which to evaluate the kick function

        Returns
        -------
        float
            Value of the kick function
        """
        return self.f_asym(theta) + self.delta_f(theta)
    
    def F(self, theta):
        r"""
        The 'potential function', :math:`F_\beta(\theta)` from which the map's kick function is derived.

        Parameters
        ----------
        theta : float
            Angle varaible at which to evaluate the potential function

        Returns
        -------
        float
            Value of the potential function
        """
        return self.F_asym(theta) + self.delta_F(theta)
    
    def F_asym(self,theta):
        r"""
        The asymptotic kick potential,

        .. math::
            -\frac{1}{2} A \ln (2 (\cosh (\lambda )-\cos (\theta )))


        Parameters
        ----------
        theta : float
            Angle parameter

        Returns
        -------
        float
            value of the kick potential
        """        
        A = self.A_const
        cosh_lambda = self.cosh_lambda
        return -0.5 * A * np.log(2 *(cosh_lambda - np.cos(theta)))
    
    def delta_F(self,theta):
        """
        Difference between the kick potential and its asymptotic value.
        """
        cos_ktheta = np.array([np.cos(k*theta) for k in range(1,self.kmax+1)])
        return self.delta_ck @ cos_ktheta

    def dfdtheta_n(self,theta,n):
        trig = np.array([k**(n) * np.sin(k*theta + 0.5 * n * np.pi) for k in range(1,self.kmax+1)])
        return self.amps @ trig
    
    def __call__(self,X):
        theta,w = X
        eps = self.eps
        w1 = w + eps * self.f(theta)
        theta1 = theta + 2 * np.pi * w1
        theta1 = self._modfn(theta1)
        return np.array([theta1,w1])
    
    def action(self,pt):
        r"""
        Evaluate the action zero-form,

        .. math::
            \lambda(\theta, w) = 2\pi \left( \frac{w'^2}{2} - \frac{\epsilon}{2\pi} F_\beta(\theta) \right),

        where :math:`w' = w - \epsilon \partial_\theta F_\beta(\theta)`. The action zero-form satisfies
        :math:`T^*(w d\theta) - w d\theta = d\lambda`, where :math:`T^*` is the pullback of the map.

        Parameters
        ----------
        pt : array-like
            The point :math:`(\theta, w)` at which to evaluate the action.

        Returns
        -------
        float
            The value of the action zero-form, :math:`\lambda(\theta, w)`.
        """
        theta,w = pt
        eps = self.eps
        w1 = w + eps * self.f(theta)
        Fval = self.F(theta)
        return np.pi * w1 * w1 - eps * Fval
    
    def with_variational(self,X,dX):
        r"""
        Apply the map along with the tangent map to point plus variationals.
        In particular, 

        .. math::
            \begin{aligned}
            (\theta', w') &= T(\theta, w) 
            \\
            (\delta \theta',\delta  w') &= DT(\theta, w) \cdot (\delta \theta,\delta  w) 
            \end{aligned}
        
        where :math:`T` is the usual map and :math:`DT` is the Jacobian of the map.

        Parameters
        ----------
        X : array-like
            The point :math:`X = (\theta,w)`
        dX : array-like
            The variational vector :math:`(\delta\theta,\delta w)`
        
        Returns
        -------
        X' : array-like
            The new point
        dX' : array-lke
            The new variationl vector
        """
        jac = self.jac(X)
        X1 = self.__call__(X)
        dX1 = jac @ dX
        return X1,dX1

    def full_map(self,pt):
        r"""
        Use version of the map, defined as:

        .. math::
                \begin{aligned}
                x' &= x - 2\mu \frac{\partial F_\beta(\theta)}{\partial \theta} 
                \\
                \theta' &= \theta + \frac{2\pi}{x'^{3/2}}~.
                \end{aligned}

        where :math:`x` represents the test particle's inverse semi-major axis.

        Parameters
        ----------
        pt : array-like
            The point :math:`(\theta, x)` to map.

        Returns
        -------
        pt1 : array
            Resulting point :math:`(\theta', x')`.
        """
        theta,x = pt
        x1 = x - 2 * self.m * self.f(theta)
        theta1 = theta + 2 * np.pi / x1**(1.5)
        theta1 = self._modfn(theta1)
        return np.array([theta1,x1])
    
    def jac(self,X):
        theta,x = X
        dx1_dx = 1
        dx1_dtheta = self.eps * self.dfdtheta_n(theta,1)
        dtheta1_dx =2*np.pi * dx1_dx
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
        T[0][0,1] = +2 * np.pi
        T[1][0,1] = 1
        n=1
        eps_fn = eps * self.dfdtheta_n(theta,n)
        T[0][1,0] = 1 + 2 * np.pi * eps_fn
        T[1][1,0] = eps_fn
        for n in range(2,Nmax+1):
            eps_fn = eps * self.dfdtheta_n(theta,n)
            T[0][n,0] = +2 * np.pi * eps_fn
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


    def get_eps_crit(self,tau=1,kmax=None):
        r"""
        Calculate the critical :math:`\epsilon` parameter at which the onset of chaos is predicted based on the resonant optical depth.

        Arguments
        ---------
        tau : float, optional
            Sets the resonant optical depth for the onset of chaos.
            The default value is 1.
        
        kmax : int, optional
            The maximum k value of terms to include in the optical depth
            calculation. By default, the critical :math:`\epsilon` is 
            calculated by including all k values, using the asympototic
            form of resonance widths for large k to estimate a correction.

        Returns
        -------
        float : 
            Critical value of epsilon.
        """
        if kmax is None:
            kmax = self.kmax
            add_remainder = True
        else:
            add_remainder = False

        lmbda = self.lambda_const
        A = self.A_const
        F0 = self.F(0)
        Fpi = self.F(np.pi)
        # full width of first-order MMR
        tot = 2 * np.sqrt((F0 - Fpi)/np.pi)
        for k_minus_1,ck in enumerate(self.ck):
            k = k_minus_1+1
            if k>1 and k<=kmax:
                half_width = np.sqrt(2 * ck / np.pi)
                tot += 2*totient(k)*half_width
        if add_remainder:
            remainder_approx  = 2*np.sqrt(kmax+0.5) * np.exp(-0.5 * (kmax+0.5) * lmbda) / lmbda  
            remainder_approx += np.sqrt(2*np.pi)*erfc(np.sqrt(0.5 * (kmax+0.5) * lmbda)) / lmbda**1.5
            remainder_approx  *=  12 * np.sqrt(2 * A / np.pi**5 )
        else:
            remainder_approx = 0
        for k in np.arange(self.kmax+1,kmax+1):
            ck = A * np.exp(-lmbda * k) / k
            half_width = np.sqrt(2 * ck / np.pi)
            tot = 2 * totient(k) * half_width
        tot = tot+remainder_approx
        return tau*tau/tot/tot
    
    def D_QL(self):
        r"""
        Compute the quasi-linear estimate for the local
        diffusion coefficient given by 

        .. math::
            D_\mathrm{QL} = \frac{1}{2}\epsilon^2\sum_{k}k^2C_{k}(\beta)^2

        Returns
        -------
        D_QL : float
        """
        eps = self.eps
        amps = self.amps
        amps_sq = amps @ amps
        return 0.5 * eps * eps * amps_sq
    def symmetry_lines(self):
        """
        Return the symmetry lines of the map.

        Returns
        -------
        tuple
            Tuple containing three functions that parameterize the symmetry lines of the map.
        """
        sline1 = lambda x: np.array((0,x))
        sline2 = lambda x: np.array((np.pi,x))
        sline3 = lambda x: np.array((np.pi * x,x))
        return (sline1,sline2,sline3)

class NamouniMap():
    r"""
    A class representing the Namouni map. The map depends on parameters,
    :math:`K`, :math:`\phi`, :math:`\lambda`, and :math:`I`. The map is defined by the equations

    .. math::
        \begin{align}
        I_{n+1} &= I_n + W_\phi(I_{n+1},K_{n+1},\lambda_n,\phi_n) \\
        \phi_{n+1} &= \phi_n + W_\I(I_{n+1},K_{n+1},\lambda_n,\phi_n) \\
        K_{n+1} = I_{n+1}-I_n+K_n \\
        \lambda_{n+1} = \lambda_n + \frac{4\pi}{3\epsilon\sqrt{8K_{n+1}/3}}
        \end{align}


    See also Namouni et al. (1996)

    Parameters
    ----------
    m : float
        Planet-star mass ratio.
    e0 : float
        Initial eccentricity of the test particle
    a_r0 : float
        Initial relative semimajor axis in Hill's unit
    lambda0 : float
        Initial conjunction longitude
    phi0 : float
        Initial pericenter longitude
    ind : int
        An integer that indicates the configuration of the planetary system. 
        If ind == 1, the massive planet comes before the test particle.
        Otherwise, the test particle comes first.
    exact_period : bool, optional
        If True, the update for lambda_{n+1} will be calculated as in Duncan et al. (1989), 
        which is the exact calculation.
        Otherwise, the update will follow the original Namouni mapping which is the approximation in Hill's limit.
        Default is True.
    """

    
    def __init__(self,m,ind,order,exact_period=True):
        self._m = m
        self._ind = ind
        # self._Nmax = Nmax
        # self._y0 = y0
        # self._update_amplitudes()
        # self.mod = mod
        self._order = order
        self.exact_period = exact_period

    @property
    def m(self):
        return self._m

    @property
    def ind(self):
        return self._ind

    @property
    def order(self):
        return self._order
    
    @property
    def epsilon(self):
        m = self._m
        return (m/3)**(1/3) 

    @property
    def exact_period(self):
        return self._exact_period
        
    @exact_period.setter
    def exact_period(self,val):
        self._exact_period = val
        if val:
            self._periodfn = lambda Knp1: 2*np.pi*np.abs(1/((1+np.sqrt(8*Knp1/3)*epsilon)**(-3/2)-1))
        else:
            self._periodfn = lambda Knp1: 4*np.pi/(3*epsilon*np.sqrt(8*Knp1/3))

    def namouni_fo_step(self, In, Kn, phin, lambdan, arn, ind):
        """First-order Namouni mapping step."""
        
        W101 = 3.359350011
        
        hn = np.sqrt(2*In)*np.cos(phin)
        kn = np.sqrt(2*In)*np.sin(phin)
        if ind == 1:
            hprime = hn + 2*W101*np.sin(lambdan)/arn**2 # + for Namouni, - for pomega
            kprime = kn - 2*W101*np.cos(lambdan)/arn**2 # - for Namouni, + for pomega
        else: #from DQT, the sign switches for the outer pair
            hprime = hn - 2*W101*np.sin(lambdan)/arn**2 # + for Namouni, - for pomega
            kprime = kn + 2*W101*np.cos(lambdan)/arn**2 # - for Namouni, + for pomega
        phiprime = np.arctan2(kprime, hprime) # 4 quadrant arctan
        #print(hn, kn)
        # both options should give the same result:
        Inprime = 1/2*(hprime**2 + kprime**2)
        #Inprime = In + 2*W1/arn**2*np.sqrt(2*In)*np.sin(thetan) + 2*W1**2/arn**4
        #Namouni update
        hnp1 = hprime
        knp1 = kprime
        Inp1 = Inprime
        Knp1 = Kn + Inp1 - In
        phinp1 = phiprime

        lambdanp1 = lambdan + self._periodfn(Knp1)
        # if exact_period==True:
        #     lambdanp1 = lambdan + 2*np.pi*np.abs(1/((1+np.sqrt(8*Knp1/3)*epsilon)**(-3/2)-1)) #same lambda
        # else:
        #     lambdanp1 = lambdan + 4*np.pi/(3*epsilon*np.sqrt(8*Knp1/3))
        return Inp1, Knp1, phinp1, lambdanp1, hnp1, knp1


    def namouni_so_step(self, In, Kn, phin, lambdan, arn, ind):
        """Second-order Namouni mapping step."""
        """The second order mapping involves an intermediate step and implicit scheme for eccentricity update."""
        """This calculates the implicit update iteratively instead of explicitly solving for the next value."""
        W200= -1
        W101= 3.359350011
        W202= -2.678665722        
 
        hn = np.sqrt(2*In)*np.cos(phin)
        kn = np.sqrt(2*In)*np.sin(phin)
        if ind == 1:
            hprime = hn + 2*W101*np.sin(lambdan)/arn**2 # + for Namouni, - for pomega
            kprime = kn - 2*W101*np.cos(lambdan)/arn**2 # - for Namouni, + for pomega
        else: #from DQT, the sign switches for the outer pair
            hprime = hn - 2*W101*np.sin(lambdan)/arn**2 # + for Namouni, - for pomega
            kprime = kn + 2*W101*np.cos(lambdan)/arn**2 # - for Namouni, + for pomega
        phiprime = np.arctan2(kprime, hprime) # 4 quadrant arctan
        #thetaprime = (pomegaprime - lambdan)
        thetaprime = (phiprime - lambdan)
    
        # A = calc_coeff(2, W2, arn, thetaprime)
        # E1 = 1/(1+A)
        # Inp1 = E1*Inprime
        #implicit I update
        Iprime = 0.5*(hprime**2 + kprime**2)
        Inp1_prev = Iprime #first guess = Iprime
        while True: #iteratively update using implicit scheme
            Inp1_next = Iprime - 4*W202*np.sin(2*thetaprime)*2*Inp1_prev/arn**3
            if np.abs((Inp1_next-Inp1_prev)/Inp1_next) < 1e-12: #break when satisfied
                break
            Inp1_prev = Inp1_next
        # Inp1 = (0.5*(hprime**2 + kprime**2)) / (1 + 4*W2*np.sin(2*thetaprime)*2/arn**3)
        Inp1 = Inp1_next
    
        #pomeganp1 = pomegaprime - (W200 + 2*W2*np.cos(2*thetaprime))*2/arn**3
        phinp1 = phiprime - (W200 + 2*W202*np.cos(2*thetaprime))*2/arn**3
        Knp1 = Kn + Inp1 - In
        hnp1 = np.sqrt(2*Inp1)*np.cos(phinp1)
        knp1 = np.sqrt(2*Inp1)*np.sin(phinp1)
        lambdanp1 = lambdan + self._periodfn(Knp1)
        # if exact_period==True:
        #     lambdanp1 = lambdan + 2*np.pi*np.abs(1/((1+np.sqrt(8*Knp1/3)*epsilon)**(-3/2)-1)) #same lambda
        # else:
        #     lambdanp1 = lambdan + 4*np.pi/(3*epsilon*np.sqrt(8*Knp1/3))
        return Inp1, Knp1, phinp1, lambdanp1, hnp1, knp1



    def namouni_4o_step(self, In, Kn, phin, lambdan, arn, ind):
        """Fourth-order Namouni mapping step."""
        """The fourth order mapping involves an intermediate step and implicit scheme for eccentricity update."""
        """This calculates the implicit update iteratively instead of explicitly solving for the next value."""
        W200, W400= -1,-3/8
        W101, W301 = 3.359350011, 0.1430675142
        W202, W402 = -2.678665722, 0.5232506456
        W303 = 2.639498803
        W404 = -2.855951527
        hn = np.sqrt(2*In)*np.cos(phin)
        kn = np.sqrt(2*In)*np.sin(phin)
        if ind == 1:
            hprime = hn + 2*W101*np.sin(lambdan)/arn**2 # + for Namouni, - for pomega
            kprime = kn - 2*W101*np.cos(lambdan)/arn**2 # - for Namouni, + for pomega
        else: #from DQT, the sign switches for the outer pair
            hprime = hn - 2*W101*np.sin(lambdan)/arn**2 # + for Namouni, - for pomega
            kprime = kn + 2*W101*np.cos(lambdan)/arn**2 # - for Namouni, + for pomega
        phiprime = np.arctan2(kprime, hprime) # 4 quadrant arctan
        #thetaprime = (pomegaprime - lambdan)
        thetaprime = (phiprime - lambdan)
    
        # A = calc_coeff(2, W2, arn, thetaprime)
        # E1 = 1/(1+A)
        # Inp1 = E1*Inprime
        #implicit I update
        Iprime = 0.5*(hprime**2 + kprime**2)
        Inp1_prev = Iprime #first guess = Iprime
        phinp1_prev = phiprime
        while True: #iteratively update using implicit scheme
            Inp1o2 = 4*W202*np.sin(2*thetaprime)*2*Inp1_prev/arn**3
            Inp1o3 = (2*W301*np.sin(thetaprime)+6*W303*np.sin(3*thetaprime))*(2*Inp1_prev)**(3/2)/arn**4
            Inp1o4 = (4*W402*np.sin(2*thetaprime)+8*W404*np.sin(4*thetaprime))*(2*Inp1_prev)**(2)/arn**5
            Inp1_next = Iprime - Inp1o2 - Inp1o3 - Inp1o4
            phinp1o2 = (W200 + 2*W202*np.cos(2*thetaprime))*2/arn**3
            phinp1o3 = 3*2**(3/2)*(W301*np.cos(thetaprime)+W303*np.cos(3*thetaprime))*Inp1_prev**0.5/arn**4
            phinp1o4 = 8*(W400+2*W402*np.cos(2*thetaprime)+2*W404*np.cos(4*thetaprime))*Inp1_prev/arn**5
            phinp1_next = phiprime - (phinp1o2+phinp1o3+phinp1o4)
            if np.abs((Inp1_next-Inp1_prev)/Inp1_next) < 1e-12 and np.abs((phinp1_next-phinp1_prev)/phinp1_next) < 1e-12: 
                #break when satisfied
                break
            Inp1_prev = Inp1_next
            phinp1_prev = phinp1_next

        Inp1 = Inp1_next
        phinp1 = phinp1_next
        Knp1 = Kn + Inp1 - In
        hnp1 = np.sqrt(2*Inp1)*np.cos(phinp1)
        knp1 = np.sqrt(2*Inp1)*np.sin(phinp1)
        lambdanp1 = lambdan + self._periodfn(Knp1)
        # if exact_period==True:
        #     lambdanp1 = lambdan + 2*np.pi*np.abs(1/((1+np.sqrt(8*Knp1/3)*epsilon)**(-3/2)-1)) #same lambda
        # else:
        #     lambdanp1 = lambdan + 4*np.pi/(3*epsilon*np.sqrt(8*Knp1/3))
        return Inp1, Knp1, phinp1, lambdanp1, hnp1, knp1

    def namouni_10o_step(self, In, Kn, phin, lambdan, arn, ind):
        """Tenth-order Namouni mapping step."""
        """The tenth order mapping involves an intermediate step and implicit scheme for eccentricity update."""
        """This calculates the implicit update iteratively instead of explicitly solving for the next value."""
        #define all coeffs in here so that we won't get confused
        #format WABC = W^{A,B}_C
        W200, W400, W600, W800, W1000 = -1,-3/8,-5/24,-35/256,-63/640
        W101, W301, W501, W701, W901 = 3.359350011, 0.1430675142, 0.2652528982, 0.1598760057, 0.1125398541
        W202, W402, W602, W802, W1002 = -2.678665722, 0.5232506456, -0.2880525893, -0.1146109541, -0.09264257446
        W303, W503, W703, W903 = 2.639498803, -1.270297850, 0.5079066956, 0.04027194541
        W404, W604, W804, W1004 = -2.855951527, 2.244981532, -1.030263325, 0.1370220235
        W505, W705, W905 = 3.259400253, -3.586274249, 2.034584646
        W606, W806, W1006 = -3.850815349, 5.467708834, -3.807558748
        W707, W907 = 4.661571101, -8.124434579
        W808, W1008 = -5.745863967, 11.88290720
        W909 = 7.181868977
        W10010 = -9.077037610
        hn = np.sqrt(2*In)*np.cos(phin)
        kn = np.sqrt(2*In)*np.sin(phin)
        if ind == 1:
            hprime = hn + 2*W101*np.sin(lambdan)/arn**2 # + for Namouni, - for pomega
            kprime = kn - 2*W101*np.cos(lambdan)/arn**2 # - for Namouni, + for pomega
        else: #from DQT, the sign switches for the outer pair
            hprime = hn - 2*W101*np.sin(lambdan)/arn**2 # + for Namouni, - for pomega
            kprime = kn + 2*W101*np.cos(lambdan)/arn**2 # - for Namouni, + for pomega
        phiprime = np.arctan2(kprime, hprime) # 4 quadrant arctan
        #thetaprime = (pomegaprime - lambdan)
        thetaprime = (phiprime - lambdan)
    
        
        #implicit I update
        Iprime = 0.5*(hprime**2 + kprime**2)
        Inp1_prev = Iprime #first guess = Iprime
        phinp1_prev = phiprime
        while True: #iteratively update using implicit scheme
            Inp1o1 = 2*(W301/arn**4*(2*Inp1_prev)**1.5+W501/arn**6*(2*Inp1_prev)**2.5+W701/arn**8*(2*Inp1_prev)**3.5+W901/arn**10*(2*Inp1_prev)**4.5)*np.sin(thetaprime)
            Inp1o2 = 4*(W202/arn**3*(2*Inp1_prev)+W402/arn**5*(2*Inp1_prev)**2+W602/arn**7*(2*Inp1_prev)**3+W802/arn**9*(2*Inp1_prev)**4+W1002/arn**11*(2*Inp1_prev)**5)*np.sin(2*thetaprime)
            Inp1o3 = 6*(W303/arn**4*(2*Inp1_prev)**1.5+W503/arn**6*(2*Inp1_prev)**2.5+W703/arn**8*(2*Inp1_prev)**3.5+W903/arn**10*(2*Inp1_prev)**4.5)*np.sin(3*thetaprime)
            Inp1o4 = 8*(W404/arn**5*(2*Inp1_prev)**2+W604/arn**7*(2*Inp1_prev)**3+W804/arn**9*(2*Inp1_prev)**4+W1004/arn**11*(2*Inp1_prev)**5)*np.sin(4*thetaprime)
            Inp1o5 = 10*(W505/arn**6*(2*Inp1_prev)**2.5+W705/arn**8*(2*Inp1_prev)**3.5+W905/arn**10*(2*Inp1_prev)**4.5)*np.sin(5*thetaprime)
            Inp1o6 = 12*(W606/arn**7*(2*Inp1_prev)**3+W806/arn**9*(2*Inp1_prev)**4+W1006/arn**11*(2*Inp1_prev)**5)*np.sin(6*thetaprime)
            Inp1o7 = 14*(W707/arn**8*(2*Inp1_prev)**3.5+W907/arn**10*(2*Inp1_prev)**4.5)*np.sin(7*thetaprime)
            Inp1o8 = 16*(W808/arn**9*(2*Inp1_prev)**4+W1008/arn**11*(2*Inp1_prev)**5)*np.sin(8*thetaprime)
            Inp1o9 = 18*(W909/arn**10*(2*Inp1_prev)**4.5)*np.sin(9*thetaprime)
            Inp1o10 = 20*(W10010/arn**11*(2*Inp1_prev)**5)*np.sin(10*thetaprime)
    
            Inp1_next = Iprime -Inp1o2-Inp1o3-Inp1o4-Inp1o5-Inp1o6-Inp1o7-Inp1o8-Inp1o9-Inp1o10
            phinp1o0 = W200*2/arn**3+W400*2**2*2*Inp1_prev/arn**5+W600*2**3*3*Inp1_prev**2/arn**7+W800*2**4*4*Inp1_prev**3/arn**9+W1000*2**5*5*Inp1_prev**4/arn**11
            phinp1o1 = 2*(W301*2**1.5*1.5*Inp1_prev**0.5/arn**4+W501*2**2.5*2.5*Inp1_prev**1.5/arn**6+W701*2**3.5*3.5*Inp1_prev**2.5/arn**8+W901*2**4.5*4.5*Inp1_prev**3.5/arn**10)*np.cos(thetaprime)
            phinp1o2 = 2*(W202*2/arn**3+W402*2**2*2*Inp1_prev/arn**5+W602*2**3*3*Inp1_prev**2/arn**7+W802*2**4*4*Inp1_prev**3/arn**9+W1002*2**5*5*Inp1_prev**4/arn**11)*np.cos(2*thetaprime)
            phinp1o3 = 2*(W303*2**1.5*1.5*Inp1_prev**0.5/arn**4+W503*2**2.5*2.5*Inp1_prev**1.5/arn**6+W703*2**3.5*3.5*Inp1_prev**2.5/arn**8+W903*2**4.5*4.5*Inp1_prev**3.5/arn**10)*np.cos(3*thetaprime)
            phinp1o4 = 2*(W404*2**2*2*Inp1_prev/arn**5+W604*2**3*3*Inp1_prev**2/arn**7+W804*2**4*4*Inp1_prev**3/arn**9+W1004*2**5*5*Inp1_prev**4/arn**11)*np.cos(4*thetaprime)
            phinp1o5 = 2*(W505*2**2.5*2.5*Inp1_prev**1.5/arn**6+W705*2**3.5*3.5*Inp1_prev**2.5/arn**8+W905*2**4.5*4.5*Inp1_prev**3.5/arn**10)*np.cos(5*thetaprime)
            phinp1o6 = 2*(W606*2**3*3*Inp1_prev**2/arn**7+W806*2**4*4*Inp1_prev**3/arn**9+W1006*2**5*5*Inp1_prev**4/arn**11)*np.cos(6*thetaprime)
            phinp1o7 = 2*(W707*2**3.5*3.5*Inp1_prev**2.5/arn**8+W907*2**4.5*4.5*Inp1_prev**3.5/arn**10)*np.cos(7*thetaprime)
            phinp1o8 = 2*(W808*2**4*4*Inp1_prev**3/arn**9+W1008*2**5*5*Inp1_prev**4/arn**11)*np.cos(8*thetaprime)
            phinp1o9 = 2*(W909*2**4.5*4.5*Inp1_prev**3.5/arn**10)*np.cos(9*thetaprime)
            phinp1o10 = 2*(W10010*2**5*5*Inp1_prev**4/arn**11)*np.cos(10*thetaprime)
            phinp1_next = phiprime-phinp1o0-phinp1o1-phinp1o2-phinp1o3-phinp1o4-phinp1o5-phinp1o6-phinp1o7-phinp1o8-phinp1o9-phinp1o10
            
            
            if np.abs((Inp1_next-Inp1_prev)/Inp1_next) < 1e-12 and np.abs((phinp1_next-phinp1_prev)/phinp1_next) < 1e-12: 
                #break when satisfied
                break
            Inp1_prev = Inp1_next
            phinp1_prev = phinp1_next
        Inp1 = Inp1_next
        phinp1 = phinp1_next
        Knp1 = Kn + Inp1 - In
        hnp1 = np.sqrt(2*Inp1)*np.cos(phinp1)
        knp1 = np.sqrt(2*Inp1)*np.sin(phinp1)
        lambdanp1 = lambdan + self._periodfn(Knp1)
        # if exact_period==True:
        #     lambdanp1 = lambdan + 2*np.pi*np.abs(1/((1+np.sqrt(8*Knp1/3)*epsilon)**(-3/2)-1)) #same lambda
        # else:
        #     lambdanp1 = lambdan + 4*np.pi/(3*epsilon*np.sqrt(8*Knp1/3))
        return Inp1, Knp1, phinp1, lambdanp1, hnp1, knp1

        #This function runs Namouni mapping for multiple conjunctions
    def runmap(self, I0, K0, phi0, lambda0, Nconj, order):
        try:                # minimize function later passes a 1-element list, so make it work for both scalar and list
            K0 = K0[0]
        except:
            pass
        Is, Ks, phis, lambdas, hs, ks = np.zeros(Nconj+1), np.zeros(Nconj+1), np.zeros(Nconj+1), np.zeros(Nconj+1), np.zeros(Nconj+1), np.zeros(Nconj+1)
        Is[0] = I0
        Ks[0] = K0
        phis[0] = phi0
        lambdas[0] = lambda0
        ar0 = np.sqrt(8*K0/3)
        
        for i in range(Nconj):
            if order==1:
                Is[i+1], Ks[i+1], phis[i+1], lambdas[i+1], hs[i+1], ks[i+1] = self.namouni_fo_step(Is[i], Ks[i], phis[i], lambdas[i], ar0, ind)
            elif order==2:
                Is[i+1], Ks[i+1], phis[i+1], lambdas[i+1], hs[i+1], ks[i+1] = self.namouni_so_step(Is[i], Ks[i], phis[i], lambdas[i], ar0, ind)
            elif order==4:
                Is[i+1], Ks[i+1], phis[i+1], lambdas[i+1], hs[i+1], ks[i+1] = self.namouni_4o_step(Is[i], Ks[i], phis[i], lambdas[i], ar0, ind)
            elif order==10:
                Is[i+1], Ks[i+1], phis[i+1], lambdas[i+1], hs[i+1], ks[i+1] = self.namouni_10o_step(Is[i], Ks[i], phis[i], lambdas[i], ar0, ind)
        return Is, Ks, phis, lambdas, hs, ks
    
    def __call__(self,X,Nconj,order):
        epsilon = self.epsilon
        e0, a_r0, phi0, lambda0 = X
        I0 = (e0/epsilon)**2/2
        K0 = 3/8*a_r0**2
        # x1 = x + eps * self.f(theta)
        # theta1 = theta - 2 * np.pi * x1
        # theta1 = self._modfn(theta1)
    
        Is, Ks, phis, lambdas, hs, ks = self.runmap(I0, K0, phi0, lambda0, Nconj, order)
        return Is, Ks, phis, lambdas, hs, ks
