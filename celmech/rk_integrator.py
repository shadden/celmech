import warnings
from scipy.linalg import solve as lin_solve
from numpy import sqrt
import numpy as np
_rt2 = sqrt(2)
_rt2_inv = 1 / _rt2 
_ImplicitMidpoint = {
        'a':np.array([[0.5]]),
        'c':np.array([0.5]),
        'b':np.array([1])
        }
_LobattoIIIB = {
    'a':np.array([
        [1/6, -1/6, 0],
        [1/6, 1/3, 0],
        [1/6, 5/6, 0]
    ]),
    'c':np.array([0, 1/2, 1]),
    'b':np.array([1/6, 2/3, 1/6])
}

_GL4 = {
    'a':np.array([
        [1/4, 1/4-1/6*sqrt(3)],
        [ 1/4+1/6*sqrt(3), 1/4]
    ]),
    'c':np.array([1/2 - 1/6*sqrt(3),1/2 + 1/6*sqrt(3)]),
    'b':np.array([1/2,1/2])
}

_GL6 = {
    'a':np.array([
        [5/36,2/9-1/15*sqrt(15),5/36-1/30*sqrt(15)],
        [5/36 + 1/24*sqrt(15),2/9,5/36 - 1/24*sqrt(15)],
        [5/36 + 1/30*sqrt(15),   2/9 + 1/15*sqrt(15),5/36]
    ]),
    'c':np.array([1/2 - 1/10*sqrt(15), 1/2, 1/2 + 1/10*sqrt(15)]),
    'b':np.array([5/18,  4/9,  5/18])
}

# Explicit methods
_RK4 = {
        'a':np.array([
            [0.,0.,0.,0.],
            [0.5,0.,0.,0.],
            [0,0.5,0.,0.],
            [0,0,1,0]
            ]),
        'c':np.array([0.,0.5,0.5,1.]),
        'b':np.array([1/6,1/3,1/3,1/6])
}

_ExplicitMidpoint = {
        'a':np.array([
            [0,0],
            [0.5,0]
            ]),
        'b':np.array([0,1]),
        'c':np.array([0,0.5])
        }

# all methods
_rk_methods = {
        'ImplicitMidpoint':_ImplicitMidpoint,
        'LobattoIIIB':_LobattoIIIB,
        'GL4':_GL4,
        'GL6':_GL6,
        'RK4':_RK4,
        'ExplicitMidpoint':_ExplicitMidpoint
}

class RKIntegrator():
    def __init__(self,f,f_and_Df,Ndim,dt,rtol,atol,rk_method,rk_root_method,max_iter):
        
        self.f = f
        self.f_and_Df = f_and_Df
        self.dt = dt
        self.Ndim = Ndim
        self.rtol = rtol
        self.atol = atol
        self.rk_root_method = rk_root_method
        self.rk_method = rk_method
        self.max_iter = max_iter
        
            
        # Set tolerance and iteration parameters
        tols_allowed = atol >=0 and rtol >=0
        tols_allowed = tols_allowed and (atol > 0 or rtol > 0)
        assert tols_allowed, "Tolerances must be non-negative and at least one tolerance must be positive." 
        
    @property
    def dt(self):
        return self._dt
    
    @dt.setter
    def dt(self,dt):
        self._dt = dt
        
    @property
    def rk_method(self):
        return self._rkmethod

    @rk_method.setter
    def rk_method(self,rkmethod):
        if rkmethod in _rk_methods.keys():
            self._rkmethod = rkmethod
            self._rk_tableau=_rk_methods[rkmethod]
            self.rk_a= self._rk_tableau['a']
            self.rk_b= self._rk_tableau['b']
            self.rk_c= self._rk_tableau['c']
            self.rk_s = len(self.rk_c)
        else:
            methods_list = "\n".join(["\t{:s}".format(method) for method in _rk_methods.keys()])
            raise ValueError("'rkmethod' must be one of:\n" + methods_list) 

        if self.rk_root_method=='explicit':
            for i,ai  in enumerate(self.rk_a):
                assert np.alltrue(ai[i:]==0), "RK tableau cannot be solved explicitly"

    @property 
    def rk_root_method(self):
        return self._rk_root_method
    
    @rk_root_method.setter
    def rk_root_method(self,rk_root_method):
        if rk_root_method == 'Newton':
            self.rk_step = self._implicit_rk_step_newton
        elif rk_root_method == 'quasi-Newton':
            self.rk_step = self._implicit_rk_step_quasi_newton
        elif rk_root_method == 'fixed_point':
            self.rk_step = self._implicit_rk_step_fixed_point
        elif rk_root_method == 'explicit':
            self.rk_step = self._explicit_rk_step
        else:
            raise ValueError("'rk_root_method' must be either 'Newton', 'quasi-Newton', 'fixed_point', or 'explicit'")
        self._rk_root_method = rk_root_method

    def _implicit_step_root_eqn(self,K,y):
        h = self.dt
        a = self.rk_a
        s = self.rk_s
        Ndim = self.Ndim
        f_and_Df = self.f_and_Df
        k = K.reshape((s,Ndim))
        knew = np.zeros((s,Ndim))
        Dkdy = np.zeros((s,Ndim,Ndim))
        ytemps = y + h * a @ k
        Imtrx = np.eye(s * Ndim)
        for i,ytemp in enumerate(ytemps):
            knew[i],Dkdy[i] = f_and_Df(ytemp)
        fOfK=np.hstack(knew)
        g = K - fOfK
        Dg = Imtrx - h * np.block([[ a[i,j] * Dkdy[i] for j in range(s)] for i in range(s)])
        return g,Dg

    def _implicit_rk_step_newton(self,y):
        """
        Advance ODE for input y for a timestep h
        using an implicit Runge-Kutta method defined by the
        Butcher tableau [a,b,c].

        Arguments
        ---------
        qp_vec
        """
        h = self.dt
        a = self.rk_a
        b = self.rk_b
        c = self.rk_c
        s = self.rk_s
        Ndim = self.Ndim
        f = self.f
        f_and_Df = self.f_and_Df
        
        max_iter = self.max_iter
        k = np.zeros((s,Ndim))
        Dkdy = np.zeros((s,Ndim,Ndim))
        rtol = self.rtol
        atol = self.atol
        Imtrx = np.eye(s * Ndim)
        #
        
        # Generate initial guesses from second-order approx.
        ####################################################
        # This appears to be sligtly faster than
        # than generating guesses via RK4, as in
        #  _implicit_rk_step_fix_point, but I haven't
        # tested it extensively.
        ####################################################
        ktemp,Dktemp = f_and_Df(y)
        d2ydt2 = Dktemp @ ktemp
        k = np.array([ktemp + ci*h*d2ydt2 for ci in c])
        # Main loop
        K = np.hstack(k)
        for itr in range(max_iter):
            g,Dg = self._implicit_step_root_eqn(K,y)
            dK = -1 * lin_solve(Dg,g)
            K += dK
            if np.alltrue( np.abs(dK) < rtol * np.abs(K) + atol ):
                break
        else:
            warnings.warn("'implicit_rk_step' reached maximum number of iterations ({})".format(max_iter))
        k = K.reshape(s,Ndim) 
        ynew = y + h * b @ k
        return ynew
    
    def _rk4_step(self,y,ydot,h):
        f = self.f
        h_by_2 = 0.5 * h
        k1 = ydot
        y2 = y + h_by_2 * k1
        k2 = f(y2)
        y3 = y + h_by_2 * k2
        k3 = f(y3)
        y4 = y + h * k3
        k4 = f(y4)
        yout = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6.
        return yout,f(yout)

    def _explicit_rk_step(self,qp_vec):
        h = self._dt
        a = self.rk_a
        b = self.rk_b
        s = self.rk_s
        f = self.f
        y = qp_vec
        k = np.zeros((s,self.Ndim))
        for i,ai in enumerate(a):
            k[i] = f(y + h * ai @ k)
        ynew = y + h * b @ k
        return ynew

    def _implicit_rk_step_fixed_point(self,qp_vec):
        """
        Advance ODE for input qpve for a timestep h
        using an implicit Runge-Kutta method defined by the
        Butcher tableau [a,b,c].

        Arguments
        ---------
        qp_vec
        """
        h = self._dt
        a = self.rk_a
        b = self.rk_b
        c = self.rk_c
        f = self.f
        y = qp_vec
        ktemp = f(qp_vec)
        max_iter = self.max_iter
        k = np.zeros((self.rk_s,self.Ndim))
        rtol = self.rtol
        atol = self.atol
        # 
        for i,ci in enumerate(c):
            if ci == 0:
                k[i,:] = ktemp 
            else:
                _,k[i,:] = self._rk4_step(y,ktemp,ci*h)
        for itr in xrange(max_iter):
            k_old = k.copy()
            ytemps = y + h * a @ k 
            k = np.array([f(ytemp) for ytemp in ytemps])
            delta_ks = np.abs(k-k_old)
            if np.alltrue( delta_ks < rtol * np.abs(k_old) + atol ):
                break
        else:
            warnings.warn("'implicit_rk_step' reached maximum number of iterations ({})".format(max_iter))
        ynew = y + h * b @ k
        return ynew
    
    def _implicit_rk_step_quasi_newton(self,qp_vec):
        """
        Advance ODE for input qpve for a timestep h
        using an implicit Runge-Kutta method defined by the
        Butcher tableau [a,b,c].

        Arguments
        ---------
        qp_vec
        """
        h = self._dt
        a = self.rk_a
        b = self.rk_b
        c = self.rk_c
        s = self.rk_s
        Ndim = self.Ndim
        f = self.f
        f_and_Df = self.f_and_Df
        y = qp_vec
        Imtrx = np.eye(s * Ndim)
        max_iter = self.max_iter
        rtol = self.rtol
        atol = self.atol

        # set up method
        k = np.zeros((s,Ndim))
        ktemp,Dkdy0 = f_and_Df(y)
        d2ydt2 = Dkdy0 @ ktemp
        k = np.array([ktemp + ci*h*d2ydt2 for ci in c])
        Dgdy0 = Imtrx - h * np.block([[ a[i,j] * Dkdy0 for j in range(s)] for i in range(s)])
        Dgdy0_inv = np.linalg.inv(Dgdy0)
        # Main loop
        for itr in range(max_iter):
            ytemps = y + h * a @ k
            g = np.hstack(k-np.array([f(ytemp) for ytemp in ytemps]))
            dK = -1 * Dgdy0_inv @ g
            dk = dK.reshape(s,Ndim)
            k += dk
            if np.alltrue( np.abs(dk) < rtol * np.abs(k) + atol ):
                break
        else:
            warnings.warn("'implicit_rk_step' reached maximum number of iterations ({})".format(max_iter))
        ynew = y + h * b @ k
        return ynew

