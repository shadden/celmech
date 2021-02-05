from sympy import S, diff, lambdify, symbols 
from scipy.integrate import ode
import scipy.special
def _my_elliptic_e(*args):
    if len(args) == 1:
        return scipy.special.ellipe(*args)
    else:
        return scipy.special.ellipeinc(*args)
_lambdify_kwargs = {'modules':['numpy', {
    'elliptic_k': scipy.special.ellipk,
    'elliptic_f': scipy.special.ellipkinc,
    'elliptic_e': _my_elliptic_e
    }]} 

class Hamiltonian(object):
    """
    A general class for describing and evolving Hamiltonian systems.

    Attributes
    ----------

    state : object
        An object that stores the dynamical state of the system.

    H : sympy expression
        Symbolic expression for system's Hamiltonian.

    pqpars : list
        List of canonical variable pairs. 
    """
    def __init__(self, H, pqpairs, Hparams, initial_state):
        """
        Arguments
        ---------
        H : sympy expression
            Hamiltonian made up only of sympy symbols in pqpairs and keys in Hparams
        pqpairs : list
            list of momentum, position pairs [(P1, Q1), (P2, Q2)], where each element is a sympy symbol
        Hparams : dict
            dictionary from sympy symbols for the constant parameters in H to their value
        initial_conditions : object
            Arbitrary object for holding the dynamical state.
        
        
        In addition to the above, one needs to write 2 methods to map between the two objects:
        def state_to_list(self, state): 
            returns a list of values from state in the same order as pqpairs e.g. [P1,Q1,P2,Q2]
        def update_state_from_list(self, state, y):
            updates state object from a list of values y for the variables in the same order as pqpairs
        """
        self.state = initial_state
        self.Hparams = Hparams
        self.H = H
        self.pqpairs = pqpairs
        self.varsymbols = [var for pqpair in self.pqpairs for var in pqpair]
        
        self._update()

    def integrate(self, time, integrator_kwargs={}):
        """
        Evolve Hamiltonian system from current state
        to input time by integrating the equations of 
        motion.

        Arguments
        ---------
        time : float
            Time to advance Hamiltonian system to.
        integrator_kwargs : dict,optional
            A dictionary of integrator keyword arguments
            to pass to the integrator. ``celmech`` uses
            the scipy.ode 'lsoda' integrator.  Valid 
            keyword options can be found 
            `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_.
        """
        try:
            time /= self.state.params['tau'] # scale time if defined
        except:
            pass
        if not hasattr(self, 'Nderivs'):
            self._update()
        if time > self.integrator.t:
            try:
                self.integrator.integrate(time)
            except:
                raise AttributeError("Need to initialize Hamiltonian")
        self.update_state_from_list(self.state, self.integrator.y)

    def _update(self):
        self.NH = self.H # reset to Hamiltonian with all parameters unset
        # 
        # Update raw numerical constants first then update functions
        # Less hacky way to do this?
        function_keyval_pairs = []
        for key, val in self.Hparams.items(): 
            if isinstance(val,float):
                self.NH = self.NH.subs(key, val)
            else:
                function_keyval_pairs.append((key,val)) 
        for keyval in function_keyval_pairs:
            self.NH = self.NH.subs(keyval[0],keyval[1])
        
        self.Energy = lambdify(self.varsymbols,self.NH,'numpy')
        self.derivs = {}
        self.Nderivs = []
        for pqpair in self.pqpairs:
            p = pqpair[0]
            q = pqpair[1]
            self.derivs[p] = -diff(self.H, q)
            self.derivs[q] = diff(self.H, p)
            self.Nderivs.append(lambdify(self.varsymbols, -diff(self.NH, q), **_lambdify_kwargs))
            self.Nderivs.append(lambdify(self.varsymbols, diff(self.NH, p), **_lambdify_kwargs))

        def diffeq(t, y):
            dydt = [deriv(*y) for deriv in self.Nderivs]
            #print(t, y, dydt)
            return dydt
        self.integrator = ode(diffeq).set_integrator('dop853')# ('lsoda') #
        self.integrator.set_initial_value(self.state_to_list(self.state))
