from sympy import S, diff, lambdify, symbols, Matrix
from numpy import array
import numpy as np
from scipy.integrate import ode
import scipy.special
from .miscellaneous import PoissonBracket
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

class PhaseSpaceState(object):
    def __init__(self, pqpairs, initial_values,t = 0):
        self.t = t
        self.pqpairs = pqpairs
        self.Ndof  = len(pqpairs)
        self.qpvars_list = [q for p,q in self.pqpairs] + [p for p,q in self.pqpairs]
        # numerical values of qpvars-list
        self._values = array(initial_values)

    def as_rule(self):
        return dict(zip(self.qpvars_list,self.values))
    @property 
    def Ndim(self):
        return 2 * self.Ndof
    @property
    def values(self):
        return self._values
    @values.setter
    def values(self,values):
        self._update_from_values(values)
        self._values = values 
    def _update_from_values(self,values):
        pass

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
    def __init__(self, H, Hparams, state):
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
        def update_state_from_list(self, state, y, t):
            updates state object from a list of values y for the variables in the same order as pqpairs
            and integrator time 't'
        """
        self.state = state
        self.Hparams = Hparams
        self.H = H
        self._update()

    @property
    def pqpairs(self):
        return self.state.pqpairs

    @property
    def qpvars_list(self):
        return self.state.qpvars_list

    def Lie_deriv(self,exprn):
        r"""
        Return the Lie derivative of an expression with respect to the Hamiltonian.
        In other word, compute 

        .. math::
            \mathcal{L}_{H}f

        where :math:`f` is the argument passed and :math:`\mathcal{L}_H \equiv [\cdot,H]` 
        is the Lie derivative operator.

        Arguments
        ---------
        exprn : sympy expression
            The expression to take the Lie derivative of.

        Returns
        -------
        sympy expression
            sympy expression for the resulting derivative.
        """
        return PoissonBracket(exprn,self.H,self.qpvars_list,[])

    def NLie_deriv(self,exprn):
        r"""
        Return the Lie derivative of an expression with respect to the Hamiltonian
        with numerical values substituted for parameters. Equivalent to 
        :meth:`~poincare.Hamiltonian.Lie_deriv` but using the NH attribute 
        rather than the H attribute to compute brackets.  

        Arguments
        ---------
        exprn : sympy expression
            The expression to take the Lie derivative of.

        Returns
        -------
        sympy expression
            sympy expression for the resulting derivative.
        """
        return PoissonBracket(exprn,self.NH,self.qpvars_list,[])

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
            the scipy.ode 'dop853' integrator.  Valid 
            keyword options can be found 
            `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_.
        """
        try:
            time /= self.state.params['tau'] # scale time if defined
        except:
            pass
        if not hasattr(self, 'Nderivs'):
            self._update()
        try:
            self.integrator.integrate(time)
        except:
            raise AttributeError("Need to initialize Hamiltonian")
        self.state.t = self.integrator.t
        self.state.values = self.integrator.y
        #self.update_state_from_list(self.state, self.integrator.y)

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
        
        qpvars = self.qpvars_list
        Ndim = self.state.Ndim
        self.Energy = lambdify(qpvars,self.NH,'numpy')
        self.derivs = {}
        self.Nderivs = []
        flow = []
        Nflow = []
        for v in self.qpvars_list:
            deriv = self.Lie_deriv(v)
            Nderiv = self.NLie_deriv(v)
            self.derivs[v] = self.Lie_deriv(v)
            flow.append(deriv)
            Nflow.append(Nderiv)

        self.flow = Matrix(flow)
        self.jac = Matrix(Ndim,Ndim, lambda i,j: diff(flow[i],qpvars[j]))

        self.Nderivs = [lambdify(qpvars,fun) for fun in Nflow]
        self.Nflow = lambdify(qpvars,Nflow)
        NjacMtrx = Matrix(Ndim,Ndim, lambda i,j: diff(Nflow[i],qpvars[j]))
        self.Njac = lambdify(qpvars,NjacMtrx)
        self.integrator = ode(
                lambda t,y: self.Nflow(*y),
                jac = lambda t,y: self.Njac(*y))
        self.integrator.set_integrator('dop853')# ('lsoda') #
        self.integrator.set_initial_value(self.state.values)

def reduce_hamiltonian(ham):
    state = ham.state
    new_params = ham.Hparams.copy()
    free_symbols = ham.H.free_symbols
    pq_val_rule = ham.state.as_rule()
    new_pq_pairs= []
    new_qvals = []
    new_pvals = []
    for pq_pair in state.pqpairs:
        p,q = pq_pair
        pval,qval = pq_val_rule[p],pq_val_rule[q]
        if p not in ham.H.free_symbols:
            new_params[q] = qval
        elif q not in ham.H.free_symbols:
            new_params[p] = pval
        else:
            new_pq_pairs.append(pq_pair)
            new_qvals.append(qval)
            new_pvals.append(pval)
    new_vals = np.array(new_qvals + new_pvals)
    new_state = PhaseSpaceState(new_pq_pairs, new_vals,state.t)
    new_ham = Hamiltonian(ham.H,new_params,new_state)
    return new_ham
