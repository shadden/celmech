from sympy import S, diff, lambdify, symbols, Matrix, Expr
import pprint
from numpy import array
from collections import OrderedDict
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

# units in canonical transformation, Ham doesn't know about that

# All params and units are owned by Hamiltonians. PhaseSpaceStates are just numbers with no
# knowledge of units. 

# tracked and untracked variables handled by Hamiltonian
# we do want to be able to set things in phasespacestate from hamiltonian and update hamiltonian
# state is a property in hamiltonian that sets update if we call setter, and can handle the logic on 
# which phasepsacestate variables to return based on which dof are being tracked or not

# full values gives additional variables including conserved quantitties

class PhaseSpaceState(object):
    def __init__(self, qpvars, values, t = 0):
        self.t = t
        self.val = OrderedDict(zip(qpvars, values))

    @property
    def qpvars(self):
        return list(self.val.keys()) 
    @property
    def qppairs(self):
        return [(self.qpvars[i], self.qpvars[i+self.Ndof]) for i in range(self.Ndof)]
    @property 
    def Ndof(self):
        return int(len(self.val)/2)
    @property 
    def Ndim(self):
        return len(self.val)
    @property
    def values(self):
        return list(self.val.values()) 
    @values.setter
    def values(self,values):
        for key, value in zip(self.qpvars, values):
            self.val[key] = value
    def __str__(self):
        s = "t={0}".format(self.t)
        for var, val in self.val.items():
            s += ", {0}={1}".format(var, val)
        return s
    def __repr__(self):
        return "PhaseSpaceState(qpvars={0}, values={1}, t={2})".format(self.qpvars, self.values, self.t)

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
    def __init__(self, H, Hparams, state, untracked_qpvars=None):
        """
        Arguments
        ---------
        H : sympy expression
            Hamiltonian made up only of sympy symbols in state.qppairs and keys in Hparams
        Hparams : dict
            dictionary from sympy symbols for the constant parameters in H to their value
        state : PhaseSpaceState 
            Object for holding the dynamical state.
        
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
        if untracked_qpvars:
            self.untracked_qpvars = untracked_qpvars
        else:
            self.untracked_qpvars = []
        self._update()

    @property
    def qppairs(self):
        return self.state.qppairs

    @property
    def qpvars(self):
        return self.state.qpvars
   
    @property
    def values(self):
        return self.state.values

    @property
    def full_qpvars(self):
        return self.state.qpvars

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
        return PoissonBracket(exprn,self.H,self.qpvars,[])

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
        return PoissonBracket(exprn,self.NH,self.qpvars,[])

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
        # can we remove this block?
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
        
        qpvars = self.qpvars
        Ndim = self.state.Ndim
        self.Energy = lambdify(qpvars,self.NH,'numpy')
        self.derivs = {}
        self.Nderivs = []
        flow = []
        Nflow = []
        for v in self.qpvars:
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
    for qp_pair in state.qppairs:
        q,p = qp_pair
        pval,qval = ham.state.val[p],ham.state.val[q]
        if p not in ham.H.free_symbols:
            new_params[q] = qval
        elif q not in ham.H.free_symbols:
            new_params[p] = pval
        else:
            new_pq_pairs.append(pq_pair)
            new_qvals.append(qval)
            new_pvals.append(pval)
    new_vals = np.array(new_qvals + new_pvals)
    # change to RedcuedPhaseSpaceState
    new_state = PhaseSpaceState(new_pq_pairs, new_vals,state.t)
    new_ham = Hamiltonian(ham.H,new_params,new_state)
    return new_ham
