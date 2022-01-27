from sympy import S, diff, lambdify, symbols, Matrix, Expr
from collections import MutableMapping
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

# Should reduce hamiltonian only check for cyclic q, and then set both q and p as parameters? 
# There are cases where p shows up in hamiltonian but not q

# Do we call update in fullqp setter? Do we add a needs_update that only gets called on integrate(or others)?

# units in canonical transformation, Ham doesn't know about that

# All params and units are owned by Hamiltonians. PhaseSpaceStates are just numbers with no
# knowledge of units. 

# tracked and untracked variables handled by Hamiltonian
# we do want to be able to set things in phasespacestate from hamiltonian and update hamiltonian
# state is a property in hamiltonian that sets update if we call setter, and can handle the logic on 
# which phasepsacestate variables to return based on which dof are being tracked or not

# full values gives additional variables including conserved quantitties

class PhaseSpaceState(object):
    """
    A general class for describing the phase-space state
    of a Hamiltonian dynamical system.
    
    Attributes
    ----------
    qp : OrderedDict
        An ordered dictionary containing the canonical
        coordiantes and momenta as keys and their numerical
        values as dictionary values.
    qpvars : list
        List of variable symbols used for the canonical
        coordinates and momenta.
    qppairs : list
        A list of the 2-tuples :math:`(q_i,p_i)`.
    Ndof : int
        The number of degrees of freedom.
    values : list
        List of the numerical values of `qpvars`.
    """
    def __init__(self, qpvars, values, t = 0):
        """
        Arguments
        ---------
        qpvars : list of symbols
            List of symbols to use for the canonical coordiantes
            and momenta. The list should be of length 2*N for an
            integer N with entries 0,...,N-1 representing the
            canonical coordiante variables and entries N,...,2*N-1
            representing the corresponding conjugate momenta.
        values : array-like
            The numerical values assigned to the canonical coordiantes
            and momenta.
        t : float, optional
            The current time of system represented by the phase-space
            state. Default value is 0.
        """
        self.t = t
        self.qp = OrderedDict(zip(qpvars, values))

    @property
    def qpvars(self):
        return list(self.qp.keys()) 
    @property
    def qppairs(self):
        return [(self.qpvars[i], self.qpvars[i+self.Ndof]) for i in range(self.Ndof)]
    @property 
    def Ndof(self):
        return int(len(self.qp)/2)
    @property 
    def Ndim(self):
        return len(self.qp)
    @property
    def values(self):
        return list(self.qp.values()) 
    @values.setter
    def values(self,values):
        for key, value in zip(self.qpvars, values):
            self.qp[key] = value
    def __str__(self):
        s = "t={0}".format(self.t)
        for var, val in self.qp.items():
            s += ", {0}={1}".format(var, val)
        return s
    def __repr__(self):
        return "PhaseSpaceState(qpvars={0}, values={1}, t={2})".format(self.qpvars, self.values, self.t)

class Fullqp(MutableMapping):
    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian

    def __getitem__(self, key):
        try: # first try to find in the dynamical variables qp, if not, look for conserved quantities in Hparams (for reduce_hamiltonian)
            return self.hamiltonian.qp[key]
        except:
            try:
                return self.hamiltonian.Hparams[key]
            except:
                raise AttributeError('Variable {0} not found'.format(key))

    def __setitem__(self, key, value):
        try: # first try to find in the dynamical variables qp, if not, look for conserved quantities in Hparams (for reduce_hamiltonian)
            self.hamiltonian.qp[key] = value
        except:
            try:
                self.hamiltonian.Hparams[key] = value
                # NEED TO CALL UPDATE OR SET FLAG!
            except:
                raise AttributeError('Variable {0} not found'.format(key))

    def __delitem__(self, key):
        raise AttributeError("deleting variables not implemented.")

    def __iter__(self):
        for key in self.hamiltonian.full_qpvars:
            yield key

    def __len__(self):
        return len(self.hamiltonian.full_qpvars)

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
    def __init__(self, H, Hparams, state, full_qpvars=None):
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
        self._full_qpvars = full_qpvars
        self._update()
   

    @property
    def t(self):
        return self.state.t

    @property 
    def Ndof(self):
        return self.state.Ndof

    @property 
    def Ndim(self):
        return self.state.Ndim
    
    @property
    def qp(self):
        return self.state.qp
    
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
    def full_Ndof(self):
        return int(self.full_Ndim/2)

    @property 
    def full_Ndim(self):
        return len(self.full_qpvars)
    
    @property
    def full_qp(self):
        full_qp = Fullqp(self)
        return full_qp

    @property
    def full_qpvars(self):
        if self._full_qpvars:
            return self._full_qpvars
        else:
            return self.qpvars
    
    @property
    def full_values(self):
        return list(self.full_qp.values())

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

# should this be a member function?
def reduce_hamiltonian(ham):
    state = ham.state
    new_params = ham.Hparams.copy()
    #pq_val_rule = ham.state.as_rule()
    #new_pq_pairs= []
    untracked_q, untracked_p = [], []
    new_q, new_p = [], []
    new_qvals, new_pvals = [], []
    qppairs = [(state.qpvars[i], state.qpvars[i+state.Ndof]) for i in range(state.Ndof)]
    for qp_pair in qppairs:
        q,p = qp_pair
        pval,qval = state.qp[p],state.qp[q]
        if q not in ham.H.free_symbols or p not in ham.H.free_symbols: # if q is cyclic, p is conserved and we ignore that dof
            untracked_q.append(q)
            untracked_p.append(p)
            new_params[q] = qval
            new_params[p] = pval
        else:
            new_q.append(q)
            new_p.append(p)
            new_qvals.append(qval)
            new_pvals.append(pval)
    new_vals = np.array(new_qvals + new_pvals)
    new_qpvars = new_q + new_p
    untracked_qpvars = untracked_q + untracked_p
    new_state = PhaseSpaceState(new_qpvars, new_vals,state.t)
    new_ham = Hamiltonian(ham.H,new_params,new_state, full_qpvars = ham.full_qpvars)
    return new_ham
