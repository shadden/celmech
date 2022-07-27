from warnings import warn
import numpy as np
from .miscellaneous import poisson_bracket
from sympy import lambdify, solve, diff,sin,cos,sqrt,atan2,symbols,Matrix, Mul,S
from sympy import trigsimp, simplify, postorder_traversal, powsimp
from sympy.simplify.fu import TR10,TR10i
from .hamiltonian import reduce_hamiltonian, PhaseSpaceState, Hamiltonian
from .poincare import Poincare
from sympy import Wild,atan2,sin,cos,Function,sqrt,expand,simplify,Add
def _cos_n_atan(x,y,n):
    """
    Express cos(n*atan2(y,x)) in terms
    of x and y without trig functions.
    """
    assert n.is_integer
    if n<0:
        return _cos_n_atan(x,y,-1*n)
    if n==0:
        return 1
    else:
        r2 = x*x+y*y
        r = sqrt(r2)
        return x * _cos_n_atan(x,y,n-1)/r - y * _sin_n_atan(x,y,n-1)/r
def _sin_n_atan(x,y,n):
    """
    Express sin(n*atan2(y,x)) in terms
    of x and y without trig functions.
    """
    assert n.is_integer
    if n<0:
        return -1 * _sin_n_atan(x,y,-1*n)
    if n==0:
        return 0
    else:
        r2 = x*x+y*y
        r = sqrt(r2)
        return x*_sin_n_atan(x,y,n-1)/r + y*_cos_n_atan(x,y,n-1)/r

def _simplify_atans(exprn):
    """
    Expand any occurences of cos(a + n atan(y,x)) 
    and sin(a + n atan(y,x)).
    """
    pnames = ('a','x','y')
    a_w,x_w,y_w = symbols(','.join(n_+"_w" for n_ in pnames),cls=Wild,real=True)
    n_w = Wild('n_w',properties=(lambda x: x.is_integer,))
    c = Function("c",real=True)
    s = Function("s",real=True)
    cos_patt1 = cos(n_w * atan2(y_w,x_w))
    sin_patt1 = sin(n_w * atan2(y_w,x_w))
    #cos_patt2 = cos(a_w + n_w * atan2(y_w,x_w))
    #sin_patt2 = sin(a_w + n_w * atan2(y_w,x_w))
    res = TR10(exprn)
    res = res.replace(cos_patt1,c(x_w,y_w,n_w))
    res = res.replace(sin_patt1,s(x_w,y_w,n_w))
    #res = res.replace(
    #    cos_patt2,
    #    c(x_w,y_w,n_w)*cos(a_w) - s(x_w,y_w,n_w)*sin(a_w)
    #)
    #res = res.replace(
    #    sin_patt2,
    #    s(x_w,y_w,n_w)*cos(a_w) + c(x_w,y_w,n_w)*sin(a_w)
    #)
    
    res = res.replace(c,_cos_n_atan)
    res = res.replace(s,_sin_n_atan)
    res = TR10i(res)
    return res

def _termwise_trigsimp(exprn):
    if not hasattr(exprn,'args'):
        return exprn
    if len(exprn.args)==0:
        return exprn
    return exprn.func(*[trigsimp(a) for a in exprn.args])
def _exprn_contains_funcs(exprn,funcs):
    for arg in postorder_traversal(exprn):
        if arg.func in funcs:
            return True
    return False
def _get_default_qpxy_symbols(N,cartesian_indices):
    qppairs = _get_default_qp_symbols(N)
    xypairs = _get_default_xy_symbols(N)
    func = lambda i: (xypairs[i][1],xypairs[i][0]) if i in cartesian_indices else qppairs[i]
    varpairs = list(map(func,range(N)))
    return varpairs
def _get_default_qp_symbols(N):
    default_qp_symbols=list(zip(
        symbols("Q(1:{})".format(N+1),real=True),
        symbols("P(1:{})".format(N+1),real=True)
    ))
    return default_qp_symbols

def _get_default_xy_symbols(N):
    default_xy_symbols=list(zip(
        symbols("x(1:{})".format(N+1),real=True),
        symbols("y(1:{})".format(N+1),real=True)
    ))
    return default_xy_symbols

class CanonicalTransformation():
    r"""
    A class for representing canonical transformations.
    
    An instance represents a transformation from old canonical variables,
    :math:`(q,p)` to new variables :math:`(q',p')`.

    Attributes
    ----------
    old_qp_vars : list
        A list of symbols representing the old variables, :math:`(q,p)`.
        The list should contain all coordinate variables, :math:`q`, followed
        by their conjugate momentum variables :math:`p`.
    new_qp_vars : list
        A list of symbols representing the new variables, :math:`(q',p')`.
        The list should contain all coordinate variables, :math:`q'`, followed
        by their conjugate momentum variables :math:`p'`.
    old_to_new_rule : dictionary
        A dictionary defining the substitution rules applied to transform from
        old canonical variables to new ones.
    new_to_old_rule : dictionary
        A dictionary defining the substitution rules applied to transform from
        new canonical variables to old ones.
    old_to_new_simplify : function
        A function that will be applied to expressions when transforming from
        old to new variables.
    new_to_old_simplify : function
        A function that will be applied to expression when transforming from
        new to old variables.
    params : dict
        A dictionary setting the numerical values of any parameters on which
        the transformation depends.
    H_scale : symbol or float
        A factor by which the transformation rescales the Hamiltonian. This is
        used when applying transformations that simulatneously rescale the
        momenta and the Hamiltonian.
    """
    def __init__(self,
            old_qp_vars,
            new_qp_vars,
            old_to_new_rule = None,
            new_to_old_rule = None,
            old_to_new_simplify = None,
            new_to_old_simplify = None,
            params = {},
            **kwargs
    ):
        self.old_qp_vars = old_qp_vars
        self.new_qp_vars = new_qp_vars
        self.params = params
        self.H_scale = kwargs.get('H_scale',1)

        if not (old_to_new_rule or new_to_old_rule):
            raise ValueError("Must specify at least one of 'old_to_new_rule' or 'new_to_old_rule'!")
        if old_to_new_rule:
            self.old_to_new_rule = old_to_new_rule
        else:
            self.new_to_old_rule = new_to_old_rule
            eqs = [v - self.new_to_old_rule[v] for v in self.new_qp_vars]
            self.old_to_new_rule = solve(eqs,self.old_qp_vars)
        
        if new_to_old_rule:
            self.new_to_old_rule = new_to_old_rule
        else:
            eqs = [v - self.old_to_new_rule[v] for v in self.old_qp_vars] 
            self.new_to_old_rule = solve(eqs,self.new_qp_vars)

        if old_to_new_simplify:
            self.old_to_new_simplify_function = old_to_new_simplify
        else:
            self.old_to_new_simplify_function = lambda x: x

        if new_to_old_simplify:
            self.new_to_old_simplify_function = new_to_old_simplify
        else:
            self.new_to_old_simplify_function = lambda x: x

        qpv_new = self.new_qp_vars
        qpv_old = self.old_qp_vars
        self._old_to_new_vfunc = lambdify(qpv_old,[self.N_new_to_old(v) for v in qpv_new])
        self._new_to_old_vfunc = lambdify(qpv_new,[self.N_old_to_new(v) for v in qpv_old])
   
    def old_to_new(self,exprn):
        r"""
        Transform an expression in terms of old canonical variables to one in
        terms of new varialbes.

        Arguments
        ---------
        exprn : sympy expression
            The expression to transform
        Returns
        -------
        sympy expression
            The transformed expression.
        """
        exprn = exprn.subs(self.old_to_new_rule)
        exprn = self.old_to_new_simplify_function(exprn)
        return exprn

    def N_old_to_new(self,exprn):
        r"""
        Same as
        :func:`~celmech.canonical_transformations.CanonicalTransformation.old_to_new`
        but with numerical values replacing any symbolic parameters.
        """
        exprn = self.old_to_new(exprn)
        Nexprn = exprn.subs(self.params)
        return Nexprn

    def new_to_old(self,exprn):
        r"""
        Transform an expression in terms of new canonical variables to one in
        terms of old varialbes.

        Arguments
        ---------
        exprn : sympy expression
            The expression to transform.
        Returns
        -------
        sympy expression
            The transformed expression.
        """
        exprn = exprn.subs(self.new_to_old_rule)
        exprn = self.new_to_old_simplify_function(exprn)
        return exprn
    
    def N_new_to_old(self,exprn):
        r"""
        Same as
        :func:`~celmech.canonical_transformations.CanonicalTransformation.new_to_old`
        but with numerical values replacing any symbolic parameters.
        """
        exprn = self.new_to_old(exprn)
        Nexprn = exprn.subs(self.params)
        return Nexprn

    def old_to_new_array(self,arr):
        r"""
        Convert an array of old variable values to the corresponding values of
        new canonical variables.

        Arguments
        ---------
        arr : array-like
            Value of old variables

        Returns
        -------
        new_arr : numpy.array
            Array of new variable values.
        """
        return np.array(self._old_to_new_vfunc(*arr))

    def new_to_old_array(self,arr):
        r"""
        Convert an array of new variable values to the corresponding values of
        old canonical variables.

        Arguments
        ---------
        arr : array-like
            Value of new variables

        Returns
        -------
        new_arr : numpy.array
            Array of old variable values.
        """
        return np.array(self._new_to_old_vfunc(*arr))

    def new_to_old_state(self,state):
        old_values = self.new_to_old_array(state.values)
        old_state = PhaseSpaceState(self.old_qp_vars,old_values,t=state.t)
        return old_state
    
    def old_to_new_hamiltonian(self,ham,do_reduction = False):
        r"""
        Apply canonical transformation to a
        :class:`~celmech.hamiltonian.Hamiltonian` object, transforming the
        Hamiltonian and producing a new object with a
        :attr:`~celmech.hamiltonian.Hamiltonian.state` attribute representing
        the phase space position in the new canonical variables.

        Arguments
        ---------
        ham : celmech.hamiltonian.Hamiltonian
            The Hamiltonian object to which transformation is applied.
        do_reduction : bool, optional
            If ``True``, automatically reduce the number of degrees of freedom
            by detecting which canonical variables do not appear in the
            expression of the transformed Hamiltonian.

        Returns
        -------
        new_ham : celmech.hamiltonian.Hamiltonian
            The transformed Hamiltonian.
        """
        # Get full values
        new_values = self.old_to_new_array(ham.full_values)
        # The full set of q,p variables
        old_full = set(ham.full_qp_vars)
        # Only q,p variables that are active
        old_active = set(ham.qp_vars)
        # Ignorable q,p variables
        old_ignorable = old_full.difference(old_active)

        # New, transformed Hamiltonian expression
        newH =  self.old_to_new(ham.H)
        
        new_pars = ham.H_params.copy()
        new_pars.update(self.params)

        # Handle possible cases where ignorable coordinates get transformed:
        #   First, remove old ingorable coordinates from the new H_params dictionary
        for ig in old_ignorable:
            new_pars.pop(ig,None)
        #   Next, loop through new variable pairs and detect ingorable ones.
        #       'is_active' stores a boolean array indicating which new q,p vars 
        #       will be active variables.
        is_active = np.ones(2*self.N_dof,dtype=bool)
        for i,qpnew in enumerate(self.new_qp_pairs):
            qnew,pnew = qpnew
            vars_set_qnew = self.new_to_old(qnew).free_symbols
            vars_set_pnew = self.new_to_old(pnew).free_symbols
            vars_set_qpnew = vars_set_qnew.union(vars_set_pnew)
            # A new DOF is ignorable if its expression in terms of 
            # old variables does not involve any of the old active 
            # variables.
            is_ignorable = len(old_active.intersection(vars_set_qpnew)) == 0 
            if is_ignorable:
                is_active[i] = False
                is_active[i+self.N_dof] = False
                #if qnew in newH.free_symbols:
                new_pars[qnew] = new_values[i]
                #if pnew in newH.free_symbols:
                new_pars[pnew] = new_values[i+self.N_dof]

        # The list of new active q,p variables.
        new_vars_reduced = [var for var,activeQ in zip(self.new_qp_vars,is_active) if activeQ]

        # New state consisting only of active variables.
        new_state = PhaseSpaceState(new_vars_reduced,new_values[is_active],ham.state.t)
        

        # Rescale Hamiltonian.
        # Warning--- this will not work properly if the top-level function is not sympy.Add
        # This should probably be treated more carefully.
        if self.H_scale != 1:
            assert newH.func == Add,"Re-scaling requires the top level node of the Hamiltonian expression tree must be sympy.Add"
        newH = newH.func(*[a*self.H_scale for a in newH.args])

        # New Hamiltonian: the new state only contains active variables
        # but full_qp_vars kwarg is passed the full set of new q,p variables.
        new_ham = Hamiltonian(newH,new_pars,new_state,full_qp_vars = self.new_qp_vars)
        if do_reduction:
            new_ham = reduce_hamiltonian(new_ham)
        return new_ham
    
    def new_to_old_hamiltonian(self,ham,do_reduction = False):
        old_state = self.new_to_old_state(ham.state)
        oldH = self.new_to_old(ham.H) / self.H_scale
        old_ham = Hamiltonian(oldH,ham.H_params,old_state)
        if do_reduction:
            old_ham = reduce_hamiltonian(old_ham)
        return old_ham
    
    def _test_new_to_old_canonical(self):
        pb = lambda q,p: poisson_bracket(q,p,self.old_qp_vars,[]) / self.H_scale
        bracket_results = [pb(self.N_new_to_old(qq),self.N_new_to_old(pp)).simplify() for qq,pp in self.new_qp_pairs]
        br_arr = [np.isclose(float(x.subs(self.params)),1,atol=1e-10) for x in bracket_results]
        return np.alltrue(br_arr)

    def _test_old_to_new_canonical(self):
        pb = lambda q,p: poisson_bracket(q,p,self.new_qp_vars,[]) * self.H_scale
        bracket_results = [pb(self.old_to_new(qq),self.old_to_new(pp)).simplify() for qq,pp in self.old_qp_pairs]
        br_arr = [np.isclose(float(x.subs(self.params)),1,atol=1e-10) for x in bracket_results]
        return np.alltrue(br_arr)
    def test_canonical(self):
        """
        Test whether the substitution rules of this tranformation constitute 
        a canonical transformation.

        Returns
        -------
        bool :
            Returns ``True`` if the transformation is canonical.
        """
        return self._test_new_to_old_canonical() and self._test_old_to_new_canonical()
    @property 
    def N_dof(self):
        return int(len(self.old_qp_vars)/2)

    @property
    def old_qp_pairs(self):
        return [(self.old_qp_vars[i], self.old_qp_vars[i+self.N_dof]) for i in range(self.N_dof)]
    
    @property
    def new_qp_pairs(self):
        return [(self.new_qp_vars[i], self.new_qp_vars[i+self.N_dof]) for i in range(self.N_dof)]

    @classmethod
    def from_type2_generating_function(cls,F2func,old_qp_vars,new_qp_vars,**kwargs):
        r"""
        Initialize a canonical transformation derived from a `type 2 generating
        function`_. Given a set of old variables :math:`(q,p)`, new variables,
        :math:`(Q,P)` and generating function :math:`F_2(q,P)`, the
        transformation rules are given by

        .. math::
            p &=& \frac{\partial }{\partial q}F_2(q,P) \\ 
            Q &=& \frac{\partial }{\partial P}F_2(q,P) 

        .. _type 2 generating function: https://en.wikipedia.org/wiki/Canonical_transformation#Type_2_generating_function

        Arguments
        ---------
        F2func : sympy expression
            The type 2 generating function
        old_qp_vars : array-like
            The list of old canonical variables
        new_qp_vars : array-like
            The list of new canonical variables

        Returns
        -------
        .CanonicalTransformation
            The resulting transformation.
        """
        N_dof = int(len(old_qp_vars)/2)
        old_qp_pairs = [(old_qp_vars[i], old_qp_vars[i+N_dof]) for i in range(N_dof)]
        new_qp_pairs = [(new_qp_vars[i], new_qp_vars[i+N_dof]) for i in range(N_dof)]
        eqs = [p - diff(F2func,q) for q,p in old_qp_pairs]
        eqs += [Q - diff(F2func,P) for Q,P in new_qp_pairs]
        o2n_soln = solve(eqs,old_qp_vars)

        if type(o2n_soln) is dict:
            o2n = o2n_soln
        elif type(o2n_soln) is list:
            msg ="Multiple roots found when solving for old variables in"
            msg+="of new. The desired transformation may not be produced"
            warn(msg)
            o2n = dict(zip(old_qp_vars,o2n_soln[0]))

        n2o_soln = solve(eqs,new_qp_vars)
        if type(n2o_soln) is dict:
            n2o = n2o_soln
        elif type(n2o_soln) is list:
            msg ="Multiple roots found when solving for new variables in"
            msg+="terms of old. The desired transformation may not be produced"
            warn(msg)
            n2o = dict(zip(new_qp_vars,n2o_soln[0]))

        return cls(old_qp_vars,new_qp_vars,old_to_new_rule=o2n,new_to_old_rule = n2o)
    
    @classmethod
    def cartesian_to_polar(cls,old_qp_vars,indices=None,polar_symbol_pairs=None,**kwargs):
        r"""
        Initialize a canonical transformation that transforms a user-specified
        subset of conjugate coordinate-momentum pairs, :math:`(q_i,p_i)`, to
        new 'polar-style' variable pairs

        .. math::
            (Q_i,P_i) = \left(\tan^{-1}(q_i/p_i), \frac{1}{2}(q_i^2+p_i^2)\right)

        Arguments
        ---------
        old_qp_vars : array-like
            The list of old canonical variables
        indices : array-list
            List of integer indices of the coordinate-momentum pairs to be
            transformed.
        polar_symbol_pairs : list, optional
            List of symbols to use for the new polar-style variables.
            Default symbols are :math:`(Q_i,P_i)` where the index i
            ranges over the number of variable pairs being transformed.

        Returns
        -------
        .CanonicalTransformation
            The resulting transformation.
        """
        N_dof = int(len(old_qp_vars)/2)
        old_qp_pairs = [(old_qp_vars[i], old_qp_vars[i+N_dof]) for i in range(N_dof)]
        if not indices:
            indices = range(N_dof)
        N = len(indices)
        if not polar_symbol_pairs:
            polar_symbol_pairs = _get_default_qp_symbols(N)
        new_q, new_p = [], []
        o2n,n2o = dict(),dict()
        for i,qp in enumerate(old_qp_pairs):
            if i in indices:
                y,x = qp 
                Q,P = polar_symbol_pairs.pop(0)
                o2n.update({ x:sqrt(2*P) * cos(Q) , y:sqrt(2*P) * sin(Q)})
                n2o.update({P:(x*x +y*y)/2, Q:atan2(y,x)})
                new_q.append(Q)
                new_p.append(P)
            # keep indentity transformation if index not passed
            else:
                id_tr = dict(zip(qp,qp))
                o2n.update(id_tr)
                n2o.update(id_tr)
                new_q.append(qp[0])
                new_p.append(qp[1])

        # add a simplify function to remove arctans
        atan2_simplify = lambda exprn: TR10i(simplify(TR10(exprn))) if _exprn_contains_funcs(exprn,[atan2]) else powsimp(exprn)
        n2o_simplify = lambda x: x.func(*[atan2_simplify(a) for a in x.args]) if len(x.args) > 0 else x
        kwargs.setdefault("old_to_new_simplify",_termwise_trigsimp)
        kwargs.setdefault("new_to_old_simplify",n2o_simplify)
        new_qp_vars = new_q + new_p
        return cls(old_qp_vars,new_qp_vars,o2n,n2o,**kwargs)
    
    @classmethod
    def polar_to_cartesian(cls,old_qp_vars,indices=None,cartesian_symbol_pairs=None,**kwargs):
        r"""
        Initialize a canonical transformation that transforms a user-specified
        subset of conjugate coordinate-momentum pairs, :math:`(q_i,p_i)`, to
        new 'cartesian-style' variable pairs

        .. math::
            (Q_i,P_i) = \left(\sqrt{2p_i}\sin q_i,\sqrt{2p_i}\cos q_i\right)

        Arguments
        ---------
        old_qp_vars : array-like
            The list of old canonical variables
        indices : array-list
            List of integer indices of the coordinate-momentum pairs to be
            transformed.
        cartesian_symbol_pairs : list, optional
            List of symbols to use for the new cartesian-style variables.
            Default symbols are :math:`(y_i,x_i)` where the index i ranges
            over the number of variable pairs being transformed.

        Returns
        -------
        .CanonicalTransformation
            The resulting transformation.
        """
        N_dof = int(len(old_qp_vars)/2)
        old_qp_pairs = [(old_qp_vars[i], old_qp_vars[i+N_dof]) for i in range(N_dof)]
        if not indices:
            indices = range(N_dof)
        N = len(indices)
        if not cartesian_symbol_pairs:
           cartesian_symbol_pairs = _get_default_xy_symbols(N)
        new_q, new_p = [], []
        o2n,n2o = dict(),dict()
        for i,qp in enumerate(old_qp_pairs):
            if i in indices:
                x,y = cartesian_symbol_pairs.pop(0)
                Q,P = qp 
                n2o.update({ x:sqrt(2*P) * cos(Q) , y:sqrt(2*P) * sin(Q)})
                o2n.update({P:(x*x +y*y)/2, Q:atan2(y,x)})
                new_q.append(y)
                new_p.append(x)
            # keep identity transformation if index not passed
            else:
                id_tr = dict(zip(qp,qp))
                o2n.update(id_tr)
                n2o.update(id_tr)
                new_q.append(qp[0])
                new_p.append(qp[1])

        # A simplify function to remove arctans
        o2n_simplify = lambda x: x.func(*[simplify(a.expand()) for a in _simplify_atans(x).args ]) if len(x.args)>0 else x
        #atan2_simplify = lambda exprn: TR10i(simplify(TR10(exprn))) if _exprn_contains_funcs(exprn,[atan2]) else powsimp(exprn)
        #o2n_simplify = lambda x: x.func(*[atan2_simplify(a) for a in x.args]) if len(x.args) > 0 else x
        kwargs.setdefault("old_to_new_simplify",o2n_simplify)
        new_qp_vars = new_q + new_p
        return cls(old_qp_vars,new_qp_vars,o2n,n2o,**kwargs)

    @classmethod
    def from_linear_angle_transformation(cls,old_qp_vars,Tmtrx,old_cartesian_indices=[],new_cartesian_indices=[],**kwargs):
        r"""
        Produces the transformation :math:`(\pmb{q},\pmb{p})\rightarrow(\pmb{Q},\pmb{P})` given by

        .. math::
            \begin{eqnarray}
                 \pmb{Q} = T \cdot \pmb{q} ~&;~
                 \pmb{P} = (T^{-1})^\mathrm{T} \cdot \pmb{p}
            \end{eqnarray}

        where :math:`T` is a user-specified matrix.


        Arguments
        ---------
        old_qp_vars : list
            List of old variable symbols.
        Tmtrx : array-like
            Matrix defining the canonical transformation.
        old_cartesian_indicies : list
            List of indices specifying which old variables are cartesian-style
            pairs.
        new_cartesian_indicies : list
            List of indices specify which new variables should be created as
            cartesian style pairs.

        Returns
        -------
        .CanonicalTransformation
        """
        try:
            N_dof = len(old_qp_vars)//2
            Tmtrx = np.array(Tmtrx).reshape(N_dof,N_dof)
        except:
            raise ValueError("'Tmtrx' could not be shaped into a {0}x{0} array".format(N_dof))
        old_qp_pairs = list(zip(old_qp_vars[:N_dof],old_qp_vars[N_dof:]))
        new_qp_pairs = kwargs.get("QPvars",_get_default_qpxy_symbols(N_dof,new_cartesian_indices))
        new_coords = Matrix([q for q,p in new_qp_pairs])
        new_momenta = Matrix([p for q,p in new_qp_pairs])

        to_angle = lambda i,qp,cartesian_indices: atan2(*qp) if i in cartesian_indices else qp[0] 
        to_action = lambda i,qp,cartesian_indices: (qp[0]**2 + qp[1]**2)/S(2) if i in cartesian_indices else qp[1] 

        old_angvars = Matrix([to_angle(i,qp,old_cartesian_indices) for i,qp in enumerate(old_qp_pairs)])
        old_actvars = Matrix([to_action(i,qp,old_cartesian_indices) for i,qp in enumerate(old_qp_pairs)])
        new_angvars = Matrix([to_angle(i,qp,new_cartesian_indices) for i,qp in enumerate(new_qp_pairs)])
        new_actvars = Matrix([to_action(i,qp,new_cartesian_indices) for i,qp in enumerate(new_qp_pairs)])

        Tmtrx = Matrix(Tmtrx)
        Tmtrx_inv  = Tmtrx.inv()

        n2o = dict()
        for i,qp_new,ang_exprn,act_exprn in zip(range(N_dof),new_qp_pairs,Tmtrx * old_angvars,Tmtrx_inv.T * old_actvars):
            qnew,pnew = qp_new
            if i in new_cartesian_indices:
                n2o[qnew] = simplify(sqrt(2*act_exprn) * TR10i(TR10(sin(ang_exprn))))
                n2o[pnew] = simplify(sqrt(2*act_exprn) * TR10i(TR10(cos(ang_exprn))))
            else:
                n2o[qnew] = ang_exprn
                n2o[pnew] = act_exprn
        
        o2n = dict()
        for i,qp_old,ang_exprn,act_exprn in zip(range(N_dof),old_qp_pairs,Tmtrx_inv * new_angvars,Tmtrx.T * new_actvars):
            qold,pold = qp_old
            if i in old_cartesian_indices:
                o2n[qold] = simplify(sqrt(2*act_exprn) * TR10i(TR10(sin(ang_exprn))))
                o2n[pold] = simplify(sqrt(2*act_exprn) * TR10i(TR10(cos(ang_exprn))))
            else:
                o2n[qold] = ang_exprn
                o2n[pold] = act_exprn
        
        new_ang_vars = set([new_qp_pairs[i][0] for i in range(N_dof) if i not in new_cartesian_indices])
        simplify_trig = lambda a: simplify(TR10i(a.expand())) if len(new_ang_vars.intersection(a.free_symbols))>0 else a
        o2n_simplify = lambda x: x.func(*[simplify_trig(arg) for arg in x.args]) if len(x.args)>0 else x
        kwargs.setdefault("old_to_new_simplify",o2n_simplify)
        return cls(old_qp_vars,list(new_coords) + list(new_momenta), o2n, n2o,**kwargs)

    @classmethod
    def from_poincare_angles_matrix(cls,p_vars,Tmtrx,new_qp_pairs=None,**kwargs):
        r"""
        Given an input :class:`~celmech.poincare.Poincare` instanace and an
        invertible matrix, :math:`T`, generate a transformation for which 
        the new angle variables are given by
        
        .. math::
            \pmb{Q} = T \cdot 
            \begin{pmatrix}
            \lambda_1 \\ \vdots \\ \lambda_N \\
            \gamma_1 \\ \vdots \\ \gamma_N \\
            q_1 \\ \vdots \\ q_N
            \end{pmatrix}~.

        Arguments
        ---------
        pvars : celmech.poincare.Poincare
            A Poincare instance with variables to which the resulting
            transformation will apply.
        Tmtrx : ndarray, shape (N,N)
            An invertible matrix determining the new canonical angle variables
            as linear combinations of the planets' orbital elements.
        new_qp_pairs : list, optional
            A list of the symbols to use for the newly-generated
            coordinate-momentum pairs. List entries should be given in the form
            of 2-tuples containing a coordinate symbol followed by its
            conjugate momentum variable. If no list is supplied, the default
            symbols :math:`(Q_i,P_i)` will be used.

        Returns
        -------
        .CanonicalTransformation
            The resulting transformation.
        """
        if not type(p_vars) == Poincare:
            raise TypeError("'p_vars' must be of type 'Poincare'")
        try:
            N_dof = p_vars.N_dof
            Tmtrx = np.array(Tmtrx).reshape(N_dof,N_dof)
        except:
            raise ValueError("'Tmtrx' could not be shaped into a {0}x{0} array".format(N_dof))
        Tmtrx = Matrix(Tmtrx)
        if not new_qp_pairs:
            new_qp_pairs = _get_default_qp_symbols(N_dof)
        Npl = p_vars.N - 1
        old_varslist = p_vars.qp_vars
        old_angvars = [old_varslist[3*i + 0] for i in range(Npl)] # lambdas
        old_angvars += [atan2(old_varslist[3*i + 1],old_varslist[3*(i+Npl) + 1]) for i in range(Npl)] # gammas
        old_angvars += [atan2(old_varslist[3*i + 2],old_varslist[3*(i+Npl) + 2]) for i in range(Npl)] # qs
        old_actvars = [old_varslist[3*(i+Npl) + 0] for i in range(Npl)] # Lambdas
        old_actvars += [(old_varslist[3*i + 1]**2 + old_varslist[3*(i+Npl) + 1]**2) / 2 for i in range(Npl)] # Gammas
        old_actvars += [(old_varslist[3*i + 2]**2 + old_varslist[3*(i+Npl) + 2]**2) / 2 for i in range(Npl)] # Qs

        n2o = dict(zip([Q for Q,P in new_qp_pairs],Tmtrx * Matrix(old_angvars)))
        n2o.update(dict(zip([P for Q,P in new_qp_pairs],Tmtrx.inv().transpose() * Matrix(old_actvars))))
        
        o2n=dict()
        old2new_angvars = Tmtrx.inv() * Matrix([Q for Q,P in new_qp_pairs])
        old2new_actvars = Tmtrx.transpose() * Matrix([P for Q,P in new_qp_pairs])
        Lambdas,lambdas = [[old_varslist[3*i + N] for i in range(Npl)] for N in (3*Npl,0)]
        kappas,etas,sigmas,rhos = [[old_varslist[3*i + N] for i in range(Npl)] for N in (1 + 3*Npl,1,2 + 3*Npl,2)]

        o2n.update(dict(zip(Lambdas,old2new_actvars[:Npl])))
        o2n.update(dict(zip(lambdas,old2new_angvars[:Npl])))

        kappa_exprns = [sqrt(2 * old2new_actvars[i]) * cos(old2new_angvars[i]) for i in range(Npl,2*Npl)]
        eta_exprns = [sqrt(2 * old2new_actvars[i]) * sin(old2new_angvars[i]) for i in range(Npl,2*Npl)]
        sigma_exprns = [sqrt(2 * old2new_actvars[i]) * cos(old2new_angvars[i]) for i in range(2*Npl,3*Npl)]
        rho_exprns = [sqrt(2 * old2new_actvars[i]) * sin(old2new_angvars[i]) for i in range(2*Npl,3*Npl)]
        o2n.update(dict(zip(kappas,kappa_exprns)))
        o2n.update(dict(zip(etas,eta_exprns)))
        o2n.update(dict(zip(sigmas,sigma_exprns)))
        o2n.update(dict(zip(rhos,rho_exprns)))
       
        new_qp_vars = [Q for Q,P in new_qp_pairs] + [P for Q,P in new_qp_pairs]
        return cls(p_vars.qp_vars,new_qp_vars,o2n,n2o,old_to_new_simplify=_termwise_trigsimp)

    @classmethod
    def Lambdas_to_delta_Lambdas(cls,pham):
        r"""
        Generate a canonical transformation applicable to a 
        PoincareHamiltonian object, `pham`, that replaces 
        the canonical momenta :math:`\Lambda_i` conjugate to 
        the mean longitudes :math:`\lambda_i` with new momenta
        :math:`\delta\Lambda_i = \Lambda_i - \Lambda_{i,0}`.
        """
        Lambda0s = pham.Lambda0s[1:]
        N = pham.N
        Lambdas = symbols("Lambda(1:{})".format(N))
        dLambdas = symbols(r"\delta\Lambda_{{(1:{})}}".format(N))
        o2n = {L:L0+dL for L,L0,dL in zip(Lambdas,Lambda0s,dLambdas)}
        n2o = {dL:(L-L0) for L,L0,dL in zip(Lambdas,Lambda0s,dLambdas)}
        L2dL= dict(zip(Lambdas,dLambdas))
        qpnew = [v.subs(L2dL) for v in pham.qp_vars]
        qpold = pham.qp_vars
        o2n_full={v:v.subs(o2n) for v in pham.qp_vars}
        n2o_full={v:v.subs(n2o) for v in qpnew}
        params = {L0:pham.H_params[L0] for L0 in Lambda0s}
        return cls(qpold,qpnew,o2n_full,n2o_full,params = params)
    
    @classmethod
    def actions_to_delta_actions(cls, old_qp_vars, actions, delta_actions, actions_ref, params={}):
        r"""
        Generate a canonical transformation applicable to a 
        PoincareHamiltonian object, `pham`, that replaces 
        the canonical momenta :math:`\Lambda_i` conjugate to 
        the mean longitudes :math:`\lambda_i` with new momenta
        :math:`\delta\Lambda_i = \Lambda_i - \Lambda_{i,0}`.
        """
        o2n = {a:a0+da for a,a0,da in zip(actions,actions_ref,delta_actions)}
        n2o = {da:(a-a0) for a,a0,da in zip(actions,actions_ref,delta_actions)}
        a2da= dict(zip(actions,delta_actions))
        qpnew = [v.subs(a2da) for v in old_qp_vars]
        qpold = old_qp_vars
        o2n_full={v:v.subs(o2n) for v in old_qp_vars}
        n2o_full={v:v.subs(n2o) for v in qpnew}
        return cls(qpold,qpnew,o2n_full,n2o_full,params=params)


    @classmethod
    def rescale_transformation(cls,qp_pairs, scale, cartesian_pairs = [], **kwargs):
        r"""
        Get a canonical transformation that simulatneously rescales the Hamiltonian
        and canonical momenta by a common factor. 

        Arguments
        ---------
        qp_pairs : list of 2-tuples
            Pairs of canonically conjugate variable symbols.
        scale : symbol or real
            Re-scaling factor. 
            The new momenta will be given by p' = scale * p
            and the new Hamiltonian will be H' = scale * H.
        cartesian pairs : list
            List of indices of Cartesian-style canonical pairs.
            These pairs will be recscaled such that 
            (y',x') = sqrt(scale) * (y,x)
        """
        o2n = dict()
        n2o = dict()
        rtscale = sqrt(scale)
        qpvars = [q for q,p in qp_pairs] + [p for q,p in qp_pairs]
        for i,qp in enumerate(qp_pairs):
            q,p = qp
            if i in cartesian_pairs:
                o2n[p] = p / rtscale 
                o2n[q] = q / rtscale 
                n2o[p] = p * rtscale 
                n2o[q] = q * rtscale 
            else:
                o2n[p] = p / scale 
                n2o[p] = p * scale 
        return cls(qpvars,qpvars,o2n,n2o,H_scale = scale,**kwargs)

    @classmethod
    def Poincare_rescale_transformation(cls,pham,scale,**kwargs):
        N = pham.N
        cart_pairs = [3*i + 1 for i in range(N-1)] + [3*i + 2 for i in range(N-1)]
        return cls.rescale_transformation(pham.qp_pairs,scale,cart_pairs,**kwargs)

    @classmethod
    def composite(cls, transformations, old_to_new_simplify = None, new_to_old_simplify = None):
        r"""
        Create a canonical transformation as the composition of canonical
        transformations.

        Arguments
        ---------
        transformations : list of :class:`~celmech.canonical_transformations.CanonicalTransformation`
            List of transformations to compose. Transformations are composed in
            the order

            .. math::
                T_N \circ T_{N-1}\circ ...\circ T_1 \circ T_0

            when passed the list :math:`[T_0,T_1,...,]`.

        Returns
        -------
        .CanonicalTransformation
        """
        old_qp_vars = transformations[0].old_qp_vars
        new_qp_vars = transformations[-1].new_qp_vars
        scale = Mul(*[t.H_scale for t in transformations])
        params = dict()
        n2o = dict()
        for var in new_qp_vars:
            var_exprn = var
            for t in reversed(transformations):
                var_exprn = t.new_to_old(var_exprn)
            n2o[var] = var_exprn
        o2n = dict()
        for var in old_qp_vars:
            var_exprn = var
            for t in transformations:
                var_exprn = t.old_to_new(var_exprn)
            o2n[var] = var_exprn
        for t in transformations:
            params.update(t.params)
    
        return cls(old_qp_vars, new_qp_vars, o2n, n2o, old_to_new_simplify, new_to_old_simplify,H_scale=scale,params=params)
