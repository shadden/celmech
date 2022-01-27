import numpy as np
from .miscellaneous import PoissonBracket
from sympy import lambdify, solve, diff,sin,cos,sqrt,atan2,symbols,Matrix, Mul,S
from sympy import trigsimp, simplify, postorder_traversal, powsimp
from sympy import Add
from sympy.simplify.fu import TR10,TR10i
from .hamiltonian import reduce_hamiltonian, PhaseSpaceState, Hamiltonian
from . import Poincare

def _termwise_trigsimp(exprn):
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
        symbols("Q(1:{})".format(N+1)),
        symbols("P(1:{})".format(N+1))
    ))
    return default_qp_symbols

def _get_default_xy_symbols(N):
    default_xy_symbols=list(zip(
        symbols("x(1:{})".format(N+1)),
        symbols("y(1:{})".format(N+1))
    ))
    return default_xy_symbols

class CanonicalTransformation():
    def __init__(self,
            old_qpvars,
            new_qpvars,
            old_to_new_rule = None,
            new_to_old_rule = None,
            old_to_new_simplify = None,
            new_to_old_simplify = None,
            params = {},
            **kwargs
    ):
        self.old_qpvars = old_qpvars
        self.new_qpvars = new_qpvars
        self.params = params
        self.Hscale = kwargs.get('Hscale',1)

        if not (old_to_new_rule or new_to_old_rule):
            raise ValueError("Must specify at least one of 'old_to_new_rule' or 'new_to_old_rule'!")
        if old_to_new_rule:
            self.old_to_new_rule = old_to_new_rule
        else:
            self.new_to_old_rule = new_to_old_rule
            eqs = [v - self.new_to_old_rule[v] for v in self.new_qpvars]
            self.old_to_new_rule = solve(eqs,self.old_qpvars)
        
        if new_to_old_rule:
            self.new_to_old_rule = new_to_old_rule
        else:
            eqs = [v - self.old_to_new_rule[v] for v in self.old_qpvars] 
            self.new_to_old_rule = solve(eqs,self.new_qpvars)

        if old_to_new_simplify:
            self.old_to_new_simplify_function = old_to_new_simplify
        else:
            self.old_to_new_simplify_function = lambda x: x

        if new_to_old_simplify:
            self.new_to_old_simplify_function = new_to_old_simplify
        else:
            self.new_to_old_simplify_function = lambda x: x

        qpv_new = self.new_qpvars
        qpv_old = self.old_qpvars
        self._old_to_new_vfunc = lambdify(qpv_old,[self.Nnew_to_old(v) for v in qpv_new])
        self._new_to_old_vfunc = lambdify(qpv_new,[self.Nold_to_new(v) for v in qpv_old])
   
    def old_to_new(self,exprn):
        exprn = exprn.subs(self.old_to_new_rule)
        exprn = self.old_to_new_simplify_function(exprn)
        return exprn
    def Nold_to_new(self,exprn):
        exprn = self.old_to_new(exprn)
        Nexprn = exprn.subs(self.params)
        return Nexprn

    def new_to_old(self,exprn):
        exprn = exprn.subs(self.new_to_old_rule)
        exprn = self.new_to_old_simplify_function(exprn)
        return exprn
    def Nnew_to_old(self,exprn):
        exprn = self.new_to_old(exprn)
        Nexprn = exprn.subs(self.params)
        return Nexprn


    def old_to_new_array(self,arr):
        return np.array(self._old_to_new_vfunc(*arr))

    def new_to_old_array(self,arr):
        return np.array(self._new_to_old_vfunc(*arr))


    def new_to_old_state(self,state):
        old_values = self.new_to_old_array(state.values)
        old_state = PhaseSpaceState(self.old_qpvars,old_values,t=state.t)
        return old_state
    
    def old_to_new_hamiltonian(self,ham,do_reduction = False):
        # Get full values
        new_values = self.old_to_new_array(ham.full_values)
        # The full set of q,p variables
        old_full = set(ham.full_qpvars)
        # Only q,p variables that are active
        old_active = set(ham.qpvars)
        # Ignorable q,p variables
        old_ignorable = old_full.difference(old_active)

        # New, transformed Hamiltonian expression
        newH =  self.old_to_new(ham.H)
        
        new_pars = ham.Hparams.copy()
        new_pars.update(self.params)

        # Handle possible cases where ignorable coordinates get transformed:
        #   First, remove old ingorable coordinates from the new Hparams dictionary
        for ig in old_ignorable:
            new_pars.pop(ig,None)
        #   Next, loop through new variable pairs and detect ingorable ones.
        #       'is_active' stores a boolean array indicating which new q,p vars 
        #       will be active variables.
        is_active = np.ones(2*self.Ndof,dtype=bool)
        for i,qpnew in enumerate(self.new_qppairs):
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
                is_active[i+self.Ndof] = False
                #if qnew in newH.free_symbols:
                new_pars[qnew] = new_values[i]
                #if pnew in newH.free_symbols:
                new_pars[pnew] = new_values[i+self.Ndof]

        # The list of new active q,p variables.
        new_vars_reduced = [var for var,activeQ in zip(self.new_qpvars,is_active) if activeQ]

        # New state consisting only of active variables.
        new_state = PhaseSpaceState(new_vars_reduced,new_values[is_active],ham.state.t)
        

        # Rescale Hamiltonian.
        # Warning--- this will not work properly if the top-level function is not sympy.Add
        # This should probably be treated more carefully.
        if self.Hscale != 1:
            assert newH.func == Add,"Re-scaling requires the top level node of the Hamiltonian expression tree must be sympy.Add"
        newH = newH.func(*[a*self.Hscale for a in newH.args])

        # New Hamiltonian: the new state only contains active variables
        # but full_qpvars kwarg is passed the full set of new q,p variables.
        new_ham = Hamiltonian(newH,new_pars,new_state,full_qpvars = self.new_qpvars)
        if do_reduction:
            new_ham = reduce_hamiltonian(new_ham)
        return new_ham
    
    def new_to_old_hamiltonian(self,ham,do_reduction = False):
        old_state = self.new_to_old_state(ham.state)
        oldH = self.new_to_old(ham.H) / self.Hscale
        old_ham = Hamiltonian(oldH,ham.Hparams,old_state)
        if do_reduction:
            old_ham = reduce_hamiltonian(old_ham)
        return old_ham
    
    def _test_new_to_old_canonical(self):
        pb = lambda q,p: PoissonBracket(q,p,self.old_qpvars,[]) / self.Hscale
        return [pb(self.new_to_old(qq),self.new_to_old(pp)).simplify() for qq,pp in self.new_qppairs]

    def _test_old_to_new_canonical(self):
        pb = lambda q,p: PoissonBracket(q,p,self.new_qpvars,[]) * self.Hscale
        return [pb(self.old_to_new(qq),self.old_to_new(pp)).simplify() for qq,pp in self.old_qppairs]
   
    @property 
    def Ndof(self):
        return int(len(self.old_qpvars)/2)

    @property
    def old_qppairs(self):
        return [(self.old_qpvars[i], self.old_qpvars[i+self.Ndof]) for i in range(self.Ndof)]
    
    @property
    def new_qppairs(self):
        return [(self.new_qpvars[i], self.new_qpvars[i+self.Ndof]) for i in range(self.Ndof)]

    @classmethod
    def from_type2_generating_function(cls,F2func,old_qpvars,new_qpvars,**kwargs):
        Ndof = int(len(old_qpvars)/2)
        old_qppairs = [(old_qpvars[i], old_qpvars[i+Ndof]) for i in range(Ndof)]
        new_qppairs = [(new_qpvars[i], new_qpvars[i+Ndof]) for i in range(Ndof)]
        eqs = [p - diff(F2func,q) for q,p in old_qppairs]
        eqs += [Q - diff(F2func,P) for Q,P in new_qppairs]
        o2n = solve(eqs,old_qpvars)
        n2o = solve(eqs,new_qpvars)
        return cls(old_qpvars,new_qpvars,old_to_new_rule=o2n,new_to_old_rule = n2o)
    @classmethod
    def CartesianToPolar(cls,old_qpvars,indices,**kwargs):
        Ndof = int(len(old_qpvars)/2)
        old_qppairs = [(old_qpvars[i], old_qpvars[i+Ndof]) for i in range(Ndof)]
        N = len(indices)
        newQPs = kwargs.get("polar_symbols",_get_default_qp_symbols(N))
        new_q, new_p = [], []
        o2n,n2o = dict(),dict()
        for i,qp in enumerate(old_qppairs):
            if i in indices:
                y,x = qp 
                Q,P = newQPs.pop(0)
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
        new_qpvars = new_q + new_p
        return cls(old_qpvars,new_qpvars,o2n,n2o,o2n_simplify,**kwargs)
    @classmethod
    def PolarToCartesian(cls,old_qpvars,indices,**kwargs):
        Ndof = int(len(old_qpvars)/2)
        old_qppairs = [(old_qpvars[i], old_qpvars[i+Ndof]) for i in range(Ndof)]
        N = len(indices)
        newXYs = kwargs.get("cartesian_symbols",_get_default_xy_symbols(N))
        new_q, new_p = [], []
        o2n,n2o = dict(),dict()
        for i,qp in enumerate(old_qppairs):
            if i in indices:
                x,y = newXYs.pop(0)
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
        atan2_simplify = lambda exprn: TR10i(simplify(TR10(exprn))) if _exprn_contains_funcs(exprn,[atan2]) else powsimp(exprn)
        o2n_simplify = lambda x: x.func(*[atan2_simplify(a) for a in x.args]) if len(x.args) > 0 else x
        kwargs.setdefault("old_to_new_simplify",o2n_simplify)
        new_qpvars = new_q + new_p
        return cls(old_qpvars,new_qpvars,o2n,n2o,**kwargs)

    @classmethod
    def FromLinearAngleTransformation(cls,old_qpvars,Tmtrx,old_cartesian_indices=[],new_cartesian_indices=[],**kwargs):
        try:
            Ndof = len(old_qpvars)//2
            Tmtrx = np.array(Tmtrx).reshape(Ndof,Ndof)
        except:
            raise ValueError("'Tmtrx' could not be shaped into a {0}x{0} array".format(Ndof))
        old_qp_pairs = list(zip(old_qpvars[:Ndof],old_qpvars[Ndof:]))
        new_qp_pairs = kwargs.get("QPvars",_get_default_qpxy_symbols(Ndof,new_cartesian_indices))

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
        for i,qp_new,ang_exprn,act_exprn in zip(range(Ndof),new_qp_pairs,Tmtrx * old_angvars,Tmtrx_inv.T * old_actvars):
            qnew,pnew = qp_new
            if i in new_cartesian_indices:
                n2o[qnew] = simplify(sqrt(2*act_exprn) * TR10i(TR10(sin(ang_exprn))))
                n2o[pnew] = simplify(sqrt(2*act_exprn) * TR10i(TR10(cos(ang_exprn))))
            else:
                n2o[qnew] = ang_exprn
                n2o[pnew] = act_exprn
        
        o2n = dict()
        for i,qp_old,ang_exprn,act_exprn in zip(range(Ndof),old_qp_pairs,Tmtrx_inv * new_angvars,Tmtrx.T * new_actvars):
            qold,pold = qp_old
            if i in old_cartesian_indices:
                o2n[qold] = simplify(sqrt(2*act_exprn) * TR10i(TR10(sin(ang_exprn))))
                o2n[pold] = simplify(sqrt(2*act_exprn) * TR10i(TR10(cos(ang_exprn))))
            else:
                o2n[qold] = ang_exprn
                o2n[pold] = act_exprn
        
        new_ang_vars = set([new_qp_pairs[i][0] for i in range(Ndof) if i not in new_cartesian_indices])
        simplify_trig = lambda a: simplify(TR10i(a.expand())) if len(new_ang_vars.intersection(a.free_symbols))>0 else a
        o2n_simplify = lambda x: x.func(*[simplify_trig(arg) for arg in x.args]) if len(x.args)>0 else x
        kwargs.setdefault("old_to_new_simplify",o2n_simplify)
        return cls(old_qpvars,list(new_coords) + list(new_momenta), o2n, n2o,**kwargs)

    @classmethod
    def from_poincare_angles_matrix(cls,pvars,Tmtrx,**kwargs):
        if not type(pvars) == Poincare:
            raise TypeError("'pvars' must be of type 'Poincare'")
        try:
            Ndof = pvars.Ndof
            Tmtrx = np.array(Tmtrx).reshape(Ndof,Ndof)
        except:
            raise ValueError("'Tmtrx' could not be shaped into a {0}x{0} array".format(Ndof))
        Tmtrx = Matrix(Tmtrx)
        QPvars = kwargs.get("QPvars",_get_default_qp_symbols(Ndof))
        Npl = pvars.N - 1
        old_varslist = pvars.qpvars
        old_angvars = [old_varslist[3*i + 0] for i in range(Npl)] # lambdas
        old_angvars += [atan2(old_varslist[3*i + 1],old_varslist[3*(i+Npl) + 1]) for i in range(Npl)] # gammas
        old_angvars += [atan2(old_varslist[3*i + 2],old_varslist[3*(i+Npl) + 2]) for i in range(Npl)] # qs
        old_actvars = [old_varslist[3*(i+Npl) + 0] for i in range(Npl)] # Lambdas
        old_actvars += [(old_varslist[3*i + 1]**2 + old_varslist[3*(i+Npl) + 1]**2) / 2 for i in range(Npl)] # Gammas
        old_actvars += [(old_varslist[3*i + 2]**2 + old_varslist[3*(i+Npl) + 2]**2) / 2 for i in range(Npl)] # Qs

        n2o = dict(zip([Q for Q,P in QPvars],Tmtrx * Matrix(old_angvars)))
        n2o.update(dict(zip([P for Q,P in QPvars],Tmtrx.inv().transpose() * Matrix(old_actvars))))
        
        o2n=dict()
        old2new_angvars = Tmtrx.inv() * Matrix([Q for Q,P in QPvars])
        old2new_actvars = Tmtrx.transpose() * Matrix([P for Q,P in QPvars])
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
       
        new_qpvars = [Q for Q,P in QPvars] + [P for Q,P in QPvars]
        return cls(pvars.qpvars,new_qpvars,o2n,n2o,old_to_new_simplify=_termwise_trigsimp)

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
        qpnew = [v.subs(L2dL) for v in pham.qpvars]
        qpold = pham.qpvars
        o2n_full={v:v.subs(o2n) for v in pham.qpvars}
        n2o_full={v:v.subs(n2o) for v in qpnew}
        params = {L0:pham.Hparams[L0] for L0 in Lambda0s}
        return cls(qpold,qpnew,o2n_full,n2o_full,params = params)

    @classmethod
    def RescaleTransformation(cls,qppairs, scale, cartesian_pairs = [], **kwargs):
        r"""
        Get a canonical transformation that simulatneously rescales the Hamiltonian
        and canonical momenta by a common factor. 

        Arguments
        ---------
        qppairs : list of 2-tuples
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
        qpvars = [q for q,p in qppairs] + [p for q,p in qppairs]
        for i,qp in enumerate(qppairs):
            q,p = qp
            if i in cartesian_pairs:
                o2n[p] = p / rtscale 
                o2n[q] = q / rtscale 
                n2o[p] = p * rtscale 
                n2o[q] = q * rtscale 
            else:
                o2n[p] = p / scale 
                n2o[p] = p * scale 
        return cls(qpvars,qpvars,o2n,n2o,Hscale = scale,**kwargs)

    @classmethod
    def PoincareRescaleTransformation(cls,pham,scale,**kwargs):
        N = pham.N
        cart_pairs = [3*i + 1 for i in range(N-1)] + [3*i + 2 for i in range(N-1)]
        return cls.RescaleTransformation(pham,scale,cart_pairs,**kwargs)

    @classmethod
    def Composite(cls, transformations, old_to_new_simplify = None, new_to_old_simplify = None):
        old_qpvars = transformations[0].old_qpvars
        new_qpvars = transformations[-1].new_qpvars
        scale = Mul(*[t.Hscale for t in transformations])
        params = dict()
        n2o = dict()
        for var in new_qpvars:
            var_exprn = var
            for t in reversed(transformations):
                var_exprn = t.new_to_old(var_exprn)
            n2o[var] = var_exprn
        o2n = dict()
        for var in old_qpvars:
            var_exprn = var
            for t in transformations:
                var_exprn = t.old_to_new(var_exprn)
            o2n[var] = var_exprn
        for t in transformations:
            params.update(t.params)
    
        return cls(old_qpvars, new_qpvars, o2n, n2o, old_to_new_simplify, new_to_old_simplify,Hscale=scale,params=params)
