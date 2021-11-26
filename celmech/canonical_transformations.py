import numpy as np
from .miscellaneous import PoissonBracket
from sympy import lambdify, solve, diff,sin,cos,sqrt,atan2,symbols,Matrix
from sympy import trigsimp
from .hamiltonian import reduce_hamiltonian, PhaseSpaceState, Hamiltonian
from . import Poincare

def _termwise_trigsimp(exprn):
    return exprn.func(*[trigsimp(a) for a in exprn.args])

def _get_default_pq_symbols(N):
    default_pq_symbols=list(zip(
        symbols("P(1:{})".format(N+1)),
        symbols("Q(1:{})".format(N+1))
    ))
    return default_pq_symbols

def _get_default_xy_symbols(N):
    default_xy_symbols=list(zip(
        symbols("x(1:{})".format(N+1)),
        symbols("y(1:{})".format(N+1))
    ))
    return default_xy_symbols

class CanonicalTransformation():
    def __init__(self,
            old_pq_pairs,
            new_pq_pairs,
            old_to_new_rule = None,
            new_to_old_rule = None,
            old_to_new_simplify = None,
            new_to_old_simplify = None,
            **kwargs
    ):

        self.old_pq_pairs = old_pq_pairs
        self.old_qpvars_list = [q for p,q in old_pq_pairs] + [p for p,q in old_pq_pairs]

        self.new_pq_pairs = new_pq_pairs
        self.new_qpvars_list = [q for p,q in new_pq_pairs] + [p for p,q in new_pq_pairs]
        if not (old_to_new_rule or new_to_old_rule):
            raise ValueError("Must specify at least one of 'old_to_new_rule' or 'new_to_old_rule'!")
        if old_to_new_rule:
            self.old_to_new_rule = old_to_new_rule
        else:
            eqs = [v - self.new_to_old_rule[v] for v in self.new_qpvars_list]
            self.old_to_new_rule = solve(eqs,self.old_qpvars_list)
        
        if new_to_old_rule:
            self.new_to_old_rule = new_to_old_rule
        else:
            eqs = [v - self.old_to_new_rule[v] for v in self.old_qpvars_list] 
            self.new_to_old_rule = solve(eqs,self.new_qpvars_list)
        
        qpv_new = self.new_qpvars_list
        qpv_old = self.old_qpvars_list
        self._old_to_new_vfunc = lambdify(qpv_old,[self.new_to_old_rule[v] for v in qpv_new])
        self._new_to_old_vfunc = lambdify(qpv_new,[self.old_to_new_rule[v] for v in qpv_old])

        if old_to_new_simplify:
            self.old_to_new_simplify_function = old_to_new_simplify
        else:
            self.old_to_new_simplify_function = lambda x: x

        if new_to_old_simplify:
            self.new_to_old_simplify_function = new_to_old_simplify
        else:
            self.new_to_old_simplify_function = lambda x: x
   
    def old_to_new(self,exprn):
        new_exprn = exprn.subs(self.old_to_new_rule)
        new_exprn_s = self.old_to_new_simplify_function(new_exprn)
        return new_exprn_s

    def new_to_old(self,exprn):
        new_exprn = exprn.subs(self.new_to_old_rule)
        new_exprn_s = self.new_to_old_simplify_function(new_exprn)
        return new_exprn_s

    def old_to_new_array(self,arr):
        return np.array(self._old_to_new_vfunc(*arr))

    def new_to_old_array(self,arr):
        return np.array(self._new_to_old_vfunc(*arr))

    def old_to_new_state(self,state):
        new_values = self.old_to_new_array(state.values)
        new_state = PhaseSpaceState(self.new_pq_pairs,new_values,t = state.t)
        return new_state

    def old_to_new_hamiltonian(self,ham,do_reduction = True):
        new_state = self.old_to_new_state(ham.state)
        newH = self.old_to_new(ham.H)
        new_ham = Hamiltonian(newH,ham.Hparams,new_state)
        if do_reduction:
            new_ham = reduce_hamiltonian(new_ham)
        return new_ham
    
    def _test_new_to_old_canonical(self):
        pb = lambda q,p: PoissonBracket(q,p,self.old_qpvars_list,[])
        return [pb(self.new_to_old_rule[qq],self.new_to_old_rule[pp]).simplify() for pp,qq in self.new_pq_pairs]

    def _test_old_to_new_canonical(self):
        pb = lambda q,p: PoissonBracket(q,p,self.new_qpvars_list,[])
        return [pb(self.old_to_new_rule[qq],self.old_to_new_rule[pp]).simplify() for pp,qq in self.old_pq_pairs]

    @classmethod
    def from_type2_generating_function(cls,F2func,old_pq_pairs,new_pq_pairs,**kwargs):
        old_vars = [v for qp in old_pq_pairs for v in qp] 
        new_vars = [v for qp in new_pq_pairs for v in qp] 
        eqs = [p - diff(F2func,q) for p,q in old_pq_pairs]
        eqs += [Q - diff(F2func,P) for P,Q in new_pq_pairs]
        o2n = solve(eqs,old_vars)
        return cls(old_pq_pairs,new_pq_pairs,old_to_new_rule = o2n,**kwargs) 
    @classmethod
    def PolarCartesianTransformation(cls,old_pq_pairs,indices_to_polar,indices_to_cartesian,**kwargs):
        N2polar = len(indices_to_polar)
        N2cart = len(indices_to_cartesian)
        newPQs = kwargs.get("polar_symbols",_get_default_pq_symbols(N2polar))
        newXYs = kwargs.get("cartesian_symbols",_get_default_xy_symbols(N2cart))
        new_pq_pairs = []
        o2n,n2o = dict(),dict()
        for i,pq in enumerate(old_pq_pairs):
            if i in indices_to_polar:
                x,y = pq
                P,Q = newPQs.pop(0)
                o2n.update({ x:sqrt(2*P) * cos(Q) , y:sqrt(2*P) * sin(Q)})
                n2o.update({P:(x*x +y*y)/2, Q:atan2(y,x)})
                new_pq_pairs.append((P,Q))
            elif i in indices_to_cartesian:
                x,y = newXYs.pop(0)
                P,Q = pq
                n2o.update({ x:sqrt(2*P) * cos(Q) , y:sqrt(2*P) * sin(Q)})
                o2n.update({P:(x*x +y*y)/2, Q:atan2(y,x)})
                new_pq_pairs.append((x,y))
            else:
                id_tr = dict(zip(pq,pq))
                o2n.update(id_tr)
                n2o.update(id_tr)
                new_pq_pairs.append(pq)
        return cls(old_pq_pairs,new_pq_pairs,o2n,n2o,**kwargs)

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
        PQvars = kwargs.get("PQvars",_get_default_pq_symbols(Ndof))
        Npl = pvars.N - 1
        old_varslist = pvars.qpvars_list
        old_angvars = [old_varslist[3*i + 0] for i in range(Npl)] # lambdas
        old_angvars += [atan2(old_varslist[3*i + 1],old_varslist[3*(i+Npl) + 1]) for i in range(Npl)] # gammas
        old_angvars += [atan2(old_varslist[3*i + 2],old_varslist[3*(i+Npl) + 2]) for i in range(Npl)] # qs
        old_actvars = [old_varslist[3*(i+Npl) + 0] for i in range(Npl)] # Lambdas
        old_actvars += [(old_varslist[3*i + 1]**2 + old_varslist[3*(i+Npl) + 1]**2) / 2 for i in range(Npl)] # Gammas
        old_actvars += [(old_varslist[3*i + 2]**2 + old_varslist[3*(i+Npl) + 2]**2) / 2 for i in range(Npl)] # Qs

        n2o = dict(zip([Q for P,Q in PQvars],Tmtrx * Matrix(old_angvars)))
        n2o.update(dict(zip([P for P,Q in PQvars],Tmtrx.inv().transpose() * Matrix(old_actvars))))
        
        o2n=dict()
        old2new_angvars = Tmtrx.inv() * Matrix([Q for P,Q in PQvars])
        old2new_actvars = Tmtrx.transpose() * Matrix([P for P,Q in PQvars])
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
        
        return cls(pvars.pqpairs,PQvars,o2n,n2o, old_to_new_simplify = _termwise_trigsimp)
