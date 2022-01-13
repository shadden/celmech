import numpy as np
from .miscellaneous import PoissonBracket
from sympy import lambdify, solve, diff,sin,cos,sqrt,atan2,symbols,Matrix
from sympy import trigsimp, simplify
from .hamiltonian import reduce_hamiltonian, PhaseSpaceState, Hamiltonian
from . import Poincare

def _termwise_trigsimp(exprn):
    return exprn.func(*[trigsimp(a) for a in exprn.args])

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
            **kwargs
    ):
        self.old_qpvars = old_qpvars
        self.new_qpvars = new_qpvars
        
        if not (old_to_new_rule or new_to_old_rule):
            raise ValueError("Must specify at least one of 'old_to_new_rule' or 'new_to_old_rule'!")
        if old_to_new_rule:
            self.old_to_new_rule = old_to_new_rule
        else:
            eqs = [v - self.new_to_old_rule[v] for v in self.new_qpvars]
            self.old_to_new_rule = solve(eqs,self.old_qpvars)
        
        if new_to_old_rule:
            self.new_to_old_rule = new_to_old_rule
        else:
            eqs = [v - self.old_to_new_rule[v] for v in self.old_qpvars] 
            self.new_to_old_rule = solve(eqs,self.new_qpvars)
        
        qpv_new = self.new_qpvars
        qpv_old = self.old_qpvars
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
        exprn = exprn.subs(self.old_to_new_rule)
        exprn = self.old_to_new_simplify_function(exprn)
        return exprn

    def new_to_old(self,exprn):
        exprn = exprn.subs(self.new_to_old_rule)
        exprn = self.new_to_old_simplify_function(exprn)
        return exprn

    def old_to_new_array(self,arr):
        return np.array(self._old_to_new_vfunc(*arr))

    def new_to_old_array(self,arr):
        return np.array(self._new_to_old_vfunc(*arr))

    def old_to_new_state(self,state):
        new_values = self.old_to_new_array(state.values)
        new_state = PhaseSpaceState(self.new_qpvars,new_values,t=state.t)
        return new_state

    def new_to_old_state(self,state):
        old_values = self.new_to_old_array(state.values)
        old_state = PhaseSpaceState(self.old_qpvars,old_values,t=state.t)
        return old_state
    
    def old_to_new_hamiltonian(self,ham,do_reduction = False):
        new_state = self.old_to_new_state(ham.state)
        newH = self.old_to_new(ham.H)
        new_ham = Hamiltonian(newH,ham.Hparams,new_state)
        if do_reduction:
            new_ham = reduce_hamiltonian(new_ham)
        return new_ham
    
    def new_to_old_hamiltonian(self,ham,do_reduction = False):
        old_state = self.new_to_old_state(ham.state)
        oldH = self.new_to_old(ham.H)
        old_ham = Hamiltonian(oldH,ham.Hparams,old_state)
        if do_reduction:
            old_ham = reduce_hamiltonian(old_ham)
        return old_ham
    
    def _test_new_to_old_canonical(self):
        pb = lambda q,p: PoissonBracket(q,p,self.old_qpvars,[])
        return [pb(self.new_to_old_rule[qq],self.new_to_old_rule[pp]).simplify() for qq,pp in self.new_qppairs]

    def _test_old_to_new_canonical(self):
        pb = lambda q,p: PoissonBracket(q,p,self.new_qpvars,[])
        return [pb(self.old_to_new_rule[qq],self.old_to_new_rule[pp]).simplify() for qq,pp in self.old_qppairs]
   
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
        return cls(old_qpvars,new_qpvars,old_to_new_rule=o2n,**kwargs) 
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
        o2n_simplify = lambda x: x.func(*[simplify(trigsimp(a)) for a in x.args])
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

        # add a simplify function to remove arctans
        o2n_simplify = lambda x: x.func(*[simplify(trigsimp(a)) for a in x.args])
        new_qpvars = new_q + new_p
        return cls(old_qpvars,new_qpvars,o2n,n2o,o2n_simplify,**kwargs)

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
        QPvars = kwargs.get("PQvars",_get_default_qp_symbols(Ndof))
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
    def Composite(cls, transformations, old_to_new_simplify = None, new_to_old_simplify = None):
        old_qpvars = transformations[0].old_qpvars
        new_qpvars = transformations[-1].new_qpvars

        o2n = transformations[0].old_to_new_rule.copy()
        for trans in transformations[1:]:
            for key, val in o2n.items():
                o2n[key] = val.subs(trans.old_to_new_rule)
        n2o = transformations[-1].new_to_old_rule.copy()
        for trans in transformations[-2::-1]:
            for key, val in n2o.items():
                n2o[key] = val.subs(trans.new_to_old_rule)
    
        return cls(old_qpvars, new_qpvars, o2n, n2o, old_to_new_simplify, new_to_old_simplify)
