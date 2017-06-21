from sympy import S, diff, lambdify, symbols, sqrt, cos, numbered_symbols, simplify
from scipy.integrate import ode
import numpy as np
import rebound

class Hamiltonian(object):
    def __init__(self, H, pqpairs, initial_conditions, params, Nparams):
        self.pqpairs = pqpairs
        self.params = params
        self.H = H
        self.derivs = {}
        for pqpair in pqpairs:
            p,q = pqpair
            self.derivs[p] = -diff(self.H, q)
            self.derivs[q] = diff(self.H, p)
        
        self._calculate_numerical_expressions(Nparams)
        def diffeq(t, y):
            dydt = [deriv(*y) for deriv in self.Nderivs]
            #print(t, y, dydt)
            return dydt
        self.integrator = ode(diffeq).set_integrator('lsoda')
        self.integrator.set_initial_value(initial_conditions, 0)
        self.state = initial_conditions
    def integrate(self, time):
        if time > self.integrator.t:
            self.integrator.integrate(time)
        self._assign_variables()
    def _assign_variables(self):
        pass
    def _calculate_numerical_expressions(self, Nparams):
        self.NH = self.H
        for i, param in enumerate(self.params):
            try:
                self.NH = self.NH.subs(param, Nparams[i])
            except KeyError:
                raise AttributeError("Need to pass keyword {0} to Hamiltonian.integrate".format(param))
        symvars = [item for pqpair in self.pqpairs for item in pqpair]
        self.Nderivs = []
        for pqpair in self.pqpairs:
            p,q = pqpair
            self.Nderivs.append(lambdify(symvars, -diff(self.NH, q), 'numpy'))
            self.Nderivs.append(lambdify(symvars, diff(self.NH, p), 'numpy'))
def jacobi_masses_from_sim(sim, particles):
    print(sim.particles[0]._sim.contents.G)
    ps = sim.particles
    interior_mass = ps[0].m
    for i in range(1,sim.N):
        particles[i].mjac = ps[i].m*interior_mass/(ps[i].m+interior_mass) # reduced mass with interior mass
        particles[i].Mjac = ps[0].m*(ps[i].m+interior_mass)/interior_mass # mjac[i]*Mjac[i] always = ps[i].m*ps[0].m
        particles[i].mu = sim.G**2*particles[i].Mjac**2*particles[i].mjac**3 # Deck (2013) notation


class Particle(object): pass

class HamiltonianPoincare(Hamiltonian):
    def __init__(self, Lambdas, lambdas, Gammas, gammas):
        pass
    def initialize_from_sim(self, sim):
        self.particles = [Particle() for i in range(sim.N)]
        #makeactionanglepairs()
        #makeparams()
        masses = jacobi_masses_from_sim(sim, self.particles)
        #getNparams()
        #getic()
        self.H = ...
        #super(HamiltonianPoincare, self).__init__(H, pqpairs, initial_conditions, params, Nparams)

        for p in sim.particles[1:]:
            self.particles[-1].a = p.a
            self.particles[-1].e = p.e
