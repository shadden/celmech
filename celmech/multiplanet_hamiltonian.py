import numpy as np
from celmech import Hamiltonian,sim_to_poincare_vars
import rebound

sim = multiplanetSystem()
sim.add(m=1.)
sim.add(m=1.e-5, a=3)
sim.add(m=1.e-4, a=4)

ps = sim.particles
ps[1].Lambda

def get_symbolic_resonance(j,k,):
	
class multiplanetSystem(rebound.Simulation):

	def add(self, *args, **kwargs):
		super(multiplanetSystem, self).add(args, kwargs)
		sim_to_poincare()
		self.particles[-1].Lambda = ...
	
	def add_resonance(planet1,planet2,j,k,l):

	def add_all_resonances(planet1,planet2,j,k):

	def integrate(self,time):
		
		Nvariables = do_integration()

		self.assign_variables(Nvariables)
			
			
			

class 2bodysingleresonance(rebound.Simulation):

	def add(self, *args, **kwargs):
		super(multiplanetSystem, self).add(args, kwargs)
		Nvariables = sim_to_myvars()
		self.assign_variables(Nvariables)
		self.particles[-1].Lambda = ...
	
	def add_resonance(planet1,planet2,j,k,l):

	def integrate(self,time):
		
		Nvariables = do_integration()

		self.assign_variables(Nvariables)
