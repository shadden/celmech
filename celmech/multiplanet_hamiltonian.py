import numpy as np
# from celmech import Hamiltonian,sim_to_poincare_vars
import rebound


    
class multiplanetPoincareSystem(rebound.Simulation):
	
	
	def add(self, *args, **kwargs):

		super(multiplanetPoincareSystem, self).add(*args, **kwargs)
		self.sim_to_myvars()

	def sim_to_myvars(self):

		ps = self.particles
		Nps = len(ps)
		Mjac = np.zeros(Nps)
		mujac = np.zeros(Nps)
	
		Mstar = ps[0].m
		Mint = Mstar

		for p in ps[1:]:
			mi = p.m
			p.mjac = mi * Mint / (mi + Mint)
			p.Mjac =  Mstar * ( mi / p.mjac )
			p.mujac=sim.G**2 * p.Mjac**2 * p.mjac**3
			Mint += mi
			p.Lambda = p.mjac*np.sqrt(sim.G*p.Mjac*p.a)
			p.Gamma  = p.Lambda*(1.-np.sqrt(1.-p.e**2))
			p.lam  = p.l
			p.gamma  = -p.pomega

	def add_resonance(planet1,planet2,j,k,l):
		pass
	def add_all_resonances(planet1,planet2,j,k):
		pass
	def integrate(self,time):
		Nvariables = do_integration()
		self.assign_variables(Nvariables)
if __name__=="__main__":
	
	sim = multiplanetPoincareSystem()
	sim.add(m=1.)
	sim.add(m=1.e-5, a=3)
	sim.add(m=1.e-4, a=4)

	ps = sim.particles
	print( ps[1].m)
	print( ps[1].a)
	print( ps[1].Gamma)

# class 2bodysingleresonance(rebound.Simulation):
# 
# 	def add(self, *args, **kwargs):
# 		super(multiplanetSystem, self).add(args, kwargs)
# 		Nvariables = sim_to_myvars()
# 		self.assign_variables(Nvariables)
# 		self.particles[-1].Lambda = ...
# 	
# 	def add_resonance(planet1,planet2,j,k,l):
# 
# 	def integrate(self,time):
# 		
# 		Nvariables = do_integration()
# 
# 		self.assign_variables(Nvariables)
