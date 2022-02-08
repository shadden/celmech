import unittest
from celmech.disturbing_function import SecularTermsList
import rebound as rb
import numpy as np
from celmech.nbody_simulation_utilities import align_simulation
from celmech.secular import SecularSystemSimulation
from celmech import Poincare,PoincareHamiltonian
from sympy import S 

def get_sim(scale= 0.05,Nplanet = 3):
    sim = rb.Simulation()
    sim.add(m=1)
    for i in range(1,Nplanet+1):
        sim.add(m=i * 1e-5 , a = 2**i, 
                e = np.random.rayleigh(scale),
                inc = np.random.rayleigh(scale),
                l = 'uniform',
                pomega = 'uniform',
                Omega = 'uniform'
               )
    sim.move_to_com()
    align_simulation(sim)
    return sim

class TestSecular(unittest.TestCase):
    def setUp(self):
        self.sim = get_sim()
        self.pvars = Poincare.from_Simulation(self.sim)
        self.pham = PoincareHamiltonian(self.pvars)
        self.terms_list = SecularTermsList(4,4)
        for i in range(1,self.pvars.N):
            for j in range(i+1,self.pvars.N):
                for term in self.terms_list:
                    k,z = term
                    self.pham.add_cosine_term(k,z,indexIn=i,indexOut=j,update=False)
        self.pham._update()
        qpsymbols = [S('eta{}'.format(i)) for i in range(1,self.pvars.N)] +\
        [S('rho{}'.format(i)) for i in range(1,self.pvars.N)] +\
        [S('kappa{}'.format(i)) for i in range(1,self.pvars.N)] +\
        [S('sigma{}'.format(i)) for i in range(1,self.pvars.N)]  
        self.secular_variable_indices = [ self.pham.varsymbols.index(s) for s in qpsymbols ]
        self.secular_sim = SecularSystemSimulation(self.pvars,dtFraction = 1/50.,max_order = 4)

    def test_derivatives(self,delta=1.e-15):
        state_vec = self.pham.state_to_list(self.pvars)
        pham_nderivs=np.array([self.pham.Nderivs[i](*state_vec) for i in self.secular_variable_indices])
        qpvec = self.secular_sim.nonlinearSecOp.state_vec_to_qp_vec(state_vec)
        sec_df_nderivs = self.secular_sim.nonlinearSecOp.deriv_from_qp_vec(qpvec)
        for q1,q2 in zip(pham_nderivs,sec_df_nderivs):
            self.assertAlmostEqual(q1,q2,delta = delta)


if __name__=='__main__':
    unittest.main()
