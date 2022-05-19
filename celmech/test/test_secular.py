import unittest
from celmech.disturbing_function import list_secular_terms
import rebound as rb
import numpy as np
from celmech.nbody_simulation_utilities import align_simulation
from celmech.secular import SecularSystemSimulation
from celmech import Poincare,PoincareHamiltonian
from sympy import S 

def get_sim(scale= 0.05,Nplanet = 3):
    sim = rb.Simulation()
    sim.add(m=1)
    sim.add(m=1e-5 , a = 2, e=0.01, inc=0.01, l=0.1, pomega=0.1, Omega=0.1)
    sim.add(m=2e-5 , a = 4, e=0.02, inc=0.02, l=0.2, pomega=0.2, Omega=0.2)
    sim.add(m=3e-5 , a = 8, e=0.03, inc=0.03, l=0.3, pomega=0.3, Omega=0.3)
    sim.move_to_com()
    align_simulation(sim)
    return sim

class TestSecular(unittest.TestCase):
    def setUp(self):
        self.sim = get_sim()
        self.pvars = Poincare.from_Simulation(self.sim)
        self.pham = PoincareHamiltonian(self.pvars)
        self.terms_list = list_secular_terms(min_order=4, max_order=4)#SecularTermsList(4,4)
        for i in range(1,self.pvars.N):
            for j in range(i+1,self.pvars.N):
                for term in self.terms_list:
                    k,nu = term
                    self.pham.add_cosine_term(k_vec=k,nu_vecs=[nu],indexIn=i,indexOut=j)
        self.pham._update()
        qpsymbols = [S('eta{}'.format(i)) for i in range(1,self.pvars.N)] +\
        [S('rho{}'.format(i)) for i in range(1,self.pvars.N)] +\
        [S('kappa{}'.format(i)) for i in range(1,self.pvars.N)] +\
        [S('sigma{}'.format(i)) for i in range(1,self.pvars.N)]  
        self.secular_variable_indices = [ self.pham.qp_vars.index(s) for s in qpsymbols ]
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
