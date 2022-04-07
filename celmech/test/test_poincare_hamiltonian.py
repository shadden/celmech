import rebound
import unittest
import math
import numpy as np
from celmech import Poincare, PoincareHamiltonian, PoincareParticle
from random import random, seed

def get_sim_sec():
    sim = rebound.Simulation()
    sim.add(m=1)
    sim.G = 4*np.pi*np.pi
    mass=1e-4
    sim.add(m=mass,P=1, e=0.01)
    sim.add(m=mass,P=3.1,e=0.06,pomega=np.pi)
    sim.add(m=mass,P=10.1,e=0.02,pomega=0.5*np.pi)
    sim.dt = sim.particles[1].P/15
    sim.integrator='whfast'
    sim.move_to_com()
    return sim 

def get_sim_res():
    sim = rebound.Simulation()
    sim.add(m=1)
    sim.G = 4*np.pi*np.pi
    mass=1e-4
    sim.add(m=mass,P=1, e=0.01)
    sim.add(m=mass,P=3/2,e=0.03,pomega=np.pi)
    sim.add(m=mass,P=10.1,e=0.02,pomega=0.5*np.pi)
    sim.dt = sim.particles[1].P/15.
    sim.integrator='whfast'
    sim.move_to_com()
    return sim

class TestPoincareHamiltonian(unittest.TestCase):
    # test accuracy compared to Nbody
    def test_secular(self):
        sim = get_sim_sec()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        for i1, i2 in [(1,2),(2,3),(1,3)]:
            Hp.add_secular_terms(max_order=2, indexIn=i1, indexOut=i2)
        tmax = 1e0# 3e5
        sim.integrate(tmax)
        Hp.integrate(tmax)
        self.assertAlmostEqual(sim.particles[1].e, Hp.particles[1].e, delta=1e-2)
    
    def test_res(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        for i1, i2 in [(1,2),(2,3),(1,3)]:
            Hp.add_secular_terms(max_order=2, indexIn=i1, indexOut=i2)
        Hp.add_MMR_terms(p=3, q=1, max_order=2, indexIn=1, indexOut=2)
        tmax = 2#200
        sim.integrate(tmax)
        Hp.integrate(tmax)
        self.assertAlmostEqual(sim.particles[1].e, Hp.particles[1].e, delta=1e-2)

    def test_repeat_terms(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        Hp2 = PoincareHamiltonian(pvars)
        Hp.add_cosine_term((3,-2,-1,0,0,0),max_order=2)
        Hp2.add_cosine_term((3,-2,-1,0,0,0),max_order=1)
        Hp2.add_cosine_term((3,-2,-1,0,0,0),max_order=2)

    def test_add_cos_vs_MMR(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        Hp2 = PoincareHamiltonian(pvars)
        Hp.add_MMR_terms(p=3,q=1,max_order=1)
        Hp2.add_cosine_term((3,-2,-1,0,0,0))
        Hp2.add_cosine_term((3,-2,0,-1,0,0))
        self.assertEqual(Hp.H, Hp2.H)

    def test_add_cos_vs_MMR_high_order_ecc(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        Hp2 = PoincareHamiltonian(pvars)
        Hp.add_MMR_terms(p=3,q=1,max_order=3,inclinations=False) 
        Hp2.add_cosine_term((3,-2,-1,0,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((3,-2,0,-1,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((3,-2,-2,1,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((3,-2,1,-2,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((6,-4,-2,0,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((6,-4,-1,-1,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((6,-4,0,-2,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((9,-6,-3,0,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((9,-6,-2,-1,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((9,-6,-1,-2,0,0),max_order=3,inclinations=False)
        Hp2.add_cosine_term((9,-6,0,-3,0,0),max_order=3,inclinations=False)
        self.assertEqual(Hp.H, Hp2.H)

    def test_add_cos_vs_MMR_high_order_inc(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        Hp2 = PoincareHamiltonian(pvars)
        Hp.add_MMR_terms(p=3,q=1,max_order=3,eccentricities=False) 
        Hp2.add_cosine_term((6,-4,0,0,-2,0),max_order=3,eccentricities=False)
        Hp2.add_cosine_term((6,-4,0,0,-1,-1),max_order=3,eccentricities=False)
        Hp2.add_cosine_term((6,-4,0,0,0,-2),max_order=3,eccentricities=False)
        self.assertEqual(Hp.H, Hp2.H)
    
    def test_add_higher_order_than_max_order(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        with self.assertRaises(AttributeError):
            Hp.add_cosine_term((3,-2,0,-3,2,0),max_order=3) # kvec has order 5, max_order=3

    def test_add_cos_vs_MMR_high_order(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        Hp2 = PoincareHamiltonian(pvars)
        Hp.add_MMR_terms(p=3,q=1,max_order=3)
        Hp2.add_cosine_term((3,-2,-1,0,0,0),max_order=3)
        Hp2.add_cosine_term((3,-2,0,-1,0,0),max_order=3)
        Hp2.add_cosine_term((3,-2,0,-1,1,-1),max_order=3)
        Hp2.add_cosine_term((3,-2,0,-1,-1,1),max_order=3)
        Hp2.add_cosine_term((3,-2,-1,0,1,-1),max_order=3)
        Hp2.add_cosine_term((3,-2,-1,0,-1,1),max_order=3)
        Hp2.add_cosine_term((3,-2,-2,1,0,0),max_order=3)
        Hp2.add_cosine_term((3,-2,1,-2,0,0),max_order=3)
        Hp2.add_cosine_term((3,-2,0,1,-2,0),max_order=3)
        Hp2.add_cosine_term((3,-2,0,1,0,-2),max_order=3)
        Hp2.add_cosine_term((3,-2,0,1,-1,-1),max_order=3)
        Hp2.add_cosine_term((3,-2,1,0,-2,0),max_order=3)
        Hp2.add_cosine_term((3,-2,1,0,0,-2),max_order=3)
        Hp2.add_cosine_term((3,-2,1,0,-1,-1),max_order=3)
        Hp2.add_cosine_term((6,-4,-2,0,0,0),max_order=3)
        Hp2.add_cosine_term((6,-4,-1,-1,0,0),max_order=3)
        Hp2.add_cosine_term((6,-4,0,-2,0,0),max_order=3)
        Hp2.add_cosine_term((6,-4,0,0,-2,0),max_order=3)
        Hp2.add_cosine_term((6,-4,0,0,-1,-1),max_order=3)
        Hp2.add_cosine_term((6,-4,0,0,0,-2),max_order=3)
        Hp2.add_cosine_term((9,-6,-3,0,0,0),max_order=3)
        Hp2.add_cosine_term((9,-6,0,-3,0,0),max_order=3)
        Hp2.add_cosine_term((9,-6,0,-1,-2,0),max_order=3)
        Hp2.add_cosine_term((9,-6,0,-1,-1,-1),max_order=3)
        Hp2.add_cosine_term((9,-6,0,-1,0,-2),max_order=3)
        Hp2.add_cosine_term((9,-6,-1,0,-2,0),max_order=3)
        Hp2.add_cosine_term((9,-6,-1,0,-1,-1),max_order=3)
        Hp2.add_cosine_term((9,-6,-1,0,0,-2),max_order=3)
        Hp2.add_cosine_term((9,-6,-2,-1,0,0),max_order=3)
        Hp2.add_cosine_term((9,-6,-1,-2,0,0),max_order=3)
        self.assertEqual(Hp.H, Hp2.H)

    def test_add_double_tuple_list(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        Hp.add_secular_terms(max_order=2, eccentricities=False)
        kvec = tuple([0 for _ in range(6)])
        Hp.add_cosine_term(kvec,max_order=2,eccentricities=False)

        # But this gets through because kvec is a list ---
        kvec = [0 for _ in range(6)]
        Hp.add_cosine_term(kvec,max_order=2,eccentricities=False)
        
        Hp.add_cosine_term(kvec, nu_vecs=[[0, 1, 0, 0]], eccentricities=False) 
        Hp.add_cosine_term(kvec, nu_vecs=[(0, 1, 0, 0)], eccentricities=False) 
        Hp.add_cosine_term(kvec, nu_vecs=[(0, 1, 0, 0)], l_vecs=[[0,0]], eccentricities=False) 
        Hp.add_cosine_term(kvec, nu_vecs=[(0, 1, 0, 0)], l_vecs=[(0,0)], eccentricities=False) 
        self.assertEqual(len(Hp.resonance_indices), 4)

    def test_negative_kvec(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        kvec = (0,0,0,0,-2,2)
        Hp.add_cosine_term(kvec,nu_vecs=[(0,0,0,0)],eccentricities=False)

        kvec = (0,0,0,0,2,-2)
        Hp.add_cosine_term(kvec,nu_vecs=[(0,0,0,0)],eccentricities=False)
        self.assertEqual(len(Hp.resonance_indices), 1)
    
    def test_invalid_inc_kvec(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        kvecs = [(3,-2,0,0,0,-1),(3,-2,0,0,-2,1),(1,0,0,0,0,-1)]
        for kvec in kvecs:
            with self.assertRaises(AttributeError):
                Hp.add_cosine_term(kvec,nu_vecs=[(0,0,0,0)])
                
    def test_invalid_dalembert_kvec(self):
        sim = get_sim_res()
        pvars = Poincare.from_Simulation(sim)
        Hp = PoincareHamiltonian(pvars)
        kvecs = [(3,-2,0,0,0,0),(0,0,-2,1,0,0),(0,0,0,0,0,-1)]
        for kvec in kvecs:
            with self.assertRaises(AttributeError):
                Hp.add_cosine_term(kvec,nu_vecs=[(0,0,0,0)])
                
if __name__ == '__main__':
    unittest.main()

