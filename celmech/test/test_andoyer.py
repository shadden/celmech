import rebound
import unittest
import math
import numpy as np
from celmech import Andoyer

class TestAndoyer(unittest.TestCase):
    def setUp(self):
        self.sim = rebound.Simulation()
        self.sim.add(m=1.)
        self.sim.add(m=1.e-5, P=1.5, e=0.02, pomega=0.3, l=0.4)
        self.sim.add(m=1.e-7, P=2., e=0.01, pomega=2.3, l=4.4)
        self.sim.move_to_com()
        self.delta = 1.e-10

    def tearDown(self):
        self.sim = None

    def compare_objects(self, obj1, obj2, delta=1.e-15):
        self.assertEqual(type(obj1), type(obj2))
        for attr in [attr for attr in dir(obj1) if not attr.startswith('_')]:
            self.assertAlmostEqual(getattr(obj1, attr), getattr(obj2, attr), delta=delta)
    
    def compare_simulations(self, sim1, sim2, delta=1.e-15):
        equal = True
        for i in range(1,sim1.N):
            p1 = sim1.particles[i]
            p2 = sim2.particles[i]
            for attr in ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz']:
                self.assertAlmostEqual(getattr(p1, attr), getattr(p2, attr), delta=delta)
    
    def compare_andoyer(self, a1, a2, delta=1.e-15):
        self.assertEqual(type(a1), type(a2))
        for attr in ['X', 'Y', 'Psi2', 'psi2', 'Phiprime', 'K', 'deltalambda', 'lambda1']:
            self.assertAlmostEqual(getattr(a1, attr), getattr(a2, attr), delta=delta)
    
    def compare_poincare_particles(self, ps1, ps2, delta=1.e-15):
        self.assertEqual(type(ps1), type(ps2))
        ps1 = ps1[1:]
        ps2 = ps2[1:] # ignore the dummy particle for primary at index 0
        for p1, p2 in zip(ps1, ps2):
            for attr in ['X', 'Y', 'm', 'M', 'Lambda', 'l']:
                self.assertAlmostEqual(getattr(p1, attr), getattr(p2, attr), delta=delta)

    def test_rebound_transformations(self):
        avars = Andoyer.from_Simulation(self.sim, 4, 1)
        sim = avars.to_Simulation()
        self.compare_simulations(self.sim, sim, delta=1.e-3)

    def test_properties(self):
        a = Andoyer.from_Simulation(self.sim, 4, 1)
        p = a.params
        o = self.sim.calculate_orbits(jacobi_masses=True)
        self.assertAlmostEqual(a.lambda1, o[0].l, delta=self.delta)
        self.assertAlmostEqual(a.lambda2, o[1].l, delta=self.delta)
        self.assertAlmostEqual(a.deltalambda, o[1].l-o[0].l, delta=self.delta)
        self.assertAlmostEqual(np.mod(p['j']*a.lambda2 - (p['j']-p['k'])*a.lambda1 + p['k']*a.psi1, 2.*np.pi), a.phi, delta=self.delta)
        self.assertAlmostEqual(a.SPhi, a.Z**2/2., delta=self.delta)
        self.assertAlmostEqual(a.Phi/p['Phiscale'], a.Z**2/2., delta=self.delta)
        self.assertAlmostEqual(a.SX, a.Z*np.cos(a.phi), delta=self.delta)
        self.assertAlmostEqual(a.SY, a.Z*np.sin(a.phi), delta=self.delta)

    def test_andoyer_poincare_transformations(self):
        avars = Andoyer.from_Simulation(self.sim, 4, 1)
        
        pvars = avars.to_Poincare()
        avars2 = Andoyer.from_Poincare(pvars, 4, 1, avars.params['a10'])
        self.compare_andoyer(avars, avars2, delta=1.e-10)

        pvars2 = avars2.to_Poincare()
        self.compare_poincare_particles(pvars.particles, pvars2.particles, 1.e-10)

if __name__ == '__main__':
    unittest.main()

