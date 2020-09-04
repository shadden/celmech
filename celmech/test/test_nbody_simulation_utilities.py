import rebound as rb
import unittest
import numpy as np
from celmech.nbody_simulation_utilities import get_canonical_heliocentric_orbits
from celmech.nbody_simulation_utilities import add_canonical_heliocentric_elements_particle
from random import random, seed

class TestHeliocentricElements(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.sim = rb.Simulation()
        self.sim.add(m=1)
        for i in range(1,10):
            self.sim.add(
                    m=10**np.random.uniform(-6,-3),
                    a=i,
                    e=np.random.rayleigh(0.04),
                    inc=np.random.rayleigh(0.04),
                    pomega='uniform',
                    l='uniform',
                    Omega='uniform'
            )
        self.sim.move_to_com()
        self.helio_orbits = get_canonical_heliocentric_orbits(self.sim)
        self.masses = [p.m for p in self.sim.particles]

    def tearDown(self):
        self.sim = None

    def compare_simulations(self, sim1, sim2, rtol=1.e-15):
        for i in range(1,sim1.N):
            p1 = sim1.particles[i]
            p2 = sim2.particles[i]
            r1 = np.linalg.norm(p1.xyz)
            v1 = np.linalg.norm(p1.vxyz)
            
            for attr in ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz']:
                q1 = getattr(p1, attr)
                q2 = getattr(p2, attr)
                if attr[0]=='v':
                    eps = v1 * rtol
                else:
                    eps = r1 * rtol
                self.assertAlmostEqual(q1,q2,delta=eps)
    def test_add_helio_particles(self):
        newsim = rb.Simulation()
        newsim.add(m=self.masses[0])
        for m,orbit in zip(self.masses[1:],self.helio_orbits):
            eldict = {el:getattr(orbit,el) for el in ['a','e','inc','l','pomega','Omega']}
            add_canonical_heliocentric_elements_particle(m,eldict,newsim)
        newsim.move_to_com()
        self.compare_simulations(self.sim, newsim, rtol=1.e-13)

if __name__ == '__main__':
    unittest.main()
