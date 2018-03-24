import rebound
import unittest
import math
import numpy as np
from celmech import Andoyer, Poincare

class TestAndoyer(unittest.TestCase):
    def setUp(self):
        self.sim = rebound.Simulation()
        self.sim.add(m=1.)
        self.sim.add(m=1.e-5, P=1.5, e=0.02, pomega=0.3, l=0.4)
        self.sim.add(m=1.e-7, P=2., e=0.01, pomega=2.3, l=4.4)
        self.sim.add(m=1.e-3, a=1., e=0.2, pomega=0.3, l=0.4)
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
    
    def compare_simulations_orb(self, sim1, sim2, delta=1.e-15):
        self.assertAlmostEqual(sim1.particles[0].m, sim2.particles[0].m, delta=delta)
        orbs1 = sim1.calculate_orbits(jacobi_masses=True)
        orbs2 = sim2.calculate_orbits(jacobi_masses=True)
        for i in range(sim1.N-1):
            self.assertAlmostEqual(sim1.particles[i+1].m, sim2.particles[i+1].m, delta=delta)
            o1 = orbs1[i]
            o2 = orbs2[i]
            for attr in ['a', 'e', 'inc']:
                self.assertAlmostEqual(getattr(o1, attr), getattr(o2, attr), delta=delta)
            for attr in ['Omega', 'pomega', 'theta']:
                print(attr)
                self.assertAlmostEqual(np.cos(getattr(o1, attr)), np.cos(getattr(o2, attr)), delta=delta)
   
    def compare_particles(self, sim1, sim2, i1, i2, delta=1.e-15):
        orbs1 = sim1.calculate_orbits(jacobi_masses=True)
        orbs2 = sim2.calculate_orbits(jacobi_masses=True)
        o1 = orbs1[i1-1]
        o2 = orbs2[i2-1]
        m1 = sim1.particles[i1].m
        m2 = sim2.particles[i2].m
        self.assertAlmostEqual(m1, m2, delta=delta)
        for attr in ['a', 'e', 'inc']:
            self.assertAlmostEqual(getattr(o1, attr), getattr(o2, attr), delta=delta)
        for attr in ['Omega', 'pomega', 'theta']:
            self.assertAlmostEqual(np.cos(getattr(o1, attr)), np.cos(getattr(o2, attr)), delta=delta)

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
   
    def reduce_sim(self, i1, i2):
        sim2 = rebound.Simulation()
        sim2.G = self.sim.G
        ps = self.sim.particles
        sim2.add(ps[0])
        sim2.add(m=ps[i1].m, a=ps[i1].a, e=ps[i1].e, inc=ps[i1].inc, Omega=ps[i1].Omega, pomega=ps[i1].pomega, theta=ps[i1].theta, jacobi_masses=True)
        sim2.add(m=ps[i2].m, a=ps[i2].a, e=ps[i2].e, inc=ps[i2].inc, Omega=ps[i2].Omega, pomega=ps[i2].pomega, theta=ps[i2].theta, jacobi_masses=True)
        return sim2

    def test_masses(self):
        # turn off averaging for all transformation tests since averaging
        # is not symmetric back and forth (there's diff of O(s^2))
     
        masses = [1., 1.e-5, 1.e-7, 1.e-3]
        pairs = [[1,2], [2,3], [1,3]]
        for i1, i2 in pairs:
            m = [masses[0], masses[i1], masses[i2]]
            avars = Andoyer.from_Simulation(self.sim, 4, 1, average=False, i1=i1, i2=i2)
            sim = avars.to_Simulation(masses=m, average=False)
            self.compare_particles(self.sim, sim, i1, 1, self.delta)
            self.compare_particles(self.sim, sim, i2, 2, self.delta)
    
    def test_dP(self):
        j=5
        k=2
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=1.e-6, P=1.)
        sim.add(m=3.e-6, P=1.68)
        sim.move_to_com()
        avars = Andoyer.from_Simulation(sim,j,k, a10=0.32, average=False) # real a0 ~0.29, 10% err
        self.assertAlmostEqual(1.68-float(j)/(j-k), avars.dP, delta=0.01) # err is da^2 or smaller, so 1%

    def test_H(self):
        j=3
        k=1
        a10 = 1.02
        sim = rebound.Simulation()
        sim.G = 4*np.pi**2
        sim.add(m=1.)
        sim.add(m=1.e-6, e=0.01, P=1., pomega=-np.pi/2, f=np.pi, jacobi_masses=True)
        sim.add(m=3.e-6, e=0.03, pomega=np.pi/2, P=float(j)/(j-k), jacobi_masses=True)#float(j)/(j-k), theta=3.14)
        sim.move_to_com()
        avars = Andoyer.from_Simulation(sim,j,k, a10=a10, average=True) 
        p = avars.params
        pvars = Poincare.from_Simulation(sim, average=True) 
        Gamma1 = pvars.particles[1].Gamma
        Gamma2 = pvars.particles[2].Gamma
        Lambda1 = pvars.particles[1].Lambda
        Lambda2 = pvars.particles[2].Lambda
        lambda1 = pvars.particles[1].l
        lambda2 = pvars.particles[2].l
        gamma1 = pvars.particles[1].gamma
        gamma2 = pvars.particles[2].gamma
        n10 = np.sqrt(p['G']*p['M1']/p['a10']**3)
        n20 = np.sqrt(p['G']*p['M2']/p['a10']**3*p['alpha']**3)
        z1 = np.sqrt(2*Gamma1/p['Lambda10'])
        z2 = np.sqrt(2*Gamma2/p['Lambda20'])
        Hkep = -0.5*(n10*p['Lambda10']**3/Lambda1**2 + n20*p['Lambda20']**3/Lambda2**2)
        L10 = p['Lambda10']
        L20 = p['Lambda20']
        Hkepexpanded = -n10*L10**3/2*(1/L10**2 - 2*avars.dL1/L10**3 + 3*avars.dL1**2/L10**4)-n20*L20**3/2*(1/L20**2 - 2*avars.dL2/L20**3 + 3*avars.dL2**2/L20**4)
        Hresprefac = -p['G']*p['m1']*p['m2']/p['a10']*p['alpha']
        Hres = Hresprefac*(p['f']*z1*np.cos(j*lambda2 - (j-k)*lambda1 + gamma1)+p['g']*z2*np.cos(j*lambda2 - (j-k)*lambda1 + gamma2))
        H0 = -n20*p['K0']/2/(j-k)
        H1 = n20*p['eta']/(j-k)*avars.dK*(1. - 1.5*p['eta']/p['K0']*avars.dK)
        H2 = n20*p['eta']*p['a']*avars.dP**2
        Hkeptransformed = H0 + H1 + H2
        Hrestransformed = n20*p['eta']*p['c']*(2*avars.Psi1)**(k/2.)*np.cos(avars.theta+k*avars.psi1)

        self.assertAlmostEqual(Hkeptransformed, Hkepexpanded, delta=1.e-15) # should be exact
        self.assertAlmostEqual(Hrestransformed, Hres, delta=1.e-15) # should be exact for first order resonance (k=1)
        self.assertAlmostEqual(Hkepexpanded, Hkep, delta=(a10-1.)**2) # should match to O(da/a)^2, atrue=1, a10=a10
    
    def test_ecom(self):
        j=57
        k=2
        sim = rebound.Simulation()
        sim.G = 4*np.pi**2
        sim.add(m=1.)
        sim.add(m=1.e-6, e=0.01, P=1., pomega=-np.pi/6, f=np.pi, jacobi_masses=True)
        sim.add(m=3.e-6, e=0.03, pomega=np.pi/3, P=float(j)/(j-k), jacobi_masses=True)#float(j)/(j-k), theta=3.14)
        sim.move_to_com()
        ps = sim.particles
        e1x = ps[1].e*np.cos(ps[1].pomega)
        e1y = ps[1].e*np.sin(ps[1].pomega)
        e2x = ps[2].e*np.cos(ps[2].pomega)
        e2y = ps[2].e*np.sin(ps[2].pomega)

        avars = Andoyer.from_Simulation(sim,j,k, a10=1.02, average=True) 
        m1 = avars.params['m1']
        m2 = avars.params['m2']
        ecomx = (m1*e1x + m2*e2x)/(m1+m2)
        ecomy = (m1*e1y + m2*e2y)/(m1+m2)
        ecomsim = np.sqrt(ecomx**2 + ecomy**2)
                
        self.assertAlmostEqual(avars.ecom, ecomsim, delta=1.e-3)

    def test_from_elements(self):
        j=5
        k=2
        Zstar=0.1
        libfac=0.5
        a10=3.
        a1=3.1
        G=2.
        m1=1.e-7
        m2=1.e-4
        ecom=0.05
        phiecom=0.7

        avars = Andoyer.from_elements(j,k,Zstar,libfac,a10,a1,G,m1=m1,m2=m2,ecom=ecom,phiecom=phiecom)
        self.assertAlmostEqual(avars.Zstar, Zstar, delta=1.e-12)
        self.assertAlmostEqual(avars.ecom, ecom, delta=1.e-15)
        self.assertAlmostEqual(avars.phiecom, phiecom, delta=1.e-15)
        sim = avars.to_Simulation()
        self.assertAlmostEqual(sim.particles[1].a, a1, delta=3*((a1-a10)/a10)**2)# should match to O(da/a)^2, atrue=1, a10=a10
    '''
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
    '''

if __name__ == '__main__':
    unittest.main()

