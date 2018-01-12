import rebound
import reboundx
import unittest
import math
import numpy as np
from celmech import Poincare, PoincareHamiltonian
from random import random, seed

class TestPoincare(unittest.TestCase):
    def setUp(self):
        self.sim = rebound.Simulation()
        self.sim.add(m=1.)
        self.sim.add(m=1.e-3, a=1., e=0.2, pomega=0.3, l=0.4)
        self.sim.add(m=1.e-7, a=1.3, e=0.1, pomega=2.3, l=4.4)
        self.sim.add(m=1.e-5, a=2.9, e=0.3, pomega=1.3, l=3.4)
        self.sim.move_to_com()

    def tearDown(self):
        self.sim = None

    def compare_objects(self, obj1, obj2, delta=1.e-15):
        self.assertEqual(type(obj1), type(obj2))
        for attr in [attr for attr in dir(obj1) if not attr.startswith('_')]:
            self.assertAlmostEqual(getattr(obj1, attr), getattr(obj2, attr), delta=delta)
    
    def compare_poincare_particles(self, ps1, ps2, delta=1.e-15):
        self.assertEqual(type(ps1), type(ps2))
        for p1, p2 in zip(ps1, ps2):
            for attr in ['X', 'Y', 'm', 'M', 'Lambda', 'l']:
                self.assertAlmostEqual(getattr(p1, attr), getattr(p2, attr), delta=delta)

    def compare_simulations(self, sim1, sim2, delta=1.e-15):
        equal = True
        for i in range(1,sim1.N):
            p1 = sim1.particles[i]
            p2 = sim2.particles[i]
            for attr in ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz']:
                self.assertAlmostEqual(getattr(p1, attr), getattr(p2, attr), delta=delta)

    def test_orbelements(self):
        pvars = Poincare.from_Simulation(self.sim, average=False)
        ps = pvars.particles
        o = self.sim.calculate_orbits(jacobi_masses=True)
        for i in range(1,4):
            self.assertAlmostEqual(o[i-1].a, ps[i].a, delta=1.e-15)
            self.assertAlmostEqual(o[i-1].e, ps[i].e, delta=1.e-15)
            self.assertAlmostEqual(o[i-1].l, ps[i].l, delta=1.e-15)
            self.assertAlmostEqual(o[i-1].pomega, ps[i].pomega, delta=1.e-15)

    def test_rebound(self):
        orbs = self.sim.calculate_orbits(primary=self.sim.particles[0])
        sim = rebound.Simulation()
        sim.add(m=1.)
        for i, orb in enumerate(orbs):
            sim.add(m=self.sim.particles[i+1].m, a=orb.a, e=orb.e, pomega=orb.pomega, l=orb.l, primary=sim.particles[0])
        sim.move_to_com()
        self.compare_simulations(self.sim, sim, delta=1.e-14)

    def test_copy(self):
        pvars = Poincare.from_Simulation(self.sim)
        pvars2 = pvars.copy()
        self.compare_poincare_particles(pvars.particles[1:], pvars2.particles[1:]) # ignore nans in particles[0]
        
    def test_rebound_transformations(self):
        pvars = Poincare.from_Simulation(self.sim, average = False)
        sim = pvars.to_Simulation(average = False)
        self.compare_simulations(self.sim, sim, delta=1.e-14)

    def test_averaging(self): # see PoincareTest.ipynb in celmech/celmech/test for a description
        errs = np.array([averaging_error(Nseed) for Nseed in range(100)]) # takes about 5 sec
        self.assertLess(np.median(errs), 3.)        

def packed_sim(Nseed):
    seed(Nseed)
    sim = rebound.Simulation()
    sim.add(m=1)
     
    sim.G = 4*np.pi*np.pi
    a1 = 1
    m1 = 10**(6*random()-10) # uniform in log space between 10**[-10, -4]
    m2 = 10**(6*random()-10)
    m3 = 10**(6*random()-10) 
    RH = a1*((m1+m2)/3.)**(1./3.)
    beta = 4 + 6*random() # uniform on [4, 10] mutual hill radii separation.
    sim.add(m=m1,a=a1)
    sim.add(m=m2,a=a1+beta*RH, l=0)
    sim.add(m=m3,a=a1+2*beta*RH, l=0)

    sim.move_to_com()

    sim.dt = sim.particles[1].P / 30.
    sim.integrator='whfast'
    if sim.particles[1].a/sim.particles[2].a > 0.98:
        return packed_sim(Nseed+1000)
    return sim

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def averaging_error(Nseed):
    sim = packed_sim(Nseed)
    ps = sim.particles

    o = sim.calculate_orbits(jacobi_masses=True)
    a10 = o[0].a
    a20 = o[1].a
    a30 = o[2].a
    tsyn = 2*np.pi/(o[0].n-o[1].n)
    #print(tsyn)
    tmax = 30*tsyn
    Nout = 100
    times = np.linspace(0, tmax, Nout)

    pvars = Poincare.from_Simulation(sim)
    Hsim = PoincareHamiltonian(pvars)

    #print(ps[1].m, ps[2].m)
    
    Nsma = np.zeros((3,Nout))
    Hsma = np.zeros((3,Nout))

    for i,t in enumerate(times):
        # Store N-body data
        o = sim.calculate_orbits(jacobi_masses=True)
        Nsma[0,i]=o[0].a
        Nsma[1,i]=o[1].a
        Nsma[2,i]=o[2].a
        
        ps = Hsim.particles
        Hsma[0,i]=ps[1].a
        Hsma[1,i]=ps[2].a
        Hsma[2,i]=ps[3].a

        sim.integrate(t)
        Hsim.integrate(t)
        
    Nmad1 = mad((Nsma[0]-a10)/a10)
    Nmad2 = mad((Nsma[1]-a20)/a20)
    Nmad3 = mad((Nsma[2]-a30)/a30)
    
    #print(Nmad1, Nmad2)
    Nmed1 = np.median((Nsma[0]-a10)/a10)
    Pmed1 = np.median((Hsma[0]-a10)/a10)
    err1 = abs(Pmed1-Nmed1)/Nmad1
    
    Nmed2 = np.median((Nsma[1]-a20)/a20)
    Pmed2 = np.median((Hsma[1]-a20)/a20)
    err2 = abs(Pmed2-Nmed2)/Nmad2
    
    Nmed3 = np.median((Nsma[2]-a30)/a30)
    Pmed3 = np.median((Hsma[2]-a30)/a30)
    err3 = abs(Pmed3-Nmed3)/Nmad3
    
    return max(err1, err2, err3)

if __name__ == '__main__':
    unittest.main()

