import rebound
import unittest
import math
import numpy as np
from celmech import Poincare, PoincareHamiltonian, PoincareParticle
from random import random, seed

class TestPoincare(unittest.TestCase):

    def setUp(self):
        self.sim = rebound.Simulation()
        self.sim.add(m=1.)
        self.sim.add(m=1.e-3, a=1.0, e=0.2, inc=0.10, pomega=0.3, l=0.4, Omega=0.2)
        self.sim.add(m=1.e-7, a=1.3, e=0.1, inc=0.05, pomega=2.3, l=4.4, Omega=2.1)
        self.sim.add(m=1.e-5, a=2.9, e=0.3, inc=0.07, pomega=1.3, l=3.4, Omega=5.2)
        self.sim.move_to_com()
        self.coordinates = ['canonical heliocentric', 'democratic heliocentric']

    def tearDown(self):
        self.sim = None

    def compare_objects(self, obj1, obj2, delta=1.e-15):
        self.assertEqual(type(obj1), type(obj2))
        for attr in [attr for attr in dir(obj1) if not attr.startswith('_')]:
            self.assertAlmostEqual(getattr(obj1, attr), getattr(obj2, attr), delta=delta)
    
    def compare_poincare_particles(self, ps1, ps2, delta=1.e-15):
        self.assertEqual(type(ps1), type(ps2))
        ps1 = ps1[1:]
        ps2 = ps2[1:] # ignore the dummy particle for primary at index 0
        for p1, p2 in zip(ps1, ps2):
            for attr in ['skappa', 'seta','ssigma','srho', 'm', 'M', 'sLambda', 'l']:
                self.assertAlmostEqual(getattr(p1, attr), getattr(p2, attr), delta=delta)

    def compare_simulations(self, sim1, sim2, delta=1.e-15):
        for i in range(1,sim1.N):
            p1 = sim1.particles[i]
            p2 = sim2.particles[i]
            for attr in ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz']:
                self.assertAlmostEqual(getattr(p1, attr), getattr(p2, attr), delta=delta)

    def compare_simulations_orb(self, sim1, sim2, delta=1.e-15):
        self.assertAlmostEqual(sim1.particles[0].m, sim2.particles[0].m, delta=delta)
        orbs1 = sim1.calculate_orbits()
        orbs2 = sim2.calculate_orbits()
        for i in range(sim1.N-1):
            self.assertAlmostEqual(sim1.particles[i+1].m, sim2.particles[i+1].m, delta=delta, msg="mass")
            o1 = orbs1[i]
            o2 = orbs2[i]
            for attr in ['a', 'e', 'inc']:
                self.assertAlmostEqual(getattr(o1, attr), getattr(o2, attr), delta=delta, msg=attr)
            for attr in ['Omega', 'pomega', 'theta']:
                self.assertAlmostEqual(np.cos(getattr(o1, attr)), np.cos(getattr(o2, attr)), delta=delta)
   
    def test_masses(self):
        m=1e-3
        Mstar=1
        for coord in self.coordinates:
            p = PoincareParticle(m=m, Mstar=Mstar, a=1., coordinates=coord)
            p = PoincareParticle(mu=p.mu, M=p.M, a=1., coordinates=coord)
            self.assertAlmostEqual(p.m, m, delta=1e-15)
            self.assertAlmostEqual(p.Mstar, Mstar, delta=1e-15)

    def test_orbelements(self):
        for coord in self.coordinates:
            p = PoincareParticle(coordinates=coord, mu=1.e-3, M=3., a=2., e=0.5, inc=0.7, q=-0.3, gamma=-0.5, l=2.3)
            self.assertAlmostEqual(p.a, 2., delta=1.e-15)
            self.assertAlmostEqual(p.e, 0.5, delta=1.e-15)
            self.assertAlmostEqual(p.inc, 0.7, delta=1.e-15)
            self.assertAlmostEqual(p.Omega, 0.3, delta=1.e-15)
            self.assertAlmostEqual(p.pomega, 0.5, delta=1.e-15)
            self.assertAlmostEqual(p.l, 2.3, delta=1.e-15)

    def test_tp_orbelements(self):
        for coord in self.coordinates:
            p = PoincareParticle(coordinates=coord, mu=0., M=3., a=2., e=0.5, inc=0.7, q=-0.3, gamma=-0.5, l=2.3)
            self.assertAlmostEqual(p.a, 2., delta=1.e-15)
            self.assertAlmostEqual(p.e, 0.5, delta=1.e-15)
            self.assertAlmostEqual(p.inc, 0.7, delta=1.e-15)
            self.assertAlmostEqual(p.Omega, 0.3, delta=1.e-15)
            self.assertAlmostEqual(p.pomega, 0.5, delta=1.e-15)
            self.assertAlmostEqual(p.l, 2.3, delta=1.e-15)

    def test_copy(self):
        for coord in self.coordinates:
            pvars = Poincare.from_Simulation(self.sim, coordinates=coord)
            pvars2 = pvars.copy()
            self.compare_poincare_particles(pvars.particles, pvars2.particles) # ignore nans in particles[0]
        
    def test_rebound_transformations(self):
        for coord in self.coordinates:
            pvars = Poincare.from_Simulation(self.sim, coordinates=coord)
            sim = pvars.to_Simulation()
            self.compare_simulations_orb(self.sim, sim, delta=1.e-14) # can't get it to 1e-15

    def test_particles(self):
        m=1.e-5
        M=3.
        G=2.
        a=7.
        e=0.1
        pomega = 1.3
        Omega = 2.1
        inc = 0.07
        l=0.7
        sLambda = np.sqrt(G*M*a)
        sGamma = sLambda*(1.-np.sqrt(1.-e**2))
        sQ = (sLambda-sGamma)*(1-np.cos(inc))
        Lambda = m*sLambda
        Gamma = m*sGamma
        Q = m*sQ
        p = PoincareParticle(mu=m, M=M, G=G,a=a,e=e,inc=inc,pomega=pomega,Omega=Omega,l=l)
        tp = PoincareParticle(mu=0, M=M, G=G,a=a,e=e,inc=inc,pomega=pomega,Omega=Omega,l=l)
        self.assertAlmostEqual(p.a, a, delta=1.e-15)
        self.assertAlmostEqual(p.e, e, delta=1.e-15)
        self.assertAlmostEqual(p.inc, inc, delta=1.e-15)
        self.assertAlmostEqual(p.pomega, pomega, delta=1.e-15)
        self.assertAlmostEqual(p.Omega, Omega, delta=1.e-15)
        self.assertAlmostEqual(tp.a, a, delta=1.e-15)
        self.assertAlmostEqual(tp.e, e, delta=1.e-15)
        self.assertAlmostEqual(tp.inc, inc, delta=1.e-15)
        self.assertAlmostEqual(tp.pomega, pomega, delta=1.e-15)
        self.assertAlmostEqual(tp.Omega, Omega, delta=1.e-15)
        p = PoincareParticle(mu=m, M=M, G=G, Lambda=Lambda, l=l, Gamma=Gamma, gamma=-pomega, Q=Q, q=-Omega)
        tp = PoincareParticle(mu=0., M=M, G=G, sLambda=sLambda, l=l, sGamma=sGamma, gamma=-pomega, sQ=sQ, q=-Omega)
        self.assertAlmostEqual(p.a, a, delta=1.e-15)
        self.assertAlmostEqual(p.e, e, delta=1.e-15)
        self.assertAlmostEqual(p.inc, inc, delta=1.e-15)
        self.assertAlmostEqual(p.pomega, pomega, delta=1.e-15)
        self.assertAlmostEqual(p.Omega, Omega, delta=1.e-15)
        self.assertAlmostEqual(tp.a, a, delta=1.e-15)
        self.assertAlmostEqual(tp.e, e, delta=1.e-15)
        self.assertAlmostEqual(tp.inc, inc, delta=1.e-15)
        self.assertAlmostEqual(tp.pomega, pomega, delta=1.e-15)
        self.assertAlmostEqual(tp.Omega, Omega, delta=1.e-15)
        pvars = Poincare(G, poincareparticles=[p, tp])
        ps = pvars.particles
        self.assertAlmostEqual(ps[1].a, a, delta=1.e-14)
        self.assertAlmostEqual(ps[1].e, e, delta=1.e-14)
        self.assertAlmostEqual(ps[1].inc, inc, delta=1.e-14)
        self.assertAlmostEqual(ps[1].pomega, pomega, delta=1.e-14)
        self.assertAlmostEqual(ps[1].Omega, Omega, delta=1.e-14)
        # test that we raise error when accessing test particle in Poincare. Change with test if we implement
        with self.assertRaises(AttributeError):
            print(ps[2].a)

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
    tmax = 30*tsyn
    Nout = 100
    times = np.linspace(0, tmax, Nout)

    pvars = Poincare.from_Simulation(sim)
    Hsim = PoincareHamiltonian(pvars)

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

