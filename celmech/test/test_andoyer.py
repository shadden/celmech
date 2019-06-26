import rebound
import unittest
import math
import numpy as np
from celmech import Andoyer, Poincare
from celmech.transformations import ActionAngleToXY, XYToActionAngle

class TestAndoyer(unittest.TestCase):
    def setUp(self):
        self.sim = rebound.Simulation()
        self.sim.add(m=1.)
        self.sim.add(m=1.e-5, P=1.5, e=0.02, pomega=0.3, l=0.4)
        self.sim.add(m=1.e-7, P=2., e=0.01, pomega=2.3, l=4.4)
        self.sim.add(m=1.e-3, a=1., e=0.2, pomega=0.3, l=0.4)
        self.sim.move_to_com()
        self.delta = 1.e-10 # some digits are lost taking differences of nearby quantities to get resonance

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

    #def compare_andoyer(self, a1, a2, delta=1.e-15):
    #    self.assertEqual(type(a1), type(a2))
    #    for attr in ['X', 'Y', 'Psi2', 'psi2', 'Phiprime', 'K', 'deltalambda', 'lambda1']:
    #        self.assertAlmostEqual(getattr(a1, attr), getattr(a2, attr), delta=delta)
    
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
        sGamma1 = pvars.particles[1].sGamma
        sGamma2 = pvars.particles[2].sGamma
        sLambda1 = pvars.particles[1].sLambda
        sLambda2 = pvars.particles[2].sLambda
        lambda1 = pvars.particles[1].l
        lambda2 = pvars.particles[2].l
        gamma1 = pvars.particles[1].gamma
        gamma2 = pvars.particles[2].gamma
        n10 = np.sqrt(p['G']*p['M1']/p['a10']**3)
        n20 = np.sqrt(p['G']*p['M2']/p['a10']**3*p['alpha']**3)
        z1 = np.sqrt(2*sGamma1/p['sLambda10'])
        z2 = np.sqrt(2*sGamma2/p['sLambda20'])
        Hkep = -0.5*(n10*p['m1']*p['sLambda10']**3/sLambda1**2 + n20*p['m2']*p['sLambda20']**3/sLambda2**2)
        sL10 = p['sLambda10']
        sL20 = p['sLambda20']
        Hkepexpanded = -n10*p['m1']*sL10/2*(1. - 2*avars.dL1hat + 3*avars.dL1hat**2)-n20*p['m2']*sL20/2*(1. - 2*avars.dL2hat + 3*avars.dL2hat**2)
        Hresprefac = -p['G']*p['m1']*p['m2']/p['a10']*p['alpha']
        Hres = Hresprefac*(p['f']*z1*np.cos(j*lambda2 - (j-k)*lambda1 + gamma1)+p['g']*z2*np.cos(j*lambda2 - (j-k)*lambda1 + gamma2))
        H0 = -p['n0']*p['K0']/2
        H1 = p['eta']*p['n0']*avars.dK*(1-1.5*p['eta']/p['K0']*avars.dK)
        H2 = p['eta']*p['a']*avars.dP**2
        Hkeptransformed = H0 + H1 + H2
        Hrestransformed = p['eta']*p['c']*(2*avars.Psi1)**(k/2.)*np.cos(avars.theta+k*avars.psi1)

        self.assertAlmostEqual(Hkeptransformed, Hkepexpanded, delta=1.e-15) # should be exact
        self.assertAlmostEqual(Hrestransformed, Hres, delta=1.e-15) # should be exact for first order resonance (k=1)
        self.assertAlmostEqual(Hkepexpanded, Hkep, delta=(a10-1.)**2) # should match to O(da/a)^2, atrue=1, a10=a10

        #Hfinal = -p['eta']*p['Phi0']/p['tau']*(4.*avars.Phi**2 - 3.*avars.Phiprime*avars.Phi + 9./16.*avars.Phiprime**2 + (2.*avars.Phi)**(k/2.)*np.cos(avars.phi) - p['n0']*p['tau']/p['Phi0']*avars.dK*(1.-1.5*p['eta']/p['K0']*avars.dK))+ H0
        Hfinal = -p['eta']*p['Phi0']/p['tau']*(4.*(avars.Phi-avars.B)**2 + (2.*avars.Phi)**(k/2.)*np.cos(avars.phi) - p['n0']*p['tau']/p['Phi0']*avars.dK*(1.-1.5*p['eta']/p['K0']*avars.dK))+ H0
        self.assertAlmostEqual(Hfinal, Hkepexpanded+Hres, delta=1.e-15) # should be exact
    
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
                
        self.assertAlmostEqual(avars.Zcom, ecomsim, delta=1.e-3)
        phiecomsim = np.arctan2(ecomy, ecomx)
        self.assertAlmostEqual(avars.phiZcom, phiecomsim, delta=1.e-3)

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

        avars = Andoyer.from_elements(j,k,Zstar,libfac,a10,a1,G,m1=m1,m2=m2,Zcom=ecom,phiZcom=phiecom)
        self.assertAlmostEqual(avars.Zstar, Zstar, delta=1.e-12)
        self.assertAlmostEqual(avars.Zcom, ecom, delta=1.e-15)
        self.assertAlmostEqual(avars.phiZcom, phiecom, delta=1.e-15)
        sim = avars.to_Simulation()
        self.assertAlmostEqual(sim.particles[1].a, a1, delta=3*((a1-a10)/a10)**2)# should match to O(da/a)^2, atrue=1, a10=a10

    def test_ZsGammaConversion(self):
        avars = Andoyer.from_Simulation(self.sim, 4, 1)
        Z = avars.Z
        phiZ = avars.psi1
        Zcom = avars.Zcom
        phiZcom = avars.phiZcom
        sGamma1, gamma1, sGamma2, gamma2 = avars.Zs_to_sGammas(Z, phiZ, Zcom, phiZcom)
        nZ, nphiZ, nZcom, nphiZcom = avars.sGammas_to_Zs(sGamma1, gamma1, sGamma2, gamma2)
        self.assertAlmostEqual(nZ,Z, delta=1.e-15)
        self.assertAlmostEqual(np.mod(nphiZ, 2*np.pi), np.mod(phiZ, 2*np.pi), delta=1.e-15)
        self.assertAlmostEqual(nZcom, Zcom, delta=1.e-15)
        self.assertAlmostEqual(np.mod(nphiZcom, 2*np.pi), np.mod(phiZcom, 2*np.pi), delta=1.e-15)

        # should also be equivalent for 2 massive particles to rotation
        pvars = Poincare.from_Simulation(self.sim)
        ps = pvars.particles
        p = avars.params
        f, g = p['f'], p['g']
        ff = f*np.sqrt(p['eta']/p['m1']/p['sLambda10'])
        gg = g*np.sqrt(p['eta']/p['m2']/p['sLambda20'])
        norm = np.sqrt(ff*ff + gg*gg)
        psirotmatrix = np.array([[ff,gg],[-gg,ff]]) / norm
        invpsirotmatrix = np.array([[ff,-gg],[gg,ff]]) / norm
        Psi1,psi1,Psi2,psi2 = self.rotate_actions(ps[1].Gamma/p['eta'],ps[1].gamma,ps[2].Gamma/p['eta'],ps[2].gamma, psirotmatrix)
        self.assertAlmostEqual(Psi1, avars.Psi1, delta=1.e-15)
        self.assertAlmostEqual(np.mod(psi1, 2*np.pi), np.mod(avars.psi1, 2*np.pi), delta=1.e-15)
        self.assertAlmostEqual(Psi2, avars.Psi2, delta=1.e-15)
        self.assertAlmostEqual(np.mod(psi2, 2*np.pi), np.mod(avars.psi2, 2*np.pi), delta=1.e-15)

    def rotate_actions(self,A1,a1,A2,a2,rotmatrix):
        AX1,AY1 = ActionAngleToXY(A1,a1)
        AX2,AY2 = ActionAngleToXY(A2,a2)
        BX1,BX2 = np.dot(rotmatrix, np.array([AX1,AX2]) )
        BY1,BY2 = np.dot(rotmatrix, np.array([AY1,AY2]) )
        B1,b1 = XYToActionAngle(BX1,BY1)
        B2,b2 = XYToActionAngle(BX2,BY2)
        return B1,b1,B2,b2

    def test_scale_invariance(self):
        avars = Andoyer.from_Simulation(self.sim, 4, 1, i1=1, i2=2)
        
        mfac = 3.
        dfac = 0.05
        vfac = np.sqrt(mfac/dfac)
        for p in self.sim.particles:
            p.m *= mfac 
            p.vx *= vfac
            p.vy *= vfac
            p.vz *= vfac
            p.x *= dfac
            p.y *= dfac
            p.z *= dfac

        avarsscaled = Andoyer.from_Simulation(self.sim, 4, 1, i1=1, i2=2)
        self.assertAlmostEqual(avars.X, avarsscaled.X, delta=self.delta) 
        self.assertAlmostEqual(avars.Y, avarsscaled.Y, delta=self.delta) 
        self.assertAlmostEqual(avars.Zcom, avarsscaled.Zcom, delta=self.delta) 
        self.assertAlmostEqual(np.cos(avars.phiZcom), np.cos(avarsscaled.phiZcom), delta=self.delta) 
        self.assertAlmostEqual(avars.B, avarsscaled.B, delta=self.delta) 
        self.assertAlmostEqual(avars.dKprime, avarsscaled.dKprime, delta=self.delta) 
        self.assertAlmostEqual(np.cos(avars.theta), np.cos(avarsscaled.theta), delta=self.delta) 
        self.assertAlmostEqual(np.cos(avars.theta1), np.cos(avarsscaled.theta1), delta=self.delta) 

    def test_rotational_invariance(self):
        j=4
        k=1
        avars = Andoyer.from_Simulation(self.sim, j, k, i1=1, i2=2)

        rot = np.pi/3.
        ps = self.sim.particles
        simrot = rebound.Simulation()
        simrot.G = self.sim.G
        simrot.add(m=ps[0].m)
        for p in ps[1:]:
            simrot.add(m=p.m, a=p.a, e=p.e, inc=p.inc, Omega=p.Omega+rot, pomega=p.pomega+rot, l=p.l+rot)
        simrot.move_to_com()

        avarsrot = Andoyer.from_Simulation(simrot, j, k, i1=1, i2=2)
        self.assertAlmostEqual(avars.X, avarsrot.X, delta=self.delta) 
        self.assertAlmostEqual(avars.Y, avarsrot.Y, delta=self.delta) 
        self.assertAlmostEqual(avars.Zcom, avarsrot.Zcom, delta=self.delta) 
        self.assertAlmostEqual(np.cos(avars.phiZcom+rot), np.cos(avarsrot.phiZcom), delta=self.delta) 
        self.assertAlmostEqual(avars.B, avarsrot.B, delta=self.delta) 
        self.assertAlmostEqual(avars.dKprime, avarsrot.dKprime, delta=self.delta) 
        self.assertAlmostEqual(np.cos(avars.theta+k*rot), np.cos(avarsrot.theta), delta=self.delta) 
        p = avars.params
        fac = (p['m1']*p['sLambda10'] + p['m2']*p['sLambda20'])/p['K0']
        self.assertAlmostEqual(np.cos(avars.theta1+fac*rot), np.cos(avarsrot.theta1), delta=self.delta) 

    '''
    def test_rebound_transformations(self):
        avars = andoyer.from_simulation(self.sim, 4, 1)
        sim = avars.to_simulation()
        self.compare_simulations(self.sim, sim, delta=1.e-3)

    def test_properties(self):
        a = andoyer.from_simulation(self.sim, 4, 1)
        p = a.params
        o = self.sim.calculate_orbits(jacobi_masses=true)
        self.assertalmostequal(a.lambda1, o[0].l, delta=self.delta)
        self.assertalmostequal(a.lambda2, o[1].l, delta=self.delta)
        self.assertalmostequal(a.deltalambda, o[1].l-o[0].l, delta=self.delta)
        self.assertalmostequal(np.mod(p['j']*a.lambda2 - (p['j']-p['k'])*a.lambda1 + p['k']*a.psi1, 2.*np.pi), a.phi, delta=self.delta)
        self.assertalmostequal(a.sphi, a.z**2/2., delta=self.delta)
        self.assertalmostequal(a.phi/p['phiscale'], a.z**2/2., delta=self.delta)
        self.assertalmostequal(a.sx, a.z*np.cos(a.phi), delta=self.delta)
        self.assertalmostequal(a.sy, a.z*np.sin(a.phi), delta=self.delta)

    def test_andoyer_poincare_transformations(self):
        avars = andoyer.from_simulation(self.sim, 4, 1)
        
        pvars = avars.to_poincare()
        avars2 = andoyer.from_poincare(pvars, 4, 1, avars.params['a10'])
        self.compare_andoyer(avars, avars2, delta=1.e-10)

        pvars2 = avars2.to_poincare()
        self.compare_poincare_particles(pvars.particles, pvars2.particles, 1.e-10)
    '''

if __name__ == '__main__':
    unittest.main()

