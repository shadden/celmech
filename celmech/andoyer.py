import numpy as np
from sympy import symbols, S, cos
from celmech.hamiltonian import Hamiltonian
from celmech.poincare import PoincareParticle, Poincare
from celmech.disturbing_function import get_fg_coeffs
from celmech.transformations import ActionAngleToXY, XYToActionAngle
import rebound

def rotate_Poincare_Gammas_To_Psi1Psi2(Gamma1,gamma1,Gamma2,gamma2,f,g,inverse=False):
    if inverse is True:
        g = -g
    X1,Y1 = ActionAngleToXY(Gamma1,gamma1)
    X2,Y2 = ActionAngleToXY(Gamma2,gamma2)
    norm = np.sqrt(f*f + g*g)
    rotation_matrix = np.array([[f,g],[-g,f]]) / norm
    Psi1X,Psi2X = np.dot(rotation_matrix , np.array([X1,X2]) )
    Psi1Y,Psi2Y = np.dot(rotation_matrix , np.array([Y1,Y2]) )
    Psi1,psi1 = XYToActionAngle(Psi1X,Psi1Y)
    Psi2,psi2 = XYToActionAngle(Psi2X,Psi2Y)
    return Psi1,psi1,Psi2,psi2

class Andoyer(object):
    def __init__(self, Lambda1, Lambda2, lambda1, lambda2, Gamma1, Gamma2, gamma1, gamma2, j, k, a10=1., G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1.):
        self.calc_params(j, k, a10, G, m1, M1, m2, M2)
        self.Lambda1 = Lambda1
        self.Lambda2 = Lambda2
        self.Gamma1 = Gamma1
        self.Gamma2 = Gamma2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def calc_params(self, j, k, a10, G, m1, M1, m2, M2):
        self.p = {'j':j, 'k':k, 'a10':a10, 'G':G, 'm1':m1, 'M1':M1, 'm2':m2, 'M2':M2}

        self.p['beta'] = m2/m1*(M2/M1)**(2./3.)*(float(j)/(j-k))**(1./3.)
        self.p['Lambda10'] = m1*np.sqrt(G*M1*a10)
        self.p['Lambda20'] = self.p['beta']*self.p['Lambda10']
    
    @classmethod
    def from_Poincare(cls, pvars, j, k, a10, i1=1, i2=2):
        p1 = pvars.particles[i1]
        p2 = pvars.particles[i2]
        andvars = cls(p1.Lambda, p2.Lambda, p1.l, p2.l, p1.Gamma, p2.Gamma, p1.gamma, p2.gamma, j, k, a10, p1.G, p1.m, p1.M, p2.m, p2.M)
        return andvars

    def to_Poincare(self):
        Lambda1 = self.Lambda1
        Lambda2 = self.Lambda2
        Gamma1 = self.Gamma1
        Gamma2 = self.Gamma2
        lambda1 = self.lambda1
        lambda2 = self.lambda2
        gamma1 = self.gamma1
        gamma2 = self.gamma2
        
        p = self.p
        pvars = Poincare(p['G'])
        pvars.add(p['m1'], Lambda1, lambda1, Gamma1, gamma1, p['M1'])
        pvars.add(p['m2'], Lambda2, lambda2, Gamma2, gamma2, p['M2'])
        return pvars

    @classmethod
    def from_Simulation(cls, sim, j, k, a10=None, i1=1, i2=2, average=True):
        if a10 is None:
            a10 = sim.particles[i1].a
        pvars = Poincare.from_Simulation(sim, average)
        return Andoyer.from_Poincare(pvars, j, k, a10, i1, i2)

    def to_Simulation(self, masses=None, average=True):
        ''' By default assumes 2 planet system with self consistent jacobi masses
        (4 masses m1, M1, m2, M2 for 3 masses of star+2 planets may not be consistent).
        If 2 planets are part of larger system, need to pass physical masses=[M, m1, m2]
        '''
        pvars = self.to_Poincare()
        return pvars.to_Simulation(masses, average)

class AndoyerHamiltonian(Hamiltonian):
    pass
