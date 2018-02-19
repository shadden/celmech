import numpy as np
from sympy import symbols, S, cos
from celmech.hamiltonian import Hamiltonian
from celmech.poincare import PoincareParticle, Poincare
from celmech.disturbing_function import get_fg_coeffs
from celmech.transformations import ActionAngleToXY, XYToActionAngle, rotate_actions
import rebound

class Andoyer(object):
    def __init__(self, Phi, phi, Psi2, psi2, K, Brouwer, lambda1, theta, j, k, a10=1., G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1.):
        self.calc_params(j, k, a10, G, m1, M1, m2, M2)
        self.Phi = Phi
        self.phi = phi
        self.Psi2 = Psi2
        self.psi2 = psi2
        self.K = K
        self.Brouwer = Brouwer
        self.lambda1 = lambda1
        self.theta = theta

    @property
    def dL1(self):
        p = self.params
        return -(p['j']-p['k'])/(p['j']+p['beta']*(p['j']-p['k']))*self.dL + 1./(p['j']+p['beta']*(p['j']-p['k']))*self.dK

    @property
    def dL2(self):
        p = self.params
        return p['j']/(p['j']+p['beta']*(p['j']-p['k']))*self.dL + p['beta']/(p['j']+p['beta']*(p['j']-p['k']))*self.dK
    
    @property
    def dP(self):
        p = self.params
        prefac = p['beta']*(p['j']-p['k'])/3./p['j']
        return self.dL/prefac
        
    @property
    def lambda2(self):
        p = self.params
        return np.mod((self.theta + (p['j']-p['k'])*self.lambda1)/p['j'],2*np.pi)
        
    @property
    def psi1(self):
        return np.mod((self.phi - self.theta)/self.params['k'] ,2*np.pi)
    
    @property
    def Psi1(self):
        return self.Phi*self.params['k']
    
    def calc_params(self, j, k, a10, G, m1, M1, m2, M2):
        self.params = {'j':j, 'k':k, 'a10':a10, 'G':G, 'm1':m1, 'M1':M1, 'm2':m2, 'M2':M2}
        p = self.params
        p['beta'] = m2/m1*(M2/M1)**(2./3.)*(float(j)/(j-k))**(1./3.)
        p['Lambda10'] = m1*np.sqrt(G*M1*a10)
        p['Lambda20'] = p['beta']*p['Lambda10']
        
        f,g = get_fg_coeffs(j,k)
        p['f'], p['g'] = f,g
        ff = f/np.sqrt(p['Lambda10'])
        gg = g/np.sqrt(p['Lambda20'])
        norm = np.sqrt(f*f + g*g)
        p['psirotmatrix'] = np.array([[f,g],[-g,f]]) / norm
        p['invpsirotmatrix'] = np.array([[f,-g],[g,f]]) / norm
        p['Zfac'] = (f**2 + g**2)/(ff**2 + gg**2)
    
    @classmethod
    def from_Poincare(cls, pvars, j, k, a10, i1=1, i2=2):
        p1 = pvars.particles[i1]
        p2 = pvars.particles[i2]
        andvars = cls(p1.Lambda, p2.Lambda, p1.l, p2.l, p1.Gamma, p2.Gamma, p1.gamma, p2.gamma, j, k, a10, p1.G, p1.m, p1.M, p2.m, p2.M)
        
        p = andvars.params
        dL1 = (p1.Lambda-p['Lambda10'])/p['Lambda10']
        dL2 = (p2.Lambda-p['Lambda20'])/p['Lambda10']

        andvars.dK = (p['j']-p['k'])*dL2 + p['j']*dL1
        andvars.dL = dL2 - p['beta']*dL1
        
        andvars.theta = j*p2.l - (j-k)*p1.l
        andvars.lambda1 = p1.l
        
        Psi1,psi1,andvars.Psi2,andvars.psi2 = rotate_actions(p1.Gamma,p1.gamma,p2.Gamma,p2.gamma, p['psirotmatrix'])
        Brouwer = Psi1/k - andvars.dL
        andvars.Phi = Psi1/k
        andvars.phi = andvars.theta + k*psi1

        return andvars

    def to_Poincare(self):
        p = self.params
        
        Lambda1 = p['Lambda10'] + self.dL1*p['Lambda10']
        Lambda2 = p['Lambda20'] + self.dL2*p['Lambda10']

        Gamma1,gamma1,Gamma2,gamma2 = rotate_actions(self.Psi1,self.psi1,self.Psi2,self.psi2,p['invpsirotmatrix'])

        lambda1 = self.lambda1
        lambda2 = self.lambda2
        
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
