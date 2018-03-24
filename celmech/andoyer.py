import numpy as np
from sympy import symbols, S, cos
from celmech.hamiltonian import Hamiltonian
from celmech.poincare import PoincareParticle, Poincare
from celmech.disturbing_function import get_fg_coeffs
from celmech.transformations import ActionAngleToXY, XYToActionAngle, rotate_actions
import rebound

# will give you the phiprime that yields an equilibrium Xstar, always in the range of phiprime where there exists a separatrix
def get_Phiprime(k, Xstarres):
    if Xstarres >= 0:
        raise ValueError("Xstarres passed to get_Phiprime must be < 0")
    if k == 1:
        Phiprime = (4.*Xstarres**3 + 1.)/(3.*Xstarres)
    if k == 2:
        Phiprime = (4.*Xstarres**2 - 2.)/3.
    if k == 3:
        Phiprime = (4.*Xstarres**2 + 3.*Xstarres)/3.

    return Phiprime

def get_Xstarres(k, Phiprime): # res fixed point always exists even if there's no separatrix
    if k == 1:
        Xstarres = np.sqrt(Phiprime)*np.cos(1./3.*np.arccos(-Phiprime**(-1.5))+2.*np.pi/3.)
    if k == 2:
        Xstarres = -0.5*np.sqrt(3.*Phiprime+2.)
    if k == 3:
        Xstarres = (-3.-np.sqrt(9.+48.*Phiprime))/8.
    return Xstarres

def get_Xstarunstable(k, Phiprime):
    if k == 1:
        if Phiprime < 1.:
            raise ValueError("k=1 resonance has no unstable fixed point for Phiprime < 1")
        Xstarunstable = np.sqrt(Phiprime)*np.cos(1./3.*np.arccos(-Phiprime**(-1.5)))
    if k == 2:
        if Phiprime < -2./3.:
            raise ValueError("k=2 resonance has no unstable fixed point for Phiprime < -2/3")
        if Phiprime < 2./3.:
            Xstarunstable = 0.
        else:
            Xstarunstable = 0.5*np.sqrt(3.*Phiprime-2.)
    if k == 3:
        if Phiprime < -9./48.:
            raise ValueError("k=3 resonance has no unstable fixed point for Phiprime < -9/48")
        Xstarunstable = (-3.+np.sqrt(9.+48.*Phiprime))/8.

    return Xstarunstable        

def get_Hsep(k, Phiprime):
    Xu = get_Xstarunstable(k, Phiprime)
    Hsep = Xu**4 - 3.*Phiprime/2.*Xu**2 + np.abs(Xu)**(k-1)*Xu
    return Hsep

def get_Xsep(k, Phiprime):
    if k==1:
        if Phiprime < 1.:
            raise ValueError("k=1 resonance has no separatrix for Phiprime < 1")
        else:
            Xu = get_Xstarunstable(k, Phiprime)
            disc = np.sqrt(1.5*Phiprime - 2*Xu**2)
            Xouter = -Xu - disc
            Xinner = -Xu + disc
    if k==2:
        if Phiprime < -2./3.:
            raise ValueError("k=2 resonance has no separatrix for Phiprime < -2/3")
        else:
            Hsep = get_Hsep(k, Phiprime)
            b = (1.+1.5*Phiprime)
            disc = np.sqrt(b**2+4.*Hsep)
            Xouter = -np.sqrt((b+disc)/2.) # b >= 0 if Phiprime >= -2/3, so b + disc adds to give biggest absolute value
            Xinner = -np.sqrt((b-disc)/2.) # b >= 0 if Phiprime >= -2/3, so b - disc subtracts to give smaller absolute value
    if k==3:
        if Phiprime < -9./48.:
            raise ValueError("k=3 resonance has no separatrix for Phiprime < -9/48")
        else:
            Xu = get_Xstarunstable(k, Phiprime)
            disc = 1.+2*Xu
            Xouter = -(disc + np.sqrt(disc))/2.
            Xinner = -(disc - np.sqrt(disc))/2.
            #Xouter = -0.5*(1. + 2.*Xu + disc)
            #Xinner = -0.5*(1. + 2.*Xu - disc)
            
    return Xinner, Xouter

def get_Xplusminus(k, Phiprime):
    pass

class Andoyer(object):
    def __init__(self, j, k, X, Y, a10=1., G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1., Phiprime=1.5, Psi2=0., psi2=0., dKprime=0., theta=0., lambda1=0.):
        self.calc_params(j, k, a10, G, m1, M1, m2, M2)
        self.X = X
        self.Y = Y
        self.Psi2 = Psi2
        self.psi2 = psi2
        self.Phiprime = Phiprime
        self.dKprime = dKprime
        self.theta = theta
        self.lambda1 = lambda1
   
    @property
    def dK(self):
        return self.Kprime*self.params['K0']/self.params['eta']

    @property
    def Phi(self):
        Phi, phi = XYToActionAngle(self.X, self.Y)
        return Phi
    
    @property
    def phi(self):
        Phi, phi = XYToActionAngle(self.X, self.Y)
        return np.mod(phi, 2.*np.pi)

    @property
    def B(self):
        return self.Phiprime_to_B(self.Phiprime)
    
    @property
    def Zstar(self):
        Xstar = get_Xstarres(self.params['k'], self.Phiprime)
        Phistar = Xstar**2/2. 
        return self.Phi_to_Z(Phistar)
        
    @property
    def dP(self):
        return self.Psi1/self.params['k'] - self.B

    @property
    def dL1(self):
        p = self.params
        return p['eta']*(p['Lambda10']/p['K0']*self.dK - (p['j']-p['k'])*self.dP)

    @property
    def dL2(self):
        p = self.params
        return p['eta']*(p['Lambda20']/p['K0']*self.dK + p['j']*self.dP)
    
    @property
    def lambda2(self):
        p = self.params
        return np.mod((self.theta + (p['j']-p['k'])*self.lambda1)/p['j'],2*np.pi)
        
    @property
    def psi1(self):
        return np.mod((self.phi - self.theta)/self.params['k'] ,2*np.pi)
    
    @property
    def Psi1(self):
        return self.Phi*self.params['k']*self.params['Phi0']
   
    @property
    def Z(self):
        return np.sqrt(2.*self.Psi1)/np.sqrt(self.params['Zfac'])

    @property
    def ecom(self):
        return np.sqrt(2.*self.Psi2)/np.sqrt(self.params['Zfac'])/self.params['beta']
    
    @property
    def phiecom(self):
        return -self.psi2 + np.pi

    def calc_params(self, j, k, a10, G, m1, M1, m2, M2):
        self.params = {'j':j, 'k':k, 'a10':a10, 'G':G, 'm1':m1, 'M1':M1, 'm2':m2, 'M2':M2}
        p = self.params
        p['alpha'] = (M1/M2)**(1./3.)*(float(j-k)/j)**(2./3.) 
        p['a20'] = a10/p['alpha']
        p['Lambda10'] = m1*np.sqrt(G*M1*a10)
        p['Lambda20'] = m2*np.sqrt(G*M2*a10/p['alpha'])
        p['K0'] = (j-k)*p['Lambda20'] + j*p['Lambda10']
        p['eta'] = float(j-k)/(3*j)*p['Lambda10']*p['Lambda20']/p['K0']
        
        f,g = get_fg_coeffs(j,k)
        p['f'], p['g'] = f,g
        ff = f*np.sqrt(p['eta']/p['Lambda10'])
        gg = g*np.sqrt(p['eta']/p['Lambda20'])
        norm = np.sqrt(ff*ff + gg*gg)
        p['psirotmatrix'] = np.array([[ff,gg],[-gg,ff]]) / norm
        p['invpsirotmatrix'] = np.array([[ff,-gg],[gg,ff]]) / norm
        p['Zfac'] = (f**2 + g**2)/norm**2
        p['a'] = -0.5*(p['j']-p['k'])
        p['c'] = -m1/M2*p['Lambda20']/p['eta']*norm**k
        p['norm'] = norm
        p['Phi0'] = (4.*k**(k/2.)*p['c']/p['a'])**(2./(4.-k))
        p['tau'] = 4./(p['Phi0']*p['a'])
        p['beta'] = np.sqrt(f**2/(f**2 + g**2))*np.sqrt(p['Lambda20']/p['Lambda10'])*(m1 + m2)/m2
        p['C'] = -g/f*np.sqrt(p['alpha'])*np.sqrt(M1/M2)
    
    @classmethod
    def from_elements(cls, j, k, Zstar, libfac, a10=1., a1=None, G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1., ecom=0., phiecom=0., theta=0, lambda1=0.):
        andvars = cls(j, k, 0., 0., a10=a10, G=G, m1=m1, M1=M1, m2=m2, M2=M2, Psi2=0., psi2=0., dKprime=0., theta=theta, lambda1=lambda1)
        p = andvars.params
        Psi1star = 0.5*Zstar**2*p['Zfac']
        Phistar = Psi1star/k/p['Phi0']
        Xstar = -np.sqrt(2.*Phistar)
        andvars.Phiprime = get_Phiprime(k, Xstar)
        Xinner, Xouter = get_Xsep(k, andvars.Phiprime)
        if libfac > 0: # offset toward outer branch of separatrix
            andvars.X = Xstar - libfac*np.abs(Xstar-Xouter)
        else: # offset toward inner branch of separatrix
            andvars.X = Xstar - libfac*np.abs(Xstar-Xinner)

        if a1 is None:
            a1 = a10
        dL1hat = (np.sqrt(a1)-np.sqrt(a10))/np.sqrt(a10)
        dL2hat = dL1hat + (j-k)/(3.*j)*andvars.dP
        andvars.dKprime = ((j-k)*p['Lambda20']*dL2hat + j*p['Lambda10']*dL1hat)/p['K0']
        
        andvars.Psi2 = ecom**2/2.*p['Zfac']*p['beta']**2
        andvars.psi2 = -phiecom + np.pi
        
        return andvars
    
    @classmethod
    def from_Z(cls, j, k, Z, phi, Zstar, a10=1., G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1., Psi2=0., psi2=0., dK=0., theta=0, lambda1=0.):
        andvars = cls(j, k, 0., 0., a10=a10, G=G, m1=m1, M1=M1, m2=m2, M2=M2, Psi2=Psi2, psi2=psi2, dK=dK, theta=theta, lambda1=lambda1)
        Phistar = andvars.Z_to_Phi(Zstar)
        Xstar = -np.sqrt(2.*Phistar)
        andvars.Phiprime = get_Phiprime(k, Xstar)
        
        Phi = andvars.Z_to_Phi(Z)
        andvars.X = np.sqrt(2.*Phi)*np.cos(phi)
        andvars.Y = np.sqrt(2.*Phi)*np.sin(phi)

        return andvars

    @classmethod
    def from_dP(cls, j, k, dP, phi, Zstar, a10=1., G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1., Psi2=0., psi2=0., dK=0., theta=0, lambda1=0.):
        andvars = cls(j, k, 0., 0., a10=a10, G=G, m1=m1, M1=M1, m2=m2, M2=M2, Psi2=Psi2, psi2=psi2, dK=dK, theta=theta, lambda1=lambda1)
        Phistar = andvars.Z_to_Phi(Zstar)
        Xstar = -np.sqrt(2.*Phistar)
        andvars.Phiprime = get_Phiprime(k, Xstar)

        Phi = andvars.dP_to_Phi(dP, andvars.Phiprime)
        andvars.X = np.sqrt(2.*Phi)*np.cos(phi)
        andvars.Y = np.sqrt(2.*Phi)*np.sin(phi)

        return andvars
    
    @classmethod
    def from_Poincare(cls, pvars, j, k, a10, i1=1, i2=2):
        p1 = pvars.particles[i1]
        p2 = pvars.particles[i2]
        andvars = cls(j, k, 0., 0., a10=a10, G=p1.G, m1=p1.m, M1=p1.M, m2=p2.m, M2=p2.M)
        
        p = andvars.params
        dL1 = p1.Lambda-p['Lambda10']
        dL2 = p2.Lambda-p['Lambda20']

        andvars.dK = ((p['j']-p['k'])*dL2 + p['j']*dL1)/p['eta']
        dP = (p['Lambda10']*dL2 - p['Lambda20']*dL1)/p['K0']/p['eta']
        
        andvars.theta = j*p2.l - (j-k)*p1.l
        andvars.lambda1 = p1.l
        
        Psi1,psi1,andvars.Psi2,andvars.psi2 = rotate_actions(p1.Gamma/p['eta'],p1.gamma,p2.Gamma/p['eta'],p2.gamma, p['psirotmatrix'])
        B = Psi1/k - dP
        Phi = Psi1/k/p['Phi0']
        phi = andvars.theta + k*psi1

        andvars.X = np.sqrt(2*Phi)*np.cos(phi)
        andvars.Y = np.sqrt(2*Phi)*np.sin(phi)
        andvars.Phiprime = (8.*B/3./p['Phi0'])
        
        return andvars

    def to_Poincare(self):
        p = self.params
       
        Lambda1 = p['Lambda10'] + self.dL1
        Lambda2 = p['Lambda20'] + self.dL2

        Gamma1,gamma1,Gamma2,gamma2 = rotate_actions(self.Psi1,self.psi1,self.Psi2,self.psi2,p['invpsirotmatrix'])
        Gamma1 *= p['eta']
        Gamma2 *= p['eta']
        
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

    def Z_to_Phi(self, Z):
        p = self.params
        Psi1 = 0.5*Z**2*p['Zfac']
        Phi = Psi1/p['k']/p['Phi0']
        return Phi
    
    def Phi_to_Z(self, Phi):
        p = self.params
        Psi1 = Phi*p['Phi0']*p['k']
        Z = np.sqrt(2*Psi1)/np.sqrt(p['Zfac'])
        return Z 
    
    def dP_to_Phi(self, dP, Phiprime):
        p = self.params
        B = self.Phiprime_to_B(Phiprime)
        Phi = (dP + B)/p['Phi0']
        return Phi
    
    def Phi_to_dP(self, Phi, Phiprime):
        p = self.params
        B = self.Phiprime_to_B(Phiprime)
        dP = Phi*p['Phi0'] - B
        return dP 

    def Phiprime_to_B(self, Phiprime):
        return 3.*Phiprime*self.params['Phi0']/8.

class AndoyerHamiltonian(Hamiltonian):
    def __init__(self, andvars):
        X, Y, Phi, phi, Phiprime, k = symbols('X, Y, Phi, phi, Phiprime, k')
        pqpairs = [(X, Y)]
        Hparams = {Phiprime:andvars.Phiprime, k:andvars.params['k']}

        H = (X**2 + Y**2)**2 - S(3)/S(2)*Phiprime*(X**2 + Y**2) + (X**2 + Y**2)**((k-S(1))/S(2))*X
        if andvars.params['tau'] < 0:
            H *= -1

        super(AndoyerHamiltonian, self).__init__(H, pqpairs, Hparams, andvars)  
        self.Hpolar = 4*Phi**2 - S(3)*Phiprime*Phi + (2*Phi)**(k/S(2))*cos(phi)

    def state_to_list(self, state):
        return [state.X, state.Y] 

    def update_state_from_list(self, state, y):
        state.X = y[0]
        state.Y = y[1]
