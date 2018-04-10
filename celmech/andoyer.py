import numpy as np
from sympy import symbols, S, cos
from celmech.hamiltonian import Hamiltonian
from celmech.poincare import PoincareParticle, Poincare
from celmech.disturbing_function import get_fg_coeffs
from celmech.transformations import ActionAngleToXY, XYToActionAngle, pol_to_cart, cart_to_pol
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
        if Phiprime >= 1:
            Xstarres = np.sqrt(Phiprime)*np.cos(1./3.*np.arccos(-Phiprime**(-1.5))+2.*np.pi/3.)
        elif Phiprime > 0:
            Xstarres = np.sqrt(Phiprime)*np.cosh(1./3.*np.arccosh(Phiprime**(-1.5)))
        elif Phiprime < 0:
            Xstarres = np.sqrt(-Phiprime)*np.sinh(1./3.*np.arcsinh(-abs(Phiprime)**(-1.5)))
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
    def __init__(self, j, k, X, Y, a10=1., G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1., B=1.5, Zcom=0., phiZcom=0., dKprime=0., theta=0., theta1=0.):
        self.calc_params(j, k, a10, G, m1, M1, m2, M2)
        self.X = X
        self.Y = Y
        self.Zcom = Zcom
        self.phiZcom = phiZcom
        self.B = B
        self.dKprime = dKprime
        self.theta = theta
        self.theta1 = theta1
    
    @property
    def Phi(self):
        Phi, phi = XYToActionAngle(self.X, self.Y)
        return Phi
    
    @property
    def phi(self):
        Phi, phi = XYToActionAngle(self.X, self.Y)
        return np.mod(phi, 2.*np.pi)

    @property
    def Phiprime(self):
        return 8.*self.B/3.
    @Phiprime.setter
    def Phiprime(self, value):
        self.B = 3.*value/8.

    @property
    def Zstar(self):
        Xstar = get_Xstarres(self.params['k'], self.Phiprime)
        Phistar = Xstar**2/2. 
        return self.Phi_to_Z(Phistar)
        
    @property
    def dP(self):
        return (self.Phi-self.B)*self.params['Phi0']

    @property
    def lambda1(self):
        p = self.params
        return np.mod((p['j']*p['K0']*self.theta1 - p['m2']*p['sLambda20']*self.theta)/p['K0'],2*np.pi)
    
    @property
    def lambda2(self):
        p = self.params
        return np.mod(((p['j']-p['k'])*p['K0']*self.theta1 + p['m1']*p['sLambda10']*self.theta)/p['K0'],2*np.pi)
        
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
    def phiZ(self):
        return -self.psi1

    @property
    def Psi2(self):
        p = self.params
        return p['Zfac']*(p['C']*self.Zcom)**2/2.*p['K0']/p['eta']
    @property
    def psi2(self):
        return -(self.phiZcom + np.pi)
    
    @property
    def dL1hat(self):
        p = self.params
        return self.dKprime - (p['j']-p['k'])**2/3./p['j']*p['m2']*p['sLambda20']/p['K0']*self.dP

    @property
    def dL2hat(self):
        p = self.params
        return self.dKprime + (p['j']-p['k'])/3.*p['m1']*p['sLambda10']/p['K0']*self.dP
    
    @property
    def dK(self):
        return self.dKprime*self.params['K0']/self.params['eta']

    @property
    def wlib(self):
        return 2*(2*self.B)**(self.params['k']/4.)/self.params['tau']
    
    @property
    def tlib(self):
        return 2*np.pi/self.wlib

    def calc_params(self, j, k, a10, G, m1, M1, m2, M2):
        self.params = {'j':j, 'k':k, 'a10':a10, 'G':G, 'm1':m1, 'M1':M1, 'm2':m2, 'M2':M2}
        p = self.params
        p['alpha'] = (M1/M2)**(1./3.)*(float(j-k)/j)**(2./3.) 
        p['a20'] = a10/p['alpha']
        p['sLambda10'] = np.sqrt(G*M1*a10)
        p['sLambda20'] = np.sqrt(G*M2*a10/p['alpha'])
        p['K0'] = (j-k)*m2*p['sLambda20'] + j*m1*p['sLambda10']
        p['eta'] = float(j-k)/(3*j)*m1*m2*p['sLambda10']*p['sLambda20']/p['K0']
        
        f,g = get_fg_coeffs(j,k)
        p['f'], p['g'] = f,g
        norm = np.sqrt((j-k)/3./j)*np.sqrt((f**2*m2*p['sLambda20'] + g**2*m1*p['sLambda10'])/p['K0']) # np.sqrt(ff**2 + gg**2)
        p['Zfac'] = (f**2 + g**2)/norm**2
        p['C'] = -1./j*np.sqrt(g**2/(f**2 + g**2))*np.sqrt(float(j-k)/3./j)

        p['n0'] = 1./j*G**2*M1**2/p['sLambda10']**3
        p['a'] = -0.5*p['n0']*(p['j']-p['k'])**2
        p['c'] = -(j-k)*p['n0']*((j-k)/3./j)**((k-2.)/2.)*p['K0']/M2/np.sqrt(G*M1*a10)*((m2*p['sLambda20']*f**2 + m1*p['sLambda10']*g**2)/p['K0'])**(k/2.)
        p['Phi0'] = (4.*k**(k/2.)*p['c']/p['a'])**(2./(4.-k))
        p['tau'] = 4./(p['Phi0']*abs(p['a']))
    
    @classmethod
    def from_elements(cls, j, k, Zstar, libfac, a10=1., a1=None, G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1., Zcom=0., phiZcom=0., theta=0, theta1=0.):
        andvars = cls(j, k, 0., 0., a10=a10, G=G, m1=m1, M1=M1, m2=m2, M2=M2, Zcom=Zcom, phiZcom=phiZcom, dKprime=0., theta=theta, theta1=theta1)
        p = andvars.params
        Psi1star = 0.5*Zstar**2*p['Zfac']
        Phistar = Psi1star/k/p['Phi0']
        Xstar = -np.sqrt(2.*Phistar)
        andvars.Phiprime = get_Phiprime(k, Xstar)
        Xinner, Xouter = get_Xsep(k, andvars.Phiprime)
        #print(Zstar, Xstar, andvars.Phiprime, Xinner, Xouter)
        if libfac > 0: # offset toward outer branch of separatrix
            andvars.X = Xstar - libfac*np.abs(Xstar-Xouter)
        else: # offset toward inner branch of separatrix
            andvars.X = Xstar - libfac*np.abs(Xstar-Xinner)

        if a1 is None:
            a1 = a10
        dL1 = m1*np.sqrt(G*M1)*(np.sqrt(a1)-np.sqrt(a10))
        andvars.dKprime = p['eta']/p['m1']/p['sLambda10']/p['eta']*(dL1 + (j-k)*p['eta']*andvars.dP)
        
        return andvars
    
    @classmethod
    def from_Z(cls, j, k, Z, phi, Zstar, a10=1., G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1., Zcom=0., phiZcom=0., dKprime=0., theta=0, theta1=0.):
        andvars = cls(j, k, 0., 0., a10=a10, G=G, m1=m1, M1=M1, m2=m2, M2=M2, Zcom=Zcom, phiZcom=phiZcom, dKprime=dKprime, theta=theta, theta1=theta1)
        Phistar = andvars.Z_to_Phi(Zstar)
        Xstar = -np.sqrt(2.*Phistar)
        andvars.Phiprime = get_Phiprime(k, Xstar)
        
        Phi = andvars.Z_to_Phi(Z)
        andvars.X = np.sqrt(2.*Phi)*np.cos(phi)
        andvars.Y = np.sqrt(2.*Phi)*np.sin(phi)

        return andvars

    @classmethod
    def from_dP(cls, j, k, dP, phi, Zstar, a10=1., G=1., m1=1.e-5, M1=1., m2=1.e-5, M2=1., Zcom=0., phiZcom=0., dKprime=0., theta=0, theta1=0.):
        andvars = cls(j, k, 0., 0., a10=a10, G=G, m1=m1, M1=M1, m2=m2, M2=M2, Zcom=Zcom, phiZcom=phiZcom, dKprime=dKprime, theta=theta, theta1=theta1)
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
        dL1hat = (p1.sLambda-p['sLambda10'])/p['sLambda10']
        dL2hat = (p2.sLambda-p['sLambda20'])/p['sLambda20']

        dP = 3.*j/(j-k)*(dL2hat - dL1hat)
        andvars.dKprime = ((j-k)*p['m2']*p['sLambda20']*dL2hat + j*p['m1']*p['sLambda10']*dL1hat)/p['K0']
       
        andvars.theta = j*p2.l - (j-k)*p1.l
        andvars.theta1 = (p['m1']*p['sLambda10']*p1.l + p['m2']*p['sLambda20']*p2.l)/p['K0']
      
        Z, phiZ, andvars.Zcom, andvars.phiZcom = andvars.sGammas_to_Zs(p1.sGamma, p1.gamma, p2.sGamma, p2.gamma)
        Psi1 = p['Zfac']*Z**2/2.
        psi1 = -phiZ

        andvars.B = (Psi1/k - dP)/p['Phi0']
        Phi = Psi1/k/p['Phi0']
        phi = andvars.theta + k*psi1

        andvars.X = np.sqrt(2*Phi)*np.cos(phi)
        andvars.Y = np.sqrt(2*Phi)*np.sin(phi)
        
        return andvars

    def to_Poincare(self):
        p = self.params
        
        sLambda1 = p['sLambda10']*(1.+self.dL1hat)
        sLambda2 = p['sLambda20']*(1.+self.dL2hat)

        sGamma1, gamma1, sGamma2, gamma2 = self.Zs_to_sGammas(self.Z, self.phiZ, self.Zcom, self.phiZcom)
        
        pvars = Poincare(p['G'])
        pvars.add(m=p['m1'], sLambda=sLambda1, l=self.lambda1, sGamma=sGamma1, gamma=gamma1, M=p['M1'])
        pvars.add(m=p['m2'], sLambda=sLambda2, l=self.lambda2, sGamma=sGamma2, gamma=gamma2, M=p['M2'])

        return pvars

    @classmethod
    def from_Simulation(cls, sim, j, k, a10=None, i1=1, i2=2, average=True):
        if a10 is None:
            a10 = sim.particles[i1].a
        pvars = Poincare.from_Simulation(sim, average)
        #if average is True:
        #    pvars.average_resonant_terms(i1=i1, i2=i2, exclude=[[j,k]])
        
        return Andoyer.from_Poincare(pvars, j, k, a10, i1, i2)

    def to_Simulation(self, masses=None, average=True):
        ''' By default assumes 2 planet system with self consistent jacobi masses
        (4 masses m1, M1, m2, M2 for 3 masses of star+2 planets may not be consistent).
        If 2 planets are part of larger system, need to pass physical masses=[M, m1, m2]
        '''
        pvars = self.to_Poincare()
        #if average is True:
        #    pvars.average_resonant_terms(exclude=[[self.params['j'], self.params['k']]], inverse=True)
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
        B = 3.*Phiprime/8.
        Phi = (dP + B)/p['Phi0']
        return Phi
    
    def Phi_to_dP(self, Phi, Phiprime):
        p = self.params
        B = 3.*Phiprime/8.
        dP = Phi*p['Phi0'] - B
        return dP 

    def Zs_to_sGammas(self, Z, phiZ, Zcom, phiZcom):
        ZX, ZY = pol_to_cart(Z, phiZ)
        ZcomX, ZcomY = pol_to_cart(Zcom, phiZcom)
        p = self.params
        ratio = np.sqrt(p['M2']*p['a20']/p['M1']/p['a10'])
        m1, m2 = p['m1'], p['m2']
        f, g = p['f'], p['g']
        beta1 = abs(f/g)*ratio
        beta2 = float(p['j']-p['k'])/p['j']*ratio
        D = m1*g - f*beta1*m2
        N = np.sqrt(f**2 + g**2)
        sX1 = -np.sqrt(p['sLambda10'])/D*(beta1*m2*N*ZX - g*(m1+beta2*m2)*ZcomX)
        sY1 = -np.sqrt(p['sLambda10'])/D*(-beta1*m2*N*ZY + g*(m1+beta2*m2)*ZcomY)
        sX2 = np.sqrt(p['sLambda20'])/D*(m1*N*ZX - f*(m1+beta2*m2)*ZcomX)
        sY2 = np.sqrt(p['sLambda20'])/D*(-m1*N*ZY + f*(m1+beta2*m2)*ZcomY)
        sGamma1, gamma1 = XYToActionAngle(sX1, sY1)
        sGamma2, gamma2 = XYToActionAngle(sX2, sY2)
        return sGamma1, gamma1, sGamma2, gamma2

    def sGammas_to_Zs(self, sGamma1, gamma1, sGamma2, gamma2):
        sX1, sY1 = ActionAngleToXY(sGamma1, gamma1)
        sX2, sY2 = ActionAngleToXY(sGamma2, gamma2)
        p = self.params
        ratio = np.sqrt(p['M2']*p['a20']/p['M1']/p['a10'])
        m1, m2 = p['m1'], p['m2']
        f, g = p['f'], p['g']
        beta1 = abs(f/g)*ratio
        beta2 = float(p['j']-p['k'])/p['j']*ratio
        N = np.sqrt(f**2 + g**2)
        ZX = 1/N*(f/np.sqrt(p['sLambda10'])*sX1 + g/np.sqrt(p['sLambda20'])*sX2)
        ZY = 1/N*(-f/np.sqrt(p['sLambda10'])*sY1 - g/np.sqrt(p['sLambda20'])*sY2)
        ZcomX = 1/(m1 + beta2*m2)*(m1/np.sqrt(p['sLambda10'])*sX1 + beta1*m2/np.sqrt(p['sLambda20'])*sX2)
        ZcomY = 1/(m1 + beta2*m2)*(-m1/np.sqrt(p['sLambda10'])*sY1 - beta1*m2/np.sqrt(p['sLambda20'])*sY2)
        Z, phiZ = cart_to_pol(ZX, ZY)
        Zcom, phiZcom = cart_to_pol(ZcomX, ZcomY)
        return Z, phiZ, Zcom, phiZcom
    
class AndoyerHamiltonian(Hamiltonian):
    def __init__(self, andvars):
        X, Y, Phi, phi, B, theta, k = symbols('X, Y, Phi, phi, B, theta, k')
        pqpairs = [(X, Y), (B, theta)]
        p = andvars.params
        Hparams = {k:p['k']}

        H = -((X**2 + Y**2) - S(2)*B)**2 - (X**2 + Y**2)**((k-S(1))/S(2))*X # + n0*tau*dK*(1-1.5*eta*Phi0*dK/K0)
        
        super(AndoyerHamiltonian, self).__init__(H, pqpairs, Hparams, andvars)  
        self.Hpolar = -4*(Phi-B)**2 - (2*Phi)**(k/S(2))*cos(phi)

    def state_to_list(self, state):
        return [state.X, state.Y, state.B, -state.theta] # - because B is conjugate to -theta

    def update_state_from_list(self, state, y):
        p = state.params
        state.X = y[0]
        state.Y = y[1]
        state.B = y[2]
        state.theta = -y[3] # because B conjugate to -theta

        theta1dot = p['n0']*p['tau']*(1.-3.*p['Phi0']*state.dKprime)
        state.theta1 = theta1dot*self.integrator.t
