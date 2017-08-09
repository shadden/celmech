import numpy as np
from sympy import symbols, S, cos
from celmech.hamiltonian import Hamiltonian
from celmech.poincare import PoincareParticle, Poincare
from celmech.disturbing_function import get_fg_coeffs
from celmech.transformations import ActionAngleToXY, XYToActionAngle
import rebound

def rotate_Poincare_Gammas_To_ZW(Gamma1,gamma1,Gamma2,gamma2,f,g,inverse=False):
    if inverse is True:
        g = -g
    X1,Y1 = ActionAngleToXY(Gamma1,gamma1)
    X2,Y2 = ActionAngleToXY(Gamma2,gamma2)
    norm = np.sqrt(f*f + g*g)
    rotation_matrix = np.array([[f,g],[-g,f]]) / norm
    ZX,WX = np.dot(rotation_matrix , np.array([X1,X2]) )
    ZY,WY = np.dot(rotation_matrix , np.array([Y1,Y2]) )
    Z,z = XYToActionAngle(ZX,ZY)
    W,w = XYToActionAngle(WX,WY)
    return Z,z,W,w

def calc_expansion_params(G, masses, j, k, a10):
    p = {'j':j, 'k':k, 'G':G, 'masses':masses, 'a10':a10}       
    p['a20'] = a10*(j/(j-k))**(2./3.)
    p['Lambda10'] = masses[1]*np.sqrt(G*masses[0]*p['a10'])
    p['Lambda20'] = masses[2]*np.sqrt(G*masses[0]*p['a20'])
    p['n10'] = masses[1]**3*(G*masses[0])**2/p['Lambda10']**3
    p['n20'] = masses[2]**3*(G*masses[0])**2/p['Lambda20']**3
    Dn1DL1 = -3. * p['n10']/ p['Lambda10']
    Dn2DL2 = -3. * p['n20'] / p['Lambda20']
    f,g = get_fg_coeffs(j,k)
    p['ff']  = np.sqrt(2)*f/np.sqrt(p['Lambda10'])
    p['gg']  = np.sqrt(2)*g/np.sqrt(p['Lambda20'])
    fac = np.sqrt(2.*k*(p['ff']**2+p['gg']**2))**k
    p['Acoeff'] = Dn1DL1*(j-k)**2 + Dn2DL2*j**2
    p['Ccoeff'] = -G**2*masses[0]*masses[2]**3* masses[1]/p['Lambda20']**2*fac
    p['Phiscale'] = 2.**((k-6.)/(k-4.))*(p['Ccoeff']/p['Acoeff'])**(2./(4.-k))
    p['timescale'] = 8./(p['Phiscale']*p['Acoeff'])
    return p

def get_second_order_phiprime(Phi_eq):
    return (4*Phi_eq**2 - 2.)/3.

class Andoyer(object):
    def __init__(self, j, k, Phi, phi, a10=1., G=1., masses=[1.,1.e-5,1.e-5], Ws=0., w=0., Phiprime=1.5, Ks=0., deltalambda=np.pi, lambda1=0.):
        sX, sY, sWs, sw, sPhiprime, sKs, sdeltalambda, slambda1 = symbols('X, Y, Ws, w, Phiprime, Ks, \Delta\lambda, lambda1')
        sk, sj, sG, smasses, sa10 = symbols('k, j, G, masses, a10')
        X = np.sqrt(2.*Phi)*np.cos(phi)
        Y = np.sqrt(2.*Phi)*np.sin(phi)
        self.X = X
        self.Y = Y
        self.Ws = Ws
        self.w = w
        self.Phiprime = Phiprime
        self.Ks = Ks
        self.deltalambda = deltalambda
        self.lambda1 = lambda1

        self.params = calc_expansion_params(G, masses, j, k, a10)

    @classmethod
    def from_elements(cls, j, k, Phistar, libfac, a10=1., G=1., masses=[1.,1.e-5,1.e-5], W=0., w=0., K=0., deltalambda=np.pi, lambda1=0.):
        p = calc_expansion_params(G, masses, j, k, a10)
        Xstar = -np.sqrt(2*Phistar)
        Phiprime = get_second_order_phiprime(Xstar)
        Phi = Phistar
        phi = np.pi
        Ks = K/p['Phiscale']
        Ws = W/p['Phiscale']

        return cls(j, k, Phi, phi, a10, G, masses, Ws, w, Phiprime, Ks, deltalambda, lambda1)

    @classmethod
    def from_Poincare(cls,pvars,j,k,a10,i1=1,i2=2):
        p1 = pvars.particles[i1]
        p2 = pvars.particles[i2]
        masses = [pvars.M, p1.m, p2.m]
        p = calc_expansion_params(pvars.G, masses, j, k, a10)
        
        dL1 = p1.Lambda-p['Lambda10']
        dL2 = p2.Lambda-p['Lambda20']
        Z,z,W,w = rotate_Poincare_Gammas_To_ZW(p1.Gamma,p1.gamma,p2.Gamma,p2.gamma,p['ff'],p['gg'])
        K  = ( j * dL1 + (j-k) * dL2 ) / (j-k)
        Pa = -dL1 / (j-k) 
        Brouwer = Pa - Z/k
        phi = j * p2.l - (j-k) * p1.l + k * z
        Phi = Z / k / p['Phiscale']
        Ws = W/p['Phiscale']
        Ks = K/p['Phiscale']
        Phiprime = -Brouwer*p['Acoeff']*p['timescale']/3. 
        
        andvars = cls(j, k, Phi, phi, a10, pvars.G, masses, Ws, w, Phiprime, Ks, p2.l-p1.l, p1.l)
        return andvars

    @classmethod
    def from_Simulation(cls, sim, j, k, a10=None, i1=1, i2=2, average_synodic_terms=False):
        if a10 is None:
            a10 = sim.particles[i1].a
        pvars = Poincare.from_Simulation(sim, average_synodic_terms)
        ps = sim.particles
        return Andoyer.from_Poincare(pvars, j, k, a10, i1, i2)

    def to_Poincare(self):
        p = self.params
        j = p['j']
        k = p['k']
        W = self.Ws*p['Phiscale']
        K = self.Ks*p['Phiscale']
        Z = k*self.Phi*p['Phiscale']
        Brouwer = -3.*self.Phiprime/p['Acoeff']/p['timescale']
        lambda2 = self.lambda1 + self.deltalambda
        theta = j*self.deltalambda + k*self.lambda1 # jlambda2 - (j-k)lambda1
        z = np.mod( (self.phi - theta) / k ,2*np.pi)
        Pa = Brouwer + Z/float(k)
        dL1 = -Pa*(j-k)    
        dL2 =((j-k) * K - j * dL1)/(j-k) 

        Lambda1 = p['Lambda10']+dL1
        Lambda2 = p['Lambda20']+dL2 

        Gamma1,gamma1,Gamma2,gamma2 = rotate_Poincare_Gammas_To_ZW(Z,z,W,self.w,p['ff'],p['gg'], inverse=True)
        masses = p['masses']
        p1 = PoincareParticle(masses[1], Lambda1, self.lambda1, Gamma1, gamma1, masses[0])
        p2 = PoincareParticle(masses[2], Lambda2, lambda2, Gamma2, gamma2, masses[0])
        return Poincare(p['G'], masses[0], [p1,p2])

    def to_Simulation(self):
        pvars = self.to_Poincare()
        return pvars.to_Simulation()
    
    @property
    def Phi(self):
        return (self.X**2 + self.Y**2)/2.
    @property
    def phi(self):
        return np.arctan2(self.Y, self.X)
    @property
    def W(self):
        p = self.params
        return self.Ws*p['Phiscale']
    @property
    def K(self):
        p = self.params
        return self.Ks*p['Phiscale']
    @property
    def Brouwer(self):
        p = self.params
        return -3.*self.Phiprime/p['Acoeff']/p['timescale']

class AndoyerHamiltonian(Hamiltonian):
    def __init__(self, andvars):
        X, Y, Phi, phi, Phiprime, k = symbols('X, Y, Phi, phi, Phiprime, k')
        pqpairs = [(X, Y)]
        Hparams = {Phiprime:andvars.Phiprime, k:andvars.params['k']}
        H = (X**2 + Y**2)**2 - S(3)/S(2)*Phiprime*(X**2 + Y**2) + (X**2 + Y**2)**((k-S(1))/S(2))*X
        super(AndoyerHamiltonian, self).__init__(H, pqpairs, Hparams, andvars)  
        self.Hpolar = 4*Phi**2 - S(3)*Phiprime*Phi + (2*Phi)**(k/S(2))*cos(phi)

    def state_to_list(self, state):
        return [state.X, state.Y] 

    def update_state_from_list(self, state, y):
        state.X = y[0]
        state.Y = y[1]
