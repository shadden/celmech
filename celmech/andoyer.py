import numpy as np
from sympy import symbols, S, cos
from celmech.hamiltonian import Hamiltonian
from celmech.poincare import PoincareParticle, Poincare
from celmech.disturbing_function import get_fg_coeffs
from celmech.transformations import ActionAngleToXY, XYToActionAngle
import rebound

'''
Compare with Hadden & Lithwick 2017:
In the code, Z = |scriptZ| and z = arg(scriptZ) from paper
W = |scriptW| and w = arg(scriptW) from paper
Phi and phi in the paper are a generalized eccentricity squared and a generalized pericenter
Phi in the code = Phi in the paper / k and scaled to simplify the Hamiltonian, so also a squared eccentricity
Cartesian components in code X, Y = sqrt(2*phi)cos(phi) and sin(phi) are generalized eccentricities
Note A is proportional to scriptZ, but B is not proportional to scriptW, and depends on masses etc. Can solve for B from scriptZ and scriptW
'''

def rotate_Poincare_Gammas_To_AABB(Gamma1,gamma1,Gamma2,gamma2,f,g,inverse=False):
    if inverse is True:
        g = -g
    X1,Y1 = ActionAngleToXY(Gamma1,gamma1)
    X2,Y2 = ActionAngleToXY(Gamma2,gamma2)
    norm = np.sqrt(f*f + g*g)
    rotation_matrix = np.array([[f,g],[-g,f]]) / norm
    AX,BX = np.dot(rotation_matrix , np.array([X1,X2]) )
    AY,BY = np.dot(rotation_matrix , np.array([Y1,Y2]) )
    AA,a = XYToActionAngle(AX,AY)
    BB,b = XYToActionAngle(BX,BY)
    return AA,a,BB,b

def calc_expansion_params(G, masses, j, k, a10):
    p = {'j':j, 'k':k, 'G':G, 'masses':masses, 'a10':a10}       
    p['a20'] = a10*(j/float(j-k))**(2./3.)
    p['Lambda10'] = masses[1]*np.sqrt(G*masses[0]*p['a10'])
    p['Lambda20'] = masses[2]*np.sqrt(G*masses[0]*p['a20'])
    p['n10'] = masses[1]**3*(G*masses[0])**2/p['Lambda10']**3
    p['n20'] = masses[2]**3*(G*masses[0])**2/p['Lambda20']**3
    p['nu1'] = -3. * p['n10']/ p['Lambda10']
    p['nu2'] = -3. * p['n20'] / p['Lambda20']
    f,g = get_fg_coeffs(j,k)
    p['ff']  = np.sqrt(2)*f/np.sqrt(p['Lambda10'])
    p['gg']  = np.sqrt(2)*g/np.sqrt(p['Lambda20'])
    fac = np.sqrt(k*(p['ff']**2+p['gg']**2))**k
    p['Acoeff'] = p['nu1']*(j-k)**2 + p['nu2']*j**2
    p['Ccoeff'] = -G**2*masses[0]*masses[2]**3* masses[1]/p['Lambda20']**2*fac
    p['Phiscale'] = 2.**((k-6.)/(k-4.))*(p['Ccoeff']/p['Acoeff'])**(2./(4.-k))
    p['timescale'] = 8./(p['Phiscale']*p['Acoeff'])
    return p

def get_second_order_phiprime(Phi_eq):
    return (4*Phi_eq**2 - 2.)/3.

class Andoyer(object):
    def __init__(self, j, k, Phi, phi, a10=1., G=1., masses=[1.,1.e-5,1.e-5], BB=0., b=0., Phiprime=1.5, K=0., deltalambda=np.pi, lambda1=0.):
        sX, sY, sBB, sb, sPhiprime, sK, sdeltalambda, slambda1 = symbols('X, Y, BB, b, Phiprime, K, \Delta\lambda, lambda1')
        sk, sj, sG, smasses, sa10 = symbols('k, j, G, masses, a10')
        X = np.sqrt(2.*Phi)*np.cos(phi)
        Y = np.sqrt(2.*Phi)*np.sin(phi)
        self.X = X
        self.Y = Y
        self.BB = BB
        self.b = b
        self.Phiprime = Phiprime
        self.K = K
        self.deltalambda = deltalambda
        self.lambda1 = lambda1

        self.params = calc_expansion_params(G, masses, j, k, a10)

    @classmethod
    def new_from_elements(cls, j, k, Zstar, libfac, a1=1., G=1., masses=[1.,1.e-5,1.e-5], W=0., w=0., deltalambda=np.pi, lambda1=0.):
        a10 = a1

        p = calc_expansion_params(G, masses, j, k, a10)
        Xstar = -np.sqrt(2*Phistar)
        Phiprime = get_second_order_phiprime(Xstar)
        Phi = Phistar
        phi = np.pi

        return cls(j, k, Phi, phi, a10, G, masses, BB, b, Phiprime, K, deltalambda, lambda1)
    @classmethod
    def from_elements(cls, j, k, Phistar, libfac, a10=1., G=1., masses=[1.,1.e-5,1.e-5], BB=0., b=0., K=0., deltalambda=np.pi, lambda1=0.):
        p = calc_expansion_params(G, masses, j, k, a10)
        Xstar = -np.sqrt(2*Phistar)
        Phiprime = get_second_order_phiprime(Xstar)
        Phi = Phistar
        phi = np.pi

        return cls(j, k, Phi, phi, a10, G, masses, BB, b, Phiprime, K, deltalambda, lambda1)

    @classmethod
    def from_Poincare(cls,pvars,j,k,a10,i1=1,i2=2):
        p1 = pvars.particles[i1]
        p2 = pvars.particles[i2]
        masses = [pvars.M, p1.m, p2.m]
        p = calc_expansion_params(pvars.G, masses, j, k, a10)
        
        dL1 = p1.Lambda-p['Lambda10']
        dL2 = p2.Lambda-p['Lambda20']

        AA,a,BB,b = rotate_Poincare_Gammas_To_AABB(p1.Gamma,p1.gamma,p2.Gamma,p2.gamma,p['ff'],p['gg'])
        K  = dL2 + j * dL1 / float(j-k) 
        Brouwer = -dL1/(j-k) - AA / k
        bCoeff =  p['Acoeff'] * Brouwer + j * p['nu2'] * K

        # scale momenta
        AA /= p['Phiscale']
        K /= p['Phiscale']
        BB /= p['Phiscale']

        phi = j * p2.l - (j-k) * p1.l + k * a
        Phi = AA / k 
        Phiprime = - p['timescale'] * bCoeff / 3.
        
        andvars = cls(j, k, Phi, phi, a10, pvars.G, masses, BB, b, Phiprime, K, p2.l-p1.l, p1.l)
        return andvars

    def to_Poincare(self):
        p = self.params
        j = p['j']
        k = p['k']
       
        # Unscale momenta
        BB = self.BB*p['Phiscale']
        K = self.K*p['Phiscale']
        AA = k*self.Phi*p['Phiscale']
        bCoeff= -3. * self.Phiprime / p['timescale']

        Brouwer = (bCoeff - j * p['nu2'] * K) / p['Acoeff']
        lambda2 = self.lambda1 + self.deltalambda
        theta = j*self.deltalambda + k*self.lambda1 # jlambda2 - (j-k)lambda1
        a = np.mod( (self.phi - theta) / k ,2*np.pi)
        dL1 = -(j-k) * (Brouwer + AA / k)
        dL2 =((j-k) * K - j * dL1)/(j-k) 
        Lambda1 = p['Lambda10']+dL1
        Lambda2 = p['Lambda20']+dL2 

        Gamma1,gamma1,Gamma2,gamma2 = rotate_Poincare_Gammas_To_AABB(AA,a,BB,self.b,p['ff'],p['gg'], inverse=True)
        masses = p['masses']
        p1 = PoincareParticle(masses[1], Lambda1, self.lambda1, Gamma1, gamma1, masses[0])
        p2 = PoincareParticle(masses[2], Lambda2, lambda2, Gamma2, gamma2, masses[0])
        return Poincare(p['G'], masses[0], [p1,p2])

    @classmethod
    def from_Simulation(cls, sim, j, k, a10=None, i1=1, i2=2, average_synodic_terms=False):
        if a10 is None:
            a10 = sim.particles[i1].a
        pvars = Poincare.from_Simulation(sim, average_synodic_terms)
        ps = sim.particles
        return Andoyer.from_Poincare(pvars, j, k, a10, i1, i2)


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
    def Brouwer(self):
        p = self.params
        return -3.*self.Phiprime/p['Acoeff']/p['timescale']

class AndoyerHamiltonian(Hamiltonian):
    def __init__(self, andvars):
        X, Y, Phi, phi, Phiprime, k = symbols('X, Y, Phi, phi, Phiprime, k')
        pqpairs = [(X, Y)]
        Hparams = {Phiprime:andvars.Phiprime, k:andvars.params['k']}

        H = (X**2 + Y**2)**2 - S(3)/S(2)*Phiprime*(X**2 + Y**2) + (X**2 + Y**2)**((k-S(1))/S(2))*X
        if andvars.params['timescale'] < 0:
            H *= -1
            #andvars.params['timescale'] *= -1

        super(AndoyerHamiltonian, self).__init__(H, pqpairs, Hparams, andvars)  
        self.Hpolar = 4*Phi**2 - S(3)*Phiprime*Phi + (2*Phi)**(k/S(2))*cos(phi)

    def state_to_list(self, state):
        return [state.X, state.Y] 

    def update_state_from_list(self, state, y):
        state.X = y[0]
        state.Y = y[1]
