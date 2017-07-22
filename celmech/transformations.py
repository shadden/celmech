from celmech import disturbing_function
import numpy as np
from collections import OrderedDict
from sympy import S
from celmech.disturbing_function import laplace_coefficient 
from celmech.disturbing_function import get_fg_coeffs
from itertools import combinations
import rebound

def jacobi_masses_from_sim(sim):
    ps = sim.particles
    mjac, Mjac, mu = np.zeros(sim.N), np.zeros(sim.N), np.zeros(sim.N)
    interior_mass = ps[0].m
    for i in range(1,sim.N):
        mjac[i] = ps[i].m*interior_mass/(ps[i].m+interior_mass) # reduced mass with interior mass
        Mjac[i] = ps[0].m*(ps[i].m+interior_mass)/interior_mass # mjac[i]*Mjac[i] always = ps[i].m*ps[0].m
        mu[i] = sim.G**2*Mjac[i]**2*mjac[i]**3 # Deck (2013) notation
    mjac[0] = interior_mass
    return list(mjac), list(Mjac), list(mu)

def jacobi_masses(masses):
    N = len(masses)
    mjac, Mjac, mu = np.zeros(N), np.zeros(N), np.zeros(N)
    interior_mass = masses[0]
    for i in range(1,N):
        mjac[i] = masses[i]*interior_mass/(masses[i]+interior_mass) # reduced mass with interior mass
        Mjac[i] = masses[0]*(masses[i]+interior_mass)/interior_mass # mjac[i]*Mjac[i] always = m[i]*m[0]
    mjac[0] = interior_mass
    return list(mjac), list(Mjac)

def poincare_vars_from_sim(sim, average_synodic_terms=False):
    ps = sim.particles
    #mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    mjac = [p.m for p in ps]
    Mjac = [ps[0].m]*sim.N
    pvars = []
    for i in range(1,sim.N):
        Lambda = mjac[i]*np.sqrt(sim.G*Mjac[i]*ps[i].a)
        pvars.append(Lambda)
        pvars.append(ps[i].l)                               # lambda
        pvars.append(Lambda*(1.-np.sqrt(1.-ps[i].e**2)))    # Gamma
        pvars.append(-ps[i].pomega)                         # gamma
    if average_synodic_terms is True:    
        pairs = combinations(range(1,sim.N), 2)
        for i1, i2 in pairs:
            pvars[4*(i1-1)], pvars[4*(i2-1)] = synodic_Lambda_correction(sim, i1, i2, pvars[4*(i1-1)], pvars[4*(i2-1)])

    return pvars

def poincare_vars_to_sim(pvars, G, masses, average_synodic_terms=False):
    #mjac, Mjac = jacobi_masses(masses) 
    mjac = masses
    Mjac = [masses[0]]*len(masses)
    sim = rebound.Simulation()
    sim.G = G
    sim.add(m=masses[0])
    for i in range(1, len(masses)):
        Lambda, l, Gamma, gamma = pvars[4*(i-1):4*i]
        a = Lambda**2/mjac[i]**2/G/Mjac[i]
        e = np.sqrt(1.-(1.-Gamma/Lambda)**2)
        sim.add(m=masses[i], a=a, e=e, pomega=-gamma, l=l)
    sim.move_to_com()
    return sim

def synodic_Lambda_correction(sim, i1, i2, Lambda1, Lambda2):
    """
    Do a canonical transformation to correct the Lambdas for the fact that we have implicitly
    averaged over all the synodic terms we do not include in the Hamiltonian.
    """
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    ps = sim.particles
    s=0
    alpha = ps[i1].a/ps[i2].a
    deltan = ps[i1].n-ps[i2].n
    prefac = mu[i2]/Lambda2**2*mjac[i1]/Mjac[i1]/deltan
    deltalambda = ps[i1].l - ps[i2].l
    for j in range(1,150):
        s += laplace_coefficient(0.5, j, 0, alpha)*np.cos(j*deltalambda)
    s -= alpha*np.cos(deltalambda)
    s *= prefac
    return [Lambda1-s, Lambda2+s]

def ActionAngleToXY(Action,angle):
        return np.sqrt(2*Action)*np.cos(angle),np.sqrt(2*Action)*np.sin(angle)

def XYToActionAngle(X,Y):
        return 0.5 * (X*X+Y*Y), np.arctan2(Y,X)

def equib_andoyer_vars_from_sim(sim, j, k, a10, i1=1, i2=2, average_synodic_terms=False):
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    ps = sim.particles
    poincare_vars = poincare_vars_from_sim(sim, average_synodic_terms)
    #andvars, coeff = poincare_vars_to_andoyer_vars(poincare_vars, sim.G, Mjac[1], mjac[1], mjac[2], j, k,a10)
    return poincare_vars_to_andoyer_vars(poincare_vars, sim.G, [ps[0].m, ps[1].m, ps[2].m], j, k,a10)

def get_second_order_equilibrium(Phiprime):
    if Phiprime < -2./3.:
        raise AttributeError("Phiprime = {0}".format(Phiprime))
    return -0.5*np.sqrt(3*Phiprime + 2.)
def get_second_order_phiprime(Phi_eq):
    return (4*Phi_eq**2 - 2.)/3.

def andoyer_vars_from_sim(sim, j, k, a10, a20, i1=1, i2=2, average_synodic_terms=False):
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    poincare_vars = poincare_vars_from_sim(sim, average_synodic_terms)
    andvars, coeff = poincare_vars_to_andoyer_vars(poincare_vars, sim.G, Mjac[1], mjac[1], mjac[2], j, k,a10)
    Phi, phi, W, w, Brouwer, K, deltalambda, lambda1 = andvars
    A,B,C = coeff
    Phiscale, timescale, Phiprime = get_andoyer_params(A, B, C, k)
    #print(Phi, Brouwer, '*')
    return [Phi/Phiscale, phi, W/Phiscale, w, Brouwer/Phiscale, K/Phiscale, deltalambda, lambda1], [Phiscale, timescale, Phiprime]

def poincare_vars_to_andoyer_vars(pvars,G,masses,j,k,a10):
    """
     Convert the poincare variables in Hamiltonian
       H_kep + eps * Hres
     to variables of a model Andoyer Hamiltonian for the j:j-k resonance:
       H(p,q) = (1/2) A * (p)^2 +B p + C sqrt(p)^k cos(q)
    """
    Lambda1, lambda1, Gamma1, gamma1, Lambda2, lambda2, Gamma2, gamma2 = pvars
    p = get_andoyer_params(G, masses, j, k, a10)
    
    dL1 = Lambda1-p['Lambda10']
    dL2 = Lambda2-p['Lambda20']
    Z,z,W,w = Rotate_Poincare_Gammas_To_ZW(Gamma1,gamma1,Gamma2,gamma2,p['ff'],p['gg'])
    K  = ( j * dL1 + (j-k) * dL2 ) / (j-k)
    Pa = -dL1 / (j-k) 
    Brouwer = Pa - Z/k
    phi = j * lambda2 - (j-k) * lambda1 + k * z
    Phi = Z / k 
     
    andvars = AndoyerVars(Phi*np.cos(phi), Phi*np.sin(phi), W*np.cos(w), W*np.sin(w), Brouwer, K, lambda2-lambda1, lambda1, a10)
    andvars.scale_actions(1./p['Phiscale']) 
    return andvars

def andoyer_vars_to_poincare_vars(v, G, masses, a10, j, k):
    p = get_andoyer_params(G, masses, j, k, a10)
    v.scale_actions(p['Phiscale'])
    lambda2 = v.lambda1 + v.deltalambda
    Z = k*v.Phi
    theta = j*v.deltalambda + k*v.lambda1 # jlambda2 - (j-k)lambda1
    z = np.mod( (v.phi - theta) / k ,2*np.pi)
    Pa = v.Brouwer + Z/float(k)
    dL1 = -Pa*(j-k)    
    dL2 =((j-k) * v.K - j * dL1)/(j-k) 

    Lambda1 = p['Lambda10']+dL1
    Lambda2 = p['Lambda20']+dL2 

    Gamma1,gamma1,Gamma2,gamma2 = Rotate_Poincare_Gammas_To_ZW(Z,z,v.W,v.w,p['ff'],p['gg'], inverse=True)
    return [ Lambda1, v.lambda1, Gamma1, gamma1, Lambda2, lambda2, Gamma2, gamma2 ]

# Phix Phiy Wx Wy K B deltalambda lambda1
class AndoyerVars(object):
    def __init__(self, Phix=0., Phiy=0., Wx=0., Wy=0., Brouwer=0.,K=0.,deltalambda=np.pi, lambda1=0., a10=1.):
        self.state = OrderedDict([('Phix', Phix), 
                        ('Phiy', Phiy), 
                        ('Wx', Wx), 
                        ('Wy', Wy), 
                        ('Brouwer', Brouwer), 
                        ('K', K), 
                        ('deltalambda', deltalambda), 
                        ('lambda1', lambda1)])

    def scale_actions(self, scale):
        for action in ['Phix', 'Phiy', 'Wx', 'Wy', 'Brouwer', 'K']:
            self.state[action] *= scale
    
    @property
    def Phix(self):
        return self.state['Phix'] 
    @property
    def Phiy(self):
        return self.state['Phiy'] 
    @property
    def Wx(self):
        return self.state['Wx'] 
    @property
    def Wy(self):
        return self.state['Wy'] 
    @property
    def Brouwer(self):
        return self.state['Brouwer'] 
    @property
    def K(self):
        return self.state['K'] 
    @property
    def deltalambda(self):
        return self.state['deltalambda'] 
    @property
    def lambda1(self):
        return self.state['lambda1'] 
    @property
    def Phi(self):
        return np.sqrt(self.state['Phix']**2 + self.state['Phiy']**2)
    @property
    def phi(self):
        return np.arctan2(self.state['Phiy'], self.state['Phix'])
    @property
    def W(self):
        return np.sqrt(self.state['Wx']**2 + self.state['Wy']**2)
    @property
    def w(self):
        return np.arctan2(self.state['Wy'], self.state['Wx'])

def andoyer_phiprime(andvars, G, masses, j, k, a10):
    p = get_andoyer_params(G, masses, j,k, a10)
    Bcoeff = p['Acoeff']*andvars.Brouwer
    return -Bcoeff*p['timescale']/3.

def get_andoyer_params(G, masses, j, k, a10):
    p = {'a10':a10}
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

def setup_res(G, masses, j, k, Phistar, a10=1., libfac=0., W=0., w=0., K=0.,deltalambda=np.pi, lambda1=0.):
    p = get_andoyer_params(G, masses, j,k, a10)
    Phi = Phistar
    Xstar = np.sqrt(2*Phistar)
    alpha = get_second_order_phiprime(Xstar)
    Bcoeff = -3.*alpha/p['timescale']
    Brouwer = Bcoeff/p['Acoeff']
    Brouwer /= p['Phiscale']
    phi = np.pi
    
    andvars=AndoyerVars(Phi*np.cos(phi), Phi*np.sin(phi), W*np.cos(w), W*np.sin(w), Brouwer, K, deltalambda, lambda1, a10)
    pvars = andoyer_vars_to_poincare_vars(andvars, G, masses, a10, j, k) 
    return poincare_vars_to_sim(pvars, G, masses)

def get_equib_andoyer_params(A,B,C,k):
    """
    Rescale momenta of the Hamiltonion
       H(p,q) = (1/2) A * (p)^2 +B p + C sqrt(p)^k cos(q)
    by factor eta such that it can be written as
       H(p,q) = (1/2) (p-p')^2 + sqrt(p)^k cos(q)  
    """
    Phiscale = 2.**((k-6.)/(k-4.))*(C / A)**(2./(4.-k))
    timescale = 8./(Phiscale*A)
    Phiprime = -B * timescale/3.
    return [Phiscale, timescale, Phiprime]

def get_andoyer_params2(A,B,C,k):
    """
    Rescale momenta of the Hamiltonion
       H(p,q) = (1/2) A * (p)^2 +B p + C sqrt(p)^k cos(q)
    by factor eta such that it can be written as
       H(p,q) = (1/2) (p-p')^2 + sqrt(p)^k cos(q)  
    """
    Phiscale = (C / A)**(2./(4.-k))
    timescale = 1./(Phiscale*A)
    Phiprime = -B * timescale
    return [Phiscale, timescale, Phiprime]

def Rotate_Poincare_Gammas_To_ZW(Gamma1,gamma1,Gamma2,gamma2,f,g,inverse=False):
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

def andoyer_vars_to_sim(andoyer_vars,G,Mstar,mIn,mOut,n1,n2,j,k,aIn0,aOut0,lambda0s=(0,0),actionScale=None):
    pvars = andoyer_vars_to_poincare_vars(andoyer_vars,G,Mstar,mIn,mOut,n1,n2,j,k,aIn0,aOut0,lambda0s=(0,0),actionScale=None)
    return poincare_vars_to_sim(pvars, G, [Mstar, mIn, mOut])
