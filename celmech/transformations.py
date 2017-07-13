from celmech import disturbing_function
import numpy as np
from collections import OrderedDict
from sympy import S
from celmech.disturbing_function import laplace_coefficient 
from celmech.disturbing_function import get_fg_coeffs
from itertools import combinations
import rebound

'''
class cartesianvars():
    def __init__(self, N):
        self.__dict = {0s}
        self.scales = {'p':1, 'q':1, 't':1}

def cart_to_poin(G, m, cart)

def poin_to_cart(G, m, cart)



class vars():
    def __init__
    def from_sim(G, physical masses, cartesianvars, scales)
    def to_sim (G, phyisical masses, poincarevars, scales)
    def set
    def

class andoyer():
    def from_sim(G, physical masses, cartesianvars, scales)
    def to_sim(G, physical masses, andoyervars, scales)
'''
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

def poincare_vars_from_sim2(sim, average_synodic_terms=False):
    ps = sim.particles
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    pvars = [OrderedDict() for i in range(sim.N)]
    for i in range(1,sim.N):
        Lambda = mjac[i]*np.sqrt(sim.G*Mjac[i]*ps[i].a)
        pvars[i]['Lambda'] = Lambda
        pvars[i]['lambda'] = ps[i].l
        pvars[i]['Gamma'] = Lambda*(1.-np.sqrt(1.-ps[i].e**2))
        pvars[i]['gamma'] = -ps[i].pomega
    if average_synodic_terms is True:    
        pairs = combinations(range(1,sim.N), 2)
        for i1, i2 in pairs:
            pvars[i1]['Lambda'], pvars[i2]['Lambda'] = synodic_Lambda_correction(sim, i1, i2, pvars[i1]['Lambda'], pvars[i2]['Lambda']) 
    return pvars

def poincare_vars_from_sim(sim, average_synodic_terms=False):
    ps = sim.particles
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
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
    mjac, Mjac = jacobi_masses(masses) 
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

def equib_andoyer_vars_from_sim(sim, j, k, a10, a20, i1=1, i2=2, average_synodic_terms=False):
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    poincare_vars = poincare_vars_from_sim(sim, average_synodic_terms)
    andvars, coeff = poincare_vars_to_andoyer_vars(poincare_vars, sim.G, Mjac[1], mjac[1], mjac[2], j, k,a10,a20)
    Phi, phi, W, w, B, K, deltalambda, lambda1 = andvars
    A,B,C = coeff
    Phiscale, timescale, Phiprime = get_equib_andoyer_params(A, B, C, k)
    return [Phi/Phiscale, phi, W/Phiscale, w, B/Phiscale, K/Phiscale, deltalambda, lambda1], [Phiscale, timescale, Phiprime]

def get_second_order_equilibrium(Phiprime):
    if Phiprime < -2./3.:
        raise AttributeError("Phiprime = {0}".format(Phiprime))
    return -0.5*np.sqrt(3*Phiprime + 2.)
def get_second_order_phiprime(Phi_eq):
    return (4*Phi_eq**2 - 2.)/3.

def andoyer_vars_from_sim(sim, j, k, a10, a20, i1=1, i2=2, average_synodic_terms=False):
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    poincare_vars = poincare_vars_from_sim(sim, average_synodic_terms)
    andvars, coeff = poincare_vars_to_andoyer_vars(poincare_vars, sim.G, Mjac[1], mjac[1], mjac[2], j, k,a10,a20)
    Phi, phi, W, w, B, K, deltalambda, lambda1 = andvars
    A,B,C = coeff
    Phiscale, timescale, Phiprime = get_andoyer_params(A, B, C, k)
    return [Phi/Phiscale, phi, W/Phiscale, w, B/Phiscale, K/Phiscale, deltalambda, lambda1], [Phiscale, timescale, Phiprime]

def andoyer_vars_from_sim2(sim, j, k, a10, a20, i1=1, i2=2, average_synodic_terms=False):
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    poincare_vars = poincare_vars_from_sim(sim, average_synodic_terms)
    Phi, phi, W, w, B, K, A, B, C = poincare_vars_to_andoyer_vars2(poincare_vars, sim.G, Mjac[1], mjac[1], mjac[2], j, k,a10,a20)
    Phiscale, timescale, Phiprime = get_andoyer_params(A, B, C, k)
    return [Phi/Phiscale, phi, W/Phiscale, w, B/Phiscale, K/Phiscale, A, B, C], [Phiscale, timescale, Phiprime]

def poincare_vars_to_andoyer_vars2(poincare_vars,G,Mstar,mIn,mOut,j,k,aIn0,aOut0):
    """
     Convert the poincare variables in Hamiltonian
       H_kep + eps * Hres
     to variables of a model Andoyer Hamiltonian for the j:j-k resonance:
       H(p,q) = (1/2) A * (p)^2 +B p + C sqrt(p)^k cos(q)
    """
    from celmech.disturbing_function import get_fg_coeffs
    Lambda10 = mIn*np.sqrt(G*Mstar*aIn0)
    Lambda20 = mOut*np.sqrt(G*Mstar*aOut0)
    Lambda1, lambda1, Gamma1, gamma1, Lambda2, lambda2, Gamma2, gamma2 = poincare_vars
    pratio_res = (j-k)/float(j)
    alpha = pratio_res**(2./3.)
    
    dL1,dL2 = Lambda1-Lambda10,Lambda2-Lambda20
    
    f,g = get_fg_coeffs(j,k)
    ff  = np.sqrt(2) * f / np.sqrt(Lambda10)
    gg  = np.sqrt(2) * g / np.sqrt(Lambda20)
    Z,z,W,w = Rotate_Poincare_Gammas_To_ZW(Gamma1,gamma1,Gamma2,gamma2,ff,gg)
    
    # Derivatives of mean motions w.r.t. Lambdas evaluated at Lambda0s
    n1 = mIn**3*(G*Mstar)**2/Lambda10**3
    n2 = mOut**3*(G*Mstar)**2/Lambda20**3
    Dn1DL1,Dn2DL2 = -3 * n1 / Lambda10, -3 * n2 / Lambda20
    K  = ( j * dL1 + (j-k) * dL2 ) / (j-k)
    Pa = -dL1 / (j-k) 
    Brouwer = Pa - Z/k
    
    Acoeff = Dn1DL1 * (j-k)**2 + Dn2DL2 * j**2
    Bcoeff = j * n2 - (j-k) * n1 + Acoeff * Brouwer
    Ccoeff = -1 * G**2 * Mstar * mOut**3 * mIn  / ( Lambda20**2 ) * ( np.sqrt(ff*ff+gg*gg)**k * np.sqrt(2*k)**k )
    phi = j * lambda2 - (j-k) * lambda1 + k * z
    Phi = Z / k 
    return [Phi,phi,W,w,Brouwer,K,Acoeff,Bcoeff,Ccoeff]


def poincare_vars_to_andoyer_vars(poincare_vars,G,Mstar,mIn,mOut,j,k,aIn0,aOut0):
    """
     Convert the poincare variables in Hamiltonian
       H_kep + eps * Hres
     to variables of a model Andoyer Hamiltonian for the j:j-k resonance:
       H(p,q) = (1/2) A * (p)^2 +B p + C sqrt(p)^k cos(q)
    """
    Lambda1, lambda1, Gamma1, gamma1, Lambda2, lambda2, Gamma2, gamma2 = poincare_vars
    
    Lambda10 = mIn*np.sqrt(G*Mstar*aIn0)
    Lambda20 = mOut*np.sqrt(G*Mstar*aOut0)
    n10 = mIn**3*(G*Mstar)**2/Lambda10**3
    n20 = mOut**3*(G*Mstar)**2/Lambda20**3
    # Derivatives of mean motions w.r.t. Lambdas evaluated at Lambda0s
    Dn1DL1,Dn2DL2 = -3 * n10 / Lambda10, -3 * n20 / Lambda20
    
    dL1,dL2 = Lambda1-Lambda10,Lambda2-Lambda20
    
    f,g = get_fg_coeffs(j,k)
    ff  = np.sqrt(2) * f / np.sqrt(Lambda10)
    gg  = np.sqrt(2) * g / np.sqrt(Lambda20)
    Z,z,W,w = Rotate_Poincare_Gammas_To_ZW(Gamma1,gamma1,Gamma2,gamma2,ff,gg)
    norm = np.sqrt(ff*ff+gg*gg)**k
    
    K  = ( j * dL1 + (j-k) * dL2 ) / (j-k)
    Pa = -dL1 / (j-k) 
    Brouwer = Pa - Z/k
    
    phi = j * lambda2 - (j-k) * lambda1 + k * z
    Phi = Z / k 
   
    #print(Phi)
    Acoeff = Dn1DL1 * (j-k)**2 + Dn2DL2 * j**2
    Bcoeff = j * n20 - (j-k) * n10 + Acoeff * Brouwer
    Ccoeff = -1 * G**2 * Mstar * mOut**3 * mIn  / ( Lambda20**2 ) * ( np.sqrt(ff*ff+gg*gg)**k * np.sqrt(2*k)**k )
    #print(Acoeff, Bcoeff, Ccoeff, norm)
    return [Phi,phi,W,w,Brouwer,K,lambda2-lambda1,lambda1],[Acoeff,Bcoeff,Ccoeff]

def my_andoyer_vars_to_poincare_vars(andvars, G, masses, a10, a20, j, k, Phiscale=1.):
    from celmech.disturbing_function import get_fg_coeffs
    Phi, phi, W, w, B, K, deltalambda, lambda1 = andvars
    Phi *= Phiscale
    W *= Phiscale
    B *= Phiscale
    K *= Phiscale

    lambda2 = lambda1 + deltalambda
    Z = k*Phi
    theta = j*deltalambda + k*lambda1 # jlambda2 - (j-k)lambda1
    z = np.mod( (phi - theta) / k ,2*np.pi)
    Pa = B + Z/float(k)
    dL1 = -Pa*(j-k)    
    dL2 =((j-k) * K - j * dL1)/(j-k) 
   
    Lambda10 = masses[1]*np.sqrt(G*masses[0]*a10)
    Lambda20 = masses[2]*np.sqrt(G*masses[0]*a20)
    Lambda0s = [Lambda10, Lambda20]
    print(Lambda10, dL1, Lambda20, dL2)
    Lambda1,Lambda2 = Lambda0s[0]+dL1, Lambda0s[1]+dL2 

    from celmech.disturbing_function import get_fg_coeffs
    f,g = get_fg_coeffs(j,k)
    ff  = np.sqrt(2) * f / np.sqrt(Lambda0s[0])
    gg  = np.sqrt(2) * g / np.sqrt(Lambda0s[1])
    Gamma1,gamma1,Gamma2,gamma2 = Rotate_ZW_To_Poincare_Gammas(Z,z,W,w,ff,gg)

   # Derivatives of mean motions w.r.t. Lambdas evaluated at Lambda0s
    return [ Lambda1, lambda1, Gamma1, gamma1, Lambda2, lambda2, Gamma2, gamma2 ]

def setup_res(G, masses, j, k, Zstar, a10=1., libfac=0., W=0., w=0., K=0.,deltalambda=np.pi, lambda1=0.):
    Z = Zstar
    Phi=Z/k
    phi = np.pi
    a20 = a10*(j/(j-k))**(2./3.)
    Lambda10 = masses[1]*np.sqrt(G*masses[0]*a10)
    Lambda20 = masses[2]*np.sqrt(G*masses[0]*a20)
    Lambda0s = [Lambda10, Lambda20]
    n10 = masses[1]**3*(G*masses[0])**2/Lambda10**3
    n20 = masses[2]**3*(G*masses[0])**2/Lambda20**3
    Dn1DL1,Dn2DL2 = -3 * n10 / Lambda10, -3 * n20 / Lambda20
    f,g = get_fg_coeffs(j,k)
    ff  = np.sqrt(2) * f / np.sqrt(Lambda10)
    gg  = np.sqrt(2) * g / np.sqrt(Lambda20)
    norm = np.sqrt(ff*ff+gg*gg)**k

    print(norm, np.sqrt(2*k)**k)
    Acoeff = Dn1DL1 * (j-k)**2 + Dn2DL2 * j**2
    Ccoeff = -1 * G**2 * masses[0] * masses[2]**3 * masses[1]  / ( Lambda20**2 ) * ( np.sqrt(ff*ff+gg*gg)**k * np.sqrt(2*k)**k )
    Phiscale = 2.**((k-6.)/(k-4.))*(Ccoeff / Acoeff)**(2./(4.-k))
    print(Acoeff, Ccoeff, Phiscale)
    timescale = 8./(Phiscale*Acoeff)

    Phiprime = get_second_order_phiprime(np.sqrt(2*Zstar/k))
    Brouwer = -3.*Phiprime/timescale

    andvars=[Phi,phi,W,w,Brouwer,K,deltalambda,lambda1]
    pvars = my_andoyer_vars_to_poincare_vars(andvars, G, masses, a10, a20, j, k, Phiscale) 
    #print(pvars)
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

def get_andoyer_params(A,B,C,k):
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

def Rotate_Poincare_Gammas_To_ZW(Gamma1,gamma1,Gamma2,gamma2,f,g):
        X1,Y1 = ActionAngleToXY(Gamma1,gamma1)
        X2,Y2 = ActionAngleToXY(Gamma2,gamma2)
        norm = np.sqrt(f*f + g*g)
        rotation_matrix = np.array([[f,g],[-g,f]]) / norm
        ZX,WX = np.dot(rotation_matrix , np.array([X1,X2]) )
        ZY,WY = np.dot(rotation_matrix , np.array([Y1,Y2]) )
        Z,z = XYToActionAngle(ZX,ZY)
        W,w = XYToActionAngle(WX,WY)
        return Z,z,W,w

def Rotate_ZW_To_Poincare_Gammas(Z,z,W,w,f,g):
        ZX,ZY = ActionAngleToXY(Z,z)
        WX,WY = ActionAngleToXY(W,w)
        norm = np.sqrt(f*f + g*g)
        rotation_matrix = np.array([[f,-g],[g,f]]) / norm 
        X1,X2 = np.dot(rotation_matrix , np.array([ZX,WX]) )
        Y1,Y2 = np.dot(rotation_matrix , np.array([ZY,WY]) )
        Gamma1,gamma1 = XYToActionAngle(X1,Y1)
        Gamma2,gamma2 = XYToActionAngle(X2,Y2)
        return Gamma1,gamma1,Gamma2,gamma2

def andoyer_vars_to_sim(andoyer_vars,G,Mstar,mIn,mOut,n1,n2,j,k,aIn0,aOut0,lambda0s=(0,0),actionScale=None):
    pvars = andoyer_vars_to_poincare_vars(andoyer_vars,G,Mstar,mIn,mOut,n1,n2,j,k,aIn0,aOut0,lambda0s=(0,0),actionScale=None)
    return poincare_vars_to_sim(pvars, G, [Mstar, mIn, mOut])
'''
def my_andoyer_vars_to_poincare_vars(andvars, G, masses, a10, a20, j, k):
    from celmech.disturbing_function import get_fg_coeffs
    Phi, phi, W, w, B, K, deltalambda, lambda1 = andvars
    lambda2 = lambda1 + deltalambda
    Z = k*Phi
    theta = j*deltalambda + k*lambda1 # jlambda2 - (j-k)lambda1
    z = np.mod( (phi - theta) / k ,2*np.pi)
    Pa = B + Z/float(k)
    dL1 = -Pa*(j-k)    
    dL2 =((j-k) * K - j * dL1)/(j-k) 
   
    Lambda10 = masses[1]*np.sqrt(G*masses[0]*a10)
    Lambda20 = masses[2]*np.sqrt(G*masses[0]*a20)
    Lambda0s = [Lambda10, Lambda20]
    Lambda1,Lambda2 = Lambda0s[0]+dL1, Lambda0s[1]+dL2 

    from celmech.disturbing_function import get_fg_coeffs
    f,g = get_fg_coeffs(j,k)
    ff  = np.sqrt(2) * f / np.sqrt(Lambda0s[0])
    gg  = np.sqrt(2) * g / np.sqrt(Lambda0s[1])
    Gamma1,gamma1,Gamma2,gamma2 = Rotate_ZW_To_Poincare_Gammas(Z,z,W,w,ff,gg)

   # Derivatives of mean motions w.r.t. Lambdas evaluated at Lambda0s
    return [ Lambda1, lambda1, Gamma1, gamma1, Lambda2, lambda2, Gamma2, gamma2 ]
'''
def andoyer_vars_to_poincare_vars(andoyer_vars,G,Mstar,mIn,mOut,n1,n2,j,k,aIn0,aOut0,actionScale=None):
    """
     Convert the poincare variables in Hamiltonian
       H_kep + eps * Hres
     to variables of a model Andoyer Hamiltonian for the j:j-k resonance:
       H(p,q) = (1/2) A * (p)^2 +B p + C sqrt(p)^k cos(q)
    """
    from celmech.disturbing_function import get_fg_coeffs
    P,Q,W,w,Brouwer,K,deltalambda,lambda1 = andoyer_vars
    lambda2 = lambda1 + deltalambda
    if actionScale is None:
        actionScale = 1.
    P,W,Brouwer,K = np.array([P,W,Brouwer,K]) * actionScale
    Z = k*P
    # ! Need to specify lambdas
    z = np.mod( (Q - j * lambda2 + (j-k)*lambda1) / k ,2*np.pi)
    Pa = (k*Brouwer + Z) / float(k)
    dL1 = -Pa*(j-k)    
    dL2 =((j-k) * K - j * dL1)/(j-k) 
    
    Lambda10 = mIn*np.sqrt(G*Mstar*aIn0)
    Lambda20 = mOut*np.sqrt(G*Mstar*aOut0)
    Lambda0s = [Lambda10, Lambda20]
    Lambda1,Lambda2 = Lambda0s[0]+dL1, Lambda0s[1]+dL2 
    from celmech.disturbing_function import get_fg_coeffs
    f,g = get_fg_coeffs(j,k)
    ff  = np.sqrt(2) * f / np.sqrt(Lambda0s[0])
    gg  = np.sqrt(2) * g / np.sqrt(Lambda0s[1])
    norm = np.sqrt(ff*ff+gg*gg)
    Gamma1,gamma1,Gamma2,gamma2 = Rotate_Poincare_Gammas_To_ZW(Z,z,W,w,ff,-gg)#Rotate_ZW_To_Poincare_Gammas(Z,z,W,w,ff,gg)

   # Derivatives of mean motions w.r.t. Lambdas evaluated at Lambda0s
    return [ Lambda1, lambda1, Gamma1, gamma1, Lambda2, lambda2, Gamma2, gamma2 ]

