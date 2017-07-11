from celmech import disturbing_function
import numpy as np
from sympy import S
from celmech.disturbing_function import laplace_coefficient 
from itertools import combinations

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

def andoyer_vars_from_sim(sim, j, k, a10, a20, i1=1, i2=2, average_synodic_terms=False):
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    poincare_vars = poincare_vars_from_sim(sim, average_synodic_terms)
    Phi, phi, W, w, B, K, A, B, C = poincare_vars_to_andoyer_vars(poincare_vars, sim.G, Mjac[1], mjac[1], mjac[2], j, k,a10,a20)
    Phiscale, timescale, Phiprime = get_andoyer_params(A, B, C, k)
    return [Phi/Phiscale, phi, W/Phiscale, w, B/Phiscale, K/Phiscale, A, B, C], [Phiscale, timescale, Phiprime]

def poincare_vars_to_andoyer_vars(poincare_vars,G,Mstar,mIn,mOut,j,k,aIn0,aOut0):
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
        
def my_andoyer_vars_to_poincare_vars(Phi, phi, W, w, B, a20, deltalambda, lambda1, j, k):
    from celmech.disturbing_function import get_fg_coeffs
    lambda1,lambda2 = lambda0s     
    Z = k*Phi
    theta = j*deltalambda + k*lambda1 # jlambda2 - (j-k)lambda1
    z = np.mod( (phi - theta) / k ,2*np.pi)
    Pa = B + Z/float(k)
    dL1 = -Pa*(j-k)    
    dL2 =((j-k) * K - j * dL1)/(j-k) 
    
    if Lambda0s is None:
        Lambda0s=(0,0)
    Lambda1,Lambda2 = Lambda0s[0]+dL1, Lambda0s[1]+dL2 

    from celmech.disturbing_function import get_fg_coeffs
    f,g = get_fg_coeffs(j,k)
    ff  = np.sqrt(2) * f / np.sqrt(Lambda0s[0])
    gg  = np.sqrt(2) * g / np.sqrt(Lambda0s[1])
    Gamma1,gamma1,Gamma2,gamma2 = Rotate_ZW_To_Poincare_Gammas(Z,z,W,w,ff,gg)

   # Derivatives of mean motions w.r.t. Lambdas evaluated at Lambda0s
    return [ Lambda1, lambda1, Gamma1, gamma1, Lambda2, lambda2, Gamma2, gamma2 ]
def andoyer_vars_to_poincare_vars(andoyer_vars,G,Mstar,mIn,mOut,n1,n2,j,k,aIn0,aOut0,lambda0s=(0,0),actionScale=None):
    """
     Convert the poincare variables in Hamiltonian
       H_kep + eps * Hres
     to variables of a model Andoyer Hamiltonian for the j:j-k resonance:
       H(p,q) = (1/2) A * (p)^2 +B p + C sqrt(p)^k cos(q)
    """
    from celmech.disturbing_function import get_fg_coeffs
    P,Q,W,w,Brouwer,K,Acoeff,Bcoeff,Coeff = andoyer_vars
    if actionScale is None:
        actionScale = 1.
    P,W,Brouwer,K = np.array([P,W,Brouwer,K]) * actionScale
    lambda1,lambda2 = lambda0s     
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

# Functions for solving for the equilibrium points of first-order MMRs
def first_order_equilibrium_p(p1):
    """
    Determine the value, p*, of stable equilibirium point of hamiltonian 
        h(p,q) = (1/2) * (p-p1)^2 + sqrt(p) * cos(q)
    so that dh(p*,pi)/dp = 0.
    """
    alpha = 2**(4./3.) *  p1 / 3.
    assert alpha >= 1.0 , "No separatrix present for 'p1' value"
    xstar = np.sqrt(alpha) * np.cos( np.arccos(-1 * alpha**(-1.5) ) / 3. + 2 * np.pi / 3. ) 
    pstar =  xstar * xstar * 2**(2./3.)
    return pstar
def get_p1_value(p_eq):
    """
    Solve for the value of parameter p1 in hamiltonian
        h(p,q) = (1/2) * (p-p1)^2 + sqrt(p) * cos(q)
    such that p_eq is the (stable) equilibrium value of p
    """
    from scipy.optimize import brentq
    # Search range over p1
    p1min = 3. * 2**(-4./3.) + 1e-14
    p1max = 1.5 * p_eq
    # find p1 value that corresponds to equilibrium p_eq
    rootfn = lambda x: p_eq - first_order_equilibrium_p(x)
    return brentq(rootfn,p1min,p1max)
