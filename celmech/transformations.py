from celmech import disturbing_function
import numpy as np
from sympy import S
from celmech.disturbing_function import laplace_coefficient 

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

def poincare_vars_from_sim(sim):
    ps = sim.particles
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    pvars = []
    for i in range(1,sim.N):
        Lambda = mjac[i]*np.sqrt(sim.G*Mjac[i]*ps[i].a)
        pvars.append(Lambda)
        pvars.append(ps[i].l)                               # lambda
        pvars.append(Lambda*(1.-np.sqrt(1.-ps[i].e**2)))    # Gamma
        pvars.append(-ps[i].pomega)                         # gamma

    return pvars

def sim_to_poincare_params(sim, inner, outer, m):
    ps = sim.particles
    m1jac = ps[inner].m*ps[0].m/(ps[inner].m+ps[0].m) # jacobi masses are reduced masses with masses interior
    m2jac = ps[outer].m*(ps[inner].m+ps[0].m)/(ps[outer].m+ps[inner].m+ps[0].m)
    M1jac = ps[0].m+ps[inner].m # jacobi Ms must multiply the jacobi masses to give m0*mN (N=1,2), see Deck Eq 1
    M2jac = ps[0].m*(ps[0].m+ps[inner].m+ps[outer].m)/(ps[0].m+ps[inner].m)
    mu1 = sim.G**2*M1jac**2*m1jac**3
    mu2 = sim.G**2*M2jac**2*m2jac**3
    alpha_res = (float(m)/(m+1))**(2./3.)
    f27 = 1./2*(-2*(m+1)*laplace_coefficient(0.5, m+1, 0, alpha_res) - alpha_res*laplace_coefficient(0.5, m+1, 1, alpha_res))
    f31 = 1./2*((2*m+1)*laplace_coefficient(0.5, m, 0, alpha_res) + alpha_res*laplace_coefficient(0.5, m, 1, alpha_res))        
    return [m1jac, M1jac, mu1, mu2, m, f27, f31]

def sim_to_theta_vars(sim, inner, outer, m, average_synodic_terms=False, scales=None):
    var, params = sim_to_poincare_vars(sim, inner, outer, average_synodic_terms=average_synodic_terms)
    Theta = var['Lambda2']/(m+1)
    Theta1 = m/(m+1)*var['Lambda2'] + var['Lambda1']
    theta = (m+1)*var['lambda2'] - m*var['lambda1']
    theta1 = var['lambda1']
    
    actionscale = scales['actionscale']
    var =  {'Theta':Theta/actionscale, 'Theta1':Theta1/actionscale, 'theta':theta, 'theta1':theta1, 'Gamma1':var['Gamma1']/actionscale, 'Gamma2':var['Gamma2']/actionscale, 'gamma1':var['gamma1'], 'gamma2':var['gamma2']}
    return var, params, scales

def sim_to_theta_params(sim, inner, outer, m, average_synodic_terms=False, scales=None):
    var, params = sim_to_poincare(sim, inner, outer, average_synodic_terms=average_synodic_terms)
    params['m'] = m
    params['zeta'] = params['mu1']/params['mu2']
    Theta = var['Lambda2']/(m+1)
    Theta1 = m/(m+1)*var['Lambda2'] + var['Lambda1']
    theta = (m+1)*var['lambda2'] - m*var['lambda1']
    theta1 = var['lambda1']
    
    if scales is None:
        scales = {'actionscale':Theta1, 'timescale':Theta1**3/params['mu2']}
    
    actionscale = scales['actionscale']
    var =  {'Theta':Theta/actionscale, 'Theta1':Theta1/actionscale, 'theta':theta, 'theta1':theta1, 'Gamma1':var['Gamma1']/actionscale, 'Gamma2':var['Gamma2']/actionscale, 'gamma1':var['gamma1'], 'gamma2':var['gamma2']}
    return var, params, scales
def sim_to_theta_scales(sim, inner, outer, m, average_synodic_terms=True):
    m1jac, M1jac, mu1, mu2, m, f27, f31 = sim_to_poincare_params(sim, inner, outer, m)
    Lambda1, lambda1, Lambda2, lambda2, Gamma1, gamma1, Gamma2, gamma2 = sim_to_poincare_vars(sim, inner, outer, m, average_synodic_terms=average_synodic_terms)
    zeta = mu1/mu2
    Theta = var['Lambda2']/(m+1)
    Theta1 = m/(m+1)*var['Lambda2'] + var['Lambda1']
    theta = (m+1)*var['lambda2'] - m*var['lambda1']
    theta1 = var['lambda1']
     
    scales = {'actionscale':Theta1, 'timescale':Theta1**3/params['mu2']}
    
    actionscale = scales['actionscale']
    var =  {'Theta':Theta/actionscale, 'Theta1':Theta1/actionscale, 'theta':theta, 'theta1':theta1, 'Gamma1':var['Gamma1']/actionscale, 'Gamma2':var['Gamma2']/actionscale, 'gamma1':var['gamma1'], 'gamma2':var['gamma2']}
    return var, params, scales

def ActionAngleToXY(Action,angle):
        return np.sqrt(2*Action)*np.cos(angle),np.sqrt(2*Action)*np.sin(angle)

def XYToActionAngle(X,Y):
        return 0.5 * (X*X+Y*Y), np.arctan2(Y,X)

def andoyer_vars_from_sim(sim, j, k, a10, a20, average_synodic_terms=False):
    mjac, Mjac, mu = jacobi_masses_from_sim(sim)
    pvars = pointcare_vars_from_sim(sim)
    Phi, phi, W, w, B, K, A, B, C = poincare_vars_to_andoyer_vars(pvars, sim.G, sim.particles[0].m, mjac[1], mjac[2], j, k,a10,a20)
    Phiscale, timescale, Phiprime = get_andoyer_params(A, B, C, k)
    return [Phi/Phiscale, phi], [Phiscale, timescale]

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
    timescale = 1./(eta*A)
    Phiprime = -B * tScale
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
        
def andoyer_vars_to_poincare_vars(andoyer_vars,G,Mstar,mIn,mOut,n1,n2,j,k,Lambda0s=None,lambda0s=(0,0),actionScale=None):
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
