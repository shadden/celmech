from celmech import disturbing_function
import numpy as np
from sympy import S
from celmech.disturbing_function import laplace_coefficient 

def sim_to_poincare_vars(sim, inner, outer, m, average_synodic_terms=False):
    ps = sim.particles
    m1jac = ps[inner].m*ps[0].m/(ps[inner].m+ps[0].m) # jacobi masses are reduced masses with masses interior
    m2jac = ps[outer].m*(ps[inner].m+ps[0].m)/(ps[outer].m+ps[inner].m+ps[0].m)
    M1jac = ps[0].m+ps[inner].m # jacobi Ms must multiply the jacobi masses to give m0*mN (N=1,2), see Deck Eq 1
    M2jac = ps[0].m*(ps[0].m+ps[inner].m+ps[outer].m)/(ps[0].m+ps[inner].m)
    mu1 = sim.G**2*M1jac**2*m1jac**3
    mu2 = sim.G**2*M2jac**2*m2jac**3
    Lambda1 = m1jac*np.sqrt(sim.G*M1jac*ps[inner].a)
    Lambda2 = m2jac*np.sqrt(sim.G*M2jac*ps[outer].a)
    lambda1 = ps[inner].l
    lambda2 = ps[outer].l
    Gamma1 = Lambda1*(1.-np.sqrt(1.-ps[inner].e**2))
    Gamma2 = Lambda2*(1.-np.sqrt(1.-ps[outer].e**2))
    gamma1 = -ps[inner].pomega
    gamma2 = -ps[outer].pomega
    
    s=0
    alpha_res = (float(m)/(m+1))**(2./3.)
    if average_synodic_terms:
        deltan = ps[inner].n-ps[outer].n
        prefac = mu2/Lambda2**2*ps[inner].m/ps[0].m/deltan
        for j in range(1,150):
            s += disturbing_function.laplace_coefficient(0.5, j, 0, alpha_res)*np.cos(j*(lambda1-lambda2))
        s -= alpha_res*np.cos(lambda1-lambda2)
        s *= prefac
    return [Lambda1-s, lambda1, Lambda2+s, lambda2, Gamma1, gamma1, Gamma2, gamma2]

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
def sim_to_theta_scales(sim, inner, outer, m, average_synodic_terms=False, scales=None):
    m1jac, M1jac, mu1, mu2, m, f27, f31 = sim_to_poincare_params(sim, inner, outer, m, average_synodic_terms=average_synodic_terms)
    zeta = params['mu1']/params['mu2']
   
 
    if scales is None:
        scales = {'actionscale':Theta1, 'timescale':Theta1**3/params['mu2']}
    
    actionscale = scales['actionscale']
    var =  {'Theta':Theta/actionscale, 'Theta1':Theta1/actionscale, 'theta':theta, 'theta1':theta1, 'Gamma1':var['Gamma1']/actionscale, 'Gamma2':var['Gamma2']/actionscale, 'gamma1':var['gamma1'], 'gamma2':var['gamma2']}
    return var, params, scales

