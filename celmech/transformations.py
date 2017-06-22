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

