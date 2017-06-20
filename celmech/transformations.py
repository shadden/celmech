from celmech import disturbing_function
import numpy as np

def sim_to_poincare(sim, inner, outer, average_synodic_terms=False):
    ps = sim.particles
    alpha = sim.particles[inner].a/sim.particles[outer].a
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
    if average_synodic_terms:
        deltan = ps[inner].n-ps[outer].n
        prefac = mu2/Lambda2**2*ps[inner].m/ps[0].m/deltan
        for j in range(1,150):
            s += disturbing_function.laplace_coefficient(0.5, j, 0, alpha)*np.cos(j*(lambda1-lambda2))
        s -= alpha*np.cos(lambda1-lambda2)
        s *= prefac
    var =  {'Lambda1':Lambda1-s, 'Lambda2':Lambda2+s, 'lambda1':lambda1, 'lambda2':lambda2, 'Gamma1':Gamma1, 'Gamma2':Gamma2, 'gamma1':gamma1, 'gamma2':gamma2}
    params = {'m1':m1jac, 'M1':M1jac, 'mu1':mu1, 'mu2':mu2}
    return var, params
