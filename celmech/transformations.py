def poincare_from_sim(sim, inner, outer, m, averaged=False):
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
    if averaged:
        deltan = 2*np.pi/ps[inner].P-2*np.pi/ps[outer].P
        prefac = mu2/Lambda2**2*ps[inner].m/ps[0].m/deltan
        s = transform(ps[inner].l, ps[outer].l, alpha, prefac)
    var =  {'Lambda1':Lambda1-s, 'Lambda2':Lambda2+s, 'lambda1':lambda1, 'lambda2':lambda2, 'Gamma1':Gamma1, 'Gamma2':Gamma2, 'gamma1':gamma1, 'gamma2':gamma2}
    params = {'m1':m1jac, 'M1':M1jac, 'mu1':mu1, 'mu2':mu2, 'alpha':alpha, 'm':m}
    return var, params

'''
def averaged_Lambda_correction(sim):
def transform(lambda1, lambda2, alpha, prefac):
    deltan = 2*np.pi/ps[inner].P-2*np.pi/ps[outer].P
    prefac = mu2/Lambda2**2*ps[inner].m/ps[0].m/deltan
    s=0
    for j in range(1,150):
        s += LaplaceCoefficient(0.5, j, alpha, 0)*np.cos(j*(lambda1-lambda2))
    s -= alpha*np.cos(lambda1-lambda2)
    s = s*prefac
    return s
nptransform = np.vectorize(transform)
'''

