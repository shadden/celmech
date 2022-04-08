import numpy as np
import rebound as rb
import reboundx as rbx

def set_time_step(sim,dtfactor):
    r"""
    Set the time step of a simulation to a fraction,``dtfactor``,
    of the minimum perihelion passage timescale, defined as
    
    .. math::
        \tau_{p} = P\sqrt{\frac{(1-e)^3}{1+e}}~.
    
    Generally, the time step of `WHFAST` should be set to 
    :math:`\lesssim\tau_p/20` in order to obtain reliable
    results (see :ref:`Wisdom 2015
    <https://ui.adsabs.harvard.edu/abs/2015AJ....150..127W/abstract>`)
    """
    ps=sim.particles[1:sim.N_real]
    tperi=np.min([p.P * (1-p.e)**1.5 / np.sqrt(1+p.e) for p in ps])
    dt = tperi * dtfactor
    sim.dt = dt
def set_min_distance(sim,rhillfactor):
    ps=sim.particles[1:sim.N_real]
    mstar = sim.particles[0].m
    rhill = np.min([ p.rhill for p in ps if p.m > 0])
    mindist = rhillfactor * rhill
    sim.exit_min_distance = mindist

def get_simarchive_integration_results(sa,coordinates='jacobi'):
    """
    Read a simulation archive and store orbital elements
    as arrays in a dictionary.

    Arguments
    ---------
    sa : rebound.SimulationArchive or str
     The simulation archive to read or the file name of the simulation
     archive file. Can also be a reboundx simulation archive.
    coordinates : str
        The coordinate system to use for calculating orbital elements. 
        This can be:
        | - 'jacobi' : Use Jacobi coordinates (including Jacobi masses)
        | - 'barycentric' : Use barycentric coordinates.
        | - 'heliocentric' : Use canonical heliocentric elements. 
        | The canonical cooridantes are heliocentric distance vectors.
        | The conjugate momenta are barycentric momenta.

    Returns
    -------
    sim_results : dict
        Dictionary containing time and orbital elements at each 
        snapshot of the simulation archive.
    """
    if type(sa) == str:
        sa = rb.SimulationArchive(sa)

    if type(sa) == rb.simulationarchive.SimulationArchive:
        return _get_rebound_simarchive_integration_results(sa,coordinates)
    elif type(sa) == rbx.simulationarchive.SimulationArchive:
        return _get_reboundx_simarchive_integration_results(sa,coordinates)
    raise TypeError("{} is not a rebound or reboundx simulation archive!".format(sa))

def _get_rebound_simarchive_integration_results(sa,coordinates):
    if coordinates == 'jacobi':
        get_orbits = lambda sim: sim.calculate_orbits(jacobi_masses=True)
    elif coordinates == 'heliocentric':
        get_orbits = get_canonical_heliocentric_orbits
    elif coordinates == 'barycentric':
        get_orbits = lambda sim: sim.calculate_orbits(sim.calculate_com())
    else: 
        raise ValueError("'Coordinates must be one of 'jacobi','heliocentric', or 'barycentric'")
    N = len(sa)
    sim0 = sa[0]
    Npl= sim0.N_real - 1
    shape = (Npl,N)
    sim_results = {
        'time':np.zeros(N),
        'P':np.zeros(shape),
        'e':np.zeros(shape),
        'l':np.zeros(shape),
        'inc':np.zeros(shape),
        'pomega':np.zeros(shape),
        'omega':np.zeros(shape),
        'Omega':np.zeros(shape),
        'a':np.zeros(shape),
        'Energy':np.zeros(N)
    }
    for i,sim in enumerate(sa):
        sim_results['time'][i] = sim.t
        orbits = get_orbits(sim)
        sim_results['Energy'][i] = sim.calculate_energy()
        for j,orbit in enumerate(orbits):
            sim_results['P'][j,i] = orbit.P
            sim_results['e'][j,i] = orbit.e
            sim_results['l'][j,i] = orbit.l
            sim_results['pomega'][j,i] = orbit.pomega
            sim_results['a'][j,i] = orbit.a
            sim_results['omega'][j,i] = orbit.omega
            sim_results['Omega'][j,i] = orbit.Omega
            sim_results['inc'][j,i] = orbit.inc
    return sim_results

def _get_reboundx_simarchive_integration_results(sa,coordinates):
    if coordinates == 'jacobi':
        get_orbits = lambda sim: sim.calculate_orbits(jacobi_masses=True)
    elif coordinates == 'heliocentric':
        get_orbits = get_canonical_heliocentric_orbits
    elif coordinates == 'barycentric':
        get_orbits = lambda sim: sim.calculate_orbits(sim.calculate_com())
    else: 
       raise ValueError("'Coordinates must be one of 'jacobi','heliocentric', or 'barycentric'")
    N = len(sa)
    sim0,_ = sa[0]
    Npl= sim0.N_real - 1
    shape = (Npl,N)
    sim_results = {
        'time':np.zeros(N),
        'P':np.zeros(shape),
        'e':np.zeros(shape),
        'l':np.zeros(shape),
        'inc':np.zeros(shape),
        'pomega':np.zeros(shape),
        'omega':np.zeros(shape),
        'Omega':np.zeros(shape),
        'a':np.zeros(shape),
        'Energy':np.zeros(N)
    }
    for i,sim_extra in enumerate(sa):
        sim,extra = sim_extra
        sim_results['time'][i] = sim.t
        orbits = get_orbits(sim)
        sim_results['Energy'][i] = sim.calculate_energy()
        for j,orbit in enumerate(orbits):
            sim_results['P'][j,i] = orbit.P
            sim_results['e'][j,i] = orbit.e
            sim_results['l'][j,i] = orbit.l
            sim_results['pomega'][j,i] = orbit.pomega
            sim_results['a'][j,i] = orbit.a
            sim_results['omega'][j,i] = orbit.omega
            sim_results['Omega'][j,i] = orbit.Omega
            sim_results['inc'][j,i] = orbit.inc
    return sim_results

def get_canonical_heliocentric_orbits(sim):
    """
    Compute orbital elements in canonical
    heliocentric coordinates (e.g., `Laskar & Robutel 1995`_),
    in the center-of-mass frame.
    
    .. _Laskar & Robutel 1995: https://ui.adsabs.harvard.edu/abs/1995CeMDA..62..193L/abstract

    Arguments:
    ----------
    sim : rb.Simulation
        simulation for which to compute orbits

    Returns
    -------
    list of rebound.Orbits
        Orbits of particles in canonical heliocentric
        coordinates.
    """
    return reb_calculate_orbits(sim, coordinates="canonical heliocentric")

def reb_calculate_orbits(sim, coordinates="canonical heliocentric"):
    """
    Compute orbital elements in passed canonical coordinates

    Arguments:
    ----------
    sim : rebound.Simulation
        simulation for which to compute orbits
    coordinates: str
        Specifices the canonical coordinate system. This determines the appropriate definitions of mu and M. Options:
        'canonical heliocentric' (default): canonical heliocentric coordinates in the COM frame e.g. Laskar & Robutel 1995
        'democratic heliocentric': e.g. Duncan et al. 1998

    Returns
    -------
    list of rebound.Orbits
        Orbits of particles in passed canonical coordinates 
    """
    if coordinates not in ['canonical heliocentric', 'democratic heliocentric']:
        raise AttributeError("coordinates must either be 'canonical heliocentric' (default) or 'democratic heliocentric")

    star = sim.particles[0]
    orbits = []
    # the central body in this splitting should have star.m + mu, but 
    # this is already accounted for in REBOUND's orbit calculation 
    fictitious_star = rb.Particle(m=star.m)
    for planet in sim.particles[1:sim.N_real]:
        # Heliocentric position
        r = np.array(planet.xyz) - np.array(star.xyz)
        # For canonical heliocentric coordinates the momentum
        # is the same as the inertial momentum, pi=mi*vi
        v = np.array(planet.vxyz)

        # Mapping from (coordinate,momentum) pair to
        # orbital elments requires that the 'velocity'
        # be defined as the canonical momentum divided
        # by 'mu' appearing in the definition of the
        # Delauney variables (see, e.g., pg. 34 of
        # Morbidelli 2002).
        #
        # For Laskar & Robutel (1995)'s definition
        # of canonical action-angle pairs, this is
        # the reduced mass, so v_for_orbit = mi * vi / mu_i = (m0 + mi)/m0 * vi
        # We write it this way to remain well behaved in test particle limit
        # For DHC: v_for_orbit = mi*vi/mi = vi
       
        # Need to set right central mass for REBOUND calculation
        # for CH, set m so central M = Mstar + m
        # for DHC, m=0 so central M = Mstar
        if coordinates == 'canonical heliocentric':
            m = planet.m
            v_for_orbit = (m + star.m)/star.m*v
        elif coordinates == 'democratic heliocentric':
            m = 0
            v_for_orbit = v

        fictitious_particle =  rb.Particle(
            m=m,
            x = r[0],
            y = r[1],
            z = r[2],
            vx = v_for_orbit[0],
            vy = v_for_orbit[1],
            vz = v_for_orbit[2]
        )
        orbit = fictitious_particle.calculate_orbit(primary=fictitious_star,G = sim.G)
        orbits.append(orbit)
    return orbits

def reb_add_poincare_particle(p, sim):
    elements = {element:getattr(p,element) for element in ['a','e','inc','l','pomega','Omega']}
    reb_add_from_elements(p.m, elements, sim, p.coordinates) 

def add_canonical_heliocentric_elements_particle(m,elements,sim):
    """
    Add a new particle to a rebound simulation 
    by specifying its mass and canonical heliocentric 
    orbital elements.

    Arguments
    ---------
    m : float
        Physical mass of particle to add.
    elements : dict
        Dictionary of orbital elements for particle.
        Dictionary must contain valid set
        of orbital elements accepted by REBOUND.
    sim : rebound.Simulation
        Simulation to add particle to.
    """
    reb_add_from_elements(m, elements, sim, coordinates = 'canonical heliocentric')

def reb_add_from_elements(m,elements,sim,coordinates='canonical heliocentric'):
    """
    Add a new particle to a rebound simulation 
    by specifying its mass, orbital elements, and the set of canonical
    coordinates the elements are in.

    Arguments
    ---------
    m : float
        Physical mass of particle to add.
    elements : dict
        Dictionary of orbital elements for particle.
        Dictionary must contain valid set
        of orbital elements accepted by REBOUND.
    sim : rebound.Simulation
        Simulation to add particle to.
    coordinates: str
        Specifices the canonical coordinate system. This determines the appropriate definitions of mu and M. Options:
        'canonical heliocentric' (default): canonical heliocentric coordinates in the COM frame e.g. Laskar & Robutel 1995
        'democratic heliocentric': e.g. Duncan et al. 1998
    """
    if coordinates not in ['canonical heliocentric', 'democratic heliocentric']:
        raise AttributeError("coordinates must either be 'canonical heliocentric' (default) or 'democratic heliocentric")
    
    star = sim.particles[0]

    # Make a 2body simulation with star mass m, 
    # particle mass = mu (so REBOUND assigns central mass star.m+mu)
    # Given canonical heliocentric elements,
    # this yields xtilde = xi-xstar, vtilde = (mstar+mi)/mstar * vi
    _sim = rb.Simulation()
    _sim.G = sim.G
    _star = star.copy()
    _sim.add(m=star.m)
    # REBOUND will use central mass M = star.m + m, so need to set so that correct M set
    if coordinates == 'canonical heliocentric':
        _sim.add(m=m, **elements) # so REBOUND uses central M = Mstar + m
    elif coordinates == 'democratic heliocentric':
        _sim.add(m=0, **elements) # so REBOUND uses central M = Mstar
    _p = _sim.particles[1]
    p = rb.Particle(m=m) 
    # we cache the planet positions so we can later correct to stay in COM frame
    x = star.x + _p.x
    y = star.y + _p.y
    z = star.z + _p.z
    p.x = x
    p.y = y
    p.z = z
    # Fictitious particle provides vtilde, convert back to inertial velocities
    # vtilde = momentum/canonical mass (see get_canonical_heliocentric_orbits)
    # CH: vtilde = m*v/mu, so v = f*vtilde
    # DHC: vtilde = m*v/m, so v = vtilde
    if coordinates == 'canonical heliocentric':
        f = star.m / (m + star.m)
    elif coordinates == 'democratic heliocentric':
        f = 1.
    p.vx = f * _p.vx
    p.vy = f * _p.vy
    p.vz = f * _p.vz
    # All of these need to be the final velocities in the COM frame, so the
    # only way to do that is to compensate for the shift in COM with the star
    star.vx -= m/star.m * p.vx
    star.vy -= m/star.m * p.vy
    star.vz -= m/star.m * p.vz
    sim.add(p)
    # We now move all particles by the same offset to keep the COM at 0
    # This ensures that the xi-x0 remain the same
    Mtot = np.array([p.m for p in sim.particles[:sim.N_real]]).sum()
    for p in sim.particles[:sim.N_real]:
        p.x -= m/Mtot*x
        p.y -= m/Mtot*y
        p.z -= m/Mtot*z

def _compute_transformation_angles(sim):
    Gtot_vec = sim.calculate_angular_momentum()
    Gtot_vec = np.array(Gtot_vec)
    Gtot = np.sqrt(Gtot_vec @ Gtot_vec)
    Ghat = Gtot_vec / Gtot
    Ghat_z = Ghat[-1]
    Ghat_perp = np.sqrt(1 - Ghat_z**2)
    theta1 = np.pi/2 - np.arctan2(Ghat[1],Ghat[0])
    theta2 = np.pi/2 - np.arctan2(Ghat_z,Ghat_perp)
    return theta1,theta2

def npEulerAnglesTransform(xyz,Omega,I,omega):
    x,y,z = xyz
    s1,c1 = np.sin(omega),np.cos(omega)
    x1 = c1 * x - s1 * y
    y1 = s1 * x + c1 * y
    z1 = z

    s2,c2 = np.sin(I),np.cos(I)
    x2 = x1
    y2 = c2 * y1 - s2 * z1
    z2 = s2 * y1 + c2 * z1

    s3,c3 = np.sin(Omega),np.cos(Omega)
    x3 = c3 * x2 - s3 * y2
    y3 = s3 * x2 + c3 * y2
    z3 = z2

    return np.array([x3,y3,z3])

def align_simulation(sim):
    """
    Change particle positions and velocities
    of a rebound simulations  so that the z-axis 
    corresponds with the direction of the angular 
    momentum. 

    Arguments
    ---------
    sim : rebound.Simulation
    """
    theta1,theta2 = _compute_transformation_angles(sim)
    for p in sim.particles[:sim.N_real]:
        p.x,p.y,p.z = npEulerAnglesTransform(p.xyz,0,theta2,theta1)
        p.vx,p.vy,p.vz = npEulerAnglesTransform(p.vxyz,0,theta2,theta1) 
def _get_lhat(p):
    xyz = p.xyz
    vxyz = p.vxyz
    lvec = np.cross(xyz,vxyz)
    lhat = lvec / np.linalg.norm(lvec)
    return lhat
from itertools import combinations
def calculate_mutual_inclinations(s):
    """
    Calculate the mutual inclination between pairs
    of bodies in a simulation archive

    Arguments
    ---------
    s : rebound.Simulation or rebound.SimulationArchive

    Returns
    -------
    imutuals : dictionary
        Dictionary of mutual inclinations.
        Keys are pairs of particle numbers, (i,j),
        where i and j are integers and values are 
        arrays of mutual inclination values for that 
        key pair.
    """
    if type(s)==rb.Simulation:
        return _calculate_mutual_inc(s)
    combs = combinations(range(1,sa[0].N_real),2)
    imut = {comb:np.zeros(len(sa)) for comb in combs}
    for i,sim in enumerate(sa):
        lhats = [_get_lhat(p) for p in sim.particles[1:]]
        for comb in combinations(range(1,sa[0].N_real),2):
            j,k = comb
            cosimut = lhats[j-1] @ lhats[k-1]
            imut[comb][i] = np.arccos(cosimut)

    return imut
def _calculate_mutual_inc(sim):
    combs = combinations(range(1,sim.N_real),2)
    lhats = [_get_lhat(p) for p in sim.particles[1:]]
    return {(j,k):np.arccos(lhats[j-1]@lhats[k-1]) for j,k in combs}
    
