import numpy as np
import rebound as rb
import reboundx as rbx

def set_timestep(sim,dtfactor):
        ps=sim.particles[1:]
        tperi=np.min([p.P * (1-p.e)**1.5 / np.sqrt(1+p.e) for p in ps])
        dt = tperi * dtfactor
        sim.dt = dt
def set_min_distance(sim,rhillfactor):
        ps=sim.particles[1:]
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
             - 'jacobi' : Use Jacobi coordinates (including Jacobi masses)
             - 'heliocentric' : Use canonical heliocentric elements. The
                    canonical cooridantes are heliocentric distance vectors.
                    The conjugate momenta are barycentric momenta.

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
    if coordinates is 'jacobi':
        get_orbits = lambda sim: sim.calculate_orbits(jacobi_masses=True)
    elif coordinates is 'heliocentric':
        get_orbits = get_canonical_heliocentric_orbits
    elif coordinates is 'barycentric':
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
    if coordinates is 'jacobi':
        get_orbits = lambda sim: sim.calculate_orbits(jacobi_masses=True)
    elif coordinates is 'heliocentric':
        get_orbits = get_canonical_heliocentric_orbits
    elif coordinates is 'barycentric':
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
    Compute orbital elements in canconical 
    heliocentric coordinates.

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
    star = sim.particles[0]
    orbits = []
    fictitious_star = rb.Particle(m=star.m)
    com = sim.calculate_com()
    for planet in sim.particles[1:]:

        # Heliocentric position
        r = np.array(planet.xyz) - np.array(star.xyz)

        # Barycentric momentum
        rtilde = planet.m * ( np.array(planet.vxyz) - np.array(com.vxyz) )

        # Mapping from (coordinate,momentum) pair to
        # orbital elments requires that the 'velocity'
        # be defined as the canonical momentum divided
        # by 'mu' appearing in the definition of the
        # Delauney variables (see, e.g., pg. 34 of
        # Morbidelli 2002).
        #
        # For Laskar & Robutel (1995)'s definition
        # of canonical action-angle pairs, this is
        # the reduced mass.
        #
        # For democratic heliocentric elements,
        # 'mu' is simply the planet mass.
        mu = planet.m * star.m / (planet.m + star.m)
        v_for_orbit = rtilde / mu

        fictitious_particle =  rb.Particle(
            m=planet.m,
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

def add_canonical_heliocentric_elements_particle(m,elements,sim):
    """
    Add a new particle to a rebound simulation 
    by specifying its mass and canonical heliocentric 
    orbital elements.

    Arguments
    ---------
    m : float
        Mass of particle to add.
    elements : dict
        Dictionary of orbital elements for particle.
        Dictionary must contain valid set
        of orbital elements accepted by REBOUND.
    sim : rebound.Simulation
        Simulation to add particle to.
    """
    star = sim.particles[0]
    _sim = rb.Simulation()
    _star = star.copy()
    _sim.add(_star)
    _sim.add(
            primary=_star,
            m=m,
            **elements
    )
    _p = _sim.particles[1]
    p = _p.copy()
    f = star.m / (p.m + star.m)
    p.vx = f * ( _p.vx - star.vx )
    p.vy = f * ( _p.vy - star.vy )
    p.vz = f * ( _p.vz - star.vz )
    sim.add(p)
    star.vx -= p.m * p.vx / star.m
    star.vy -= p.m * p.vy / star.m
    star.vz -= p.m * p.vz / star.m

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
    for p in sim.particles:
        p.x,p.y,p.z = npEulerAnglesTransform(p.xyz,0,theta2,theta1)
        p.vx,p.vy,p.vz = npEulerAnglesTransform(p.vxyz,0,theta2,theta1) 
