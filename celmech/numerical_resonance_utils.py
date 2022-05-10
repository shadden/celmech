import theano.tensor as T
from celmech.theano_ops.kepler import KeplerOp
def planar_els2xv(a,lmbda,h,k,GMstar):
    ko = KeplerOp()
    e_sq = h*h+k*k
    e = T.sqrt(e_sq)
    # ko(MeanAnomaly,eccentricity) = sin_TrueAnomly, cos_TrueAnomaly
    sin_f,cos_f =  ko( lmbda - T.arctan2(h,k), e + T.zeros_like(lmbda))
    cos_theta = cos_f * (k/e) - sin_f * (h/e)
    sin_theta = sin_f * (k/e) + cos_f * (h/e)
    n = T.sqrt(GMstar) * a**(-3/2)
    ecosf = k * cos_theta + h * sin_theta
    r = a * (1-e_sq) /(1 + ecosf)
    x = r * cos_theta
    y = r * sin_theta
    vel = n * a / T.sqrt(1-e_sq)
    u = -1 * vel * (h + sin_theta)
    v = vel * (k + cos_theta)
    return x,y,u,v

def R_inc_mtrx_transform(x,y,u,v,p,q):
    cosIby2 = T.sqrt(1 - p * p - q * q)
    
    x1 = (1-2*p*p) * x + 2 * p * q * y
    y1 = 2 * p * q * x + (1-2*q*q) * y
    z1 = -2 * p * cosIby2 * x + 2 * q * cosIby2 * y 

    u1 = (1-2*p*p) * u + 2 * p * q * v
    v1 = 2 * p * q * u + (1-2*q*q) * v
    w1 = -2 * p * cosIby2 * u + 2 * q * cosIby2 * v 

    return x1,y1,z1,u1,v1,w1

def calc_Hint_components_planar(a1,a2,l1,l2,h1,k1,h2,k2,GMstar1,GMstar2):
    r"""
    Compute the value of the disturbing function components
    .. math::
        H_{dir} = -\frac{1}{|r-r'|}
        H_{ind} = v.v'  
    from a set of input orbital elements for coplanar planets.

    Arguments
    ---------
    a1 : float
        inner planet semi-major axis
    a2 : float
        outer planet semi-major axis
    l1 : float
        inner planet mean longitude
    l2 : float
        outer planet mean longitude
    h1 :
        e1 * sin(pomega1)
    k1 : 
        e1 * cos(pomega1)
    h2 :
        e2 * sin(pomega2)
    k2 : 
        e2 * cos(pomega2)

    Returns
    -------
    float :
        Disturbing function value
    """
    x1,y1,u1,v1 = planar_els2xv(a1,l1,h1,k1,GMstar1)
    x2,y2,u2,v2 = planar_els2xv(a2,l2,h2,k2,GMstar2)

    # direct term
    dx = (x2 - x1)
    dy = (y2 - y1)
    dr2 = dx*dx + dy*dy
    direct = -1 / T.sqrt(dr2)

    # indirect term
    indirect = u1*u2 + v1*v2
    return direct,indirect

def calc_Hint_components_spatial(a1,a2,l1,l2,h1,k1,h2,k2,p1,q1,p2,q2,GMstar1,GMstar2):
    r"""
    Compute the value of the disturbing function components
    .. math::
        H_{dir} = -\frac{1}{|r-r'|}
        H_{ind} = v.v'  
    from a set of input orbital elements for coplanar planets.

    Arguments
    ---------
    a1 : float
        inner planet semi-major axis
    a2 : float
        outer planet semi-major axis
    l1 : float
        inner planet mean longitude
    l2 : float
        outer planet mean longitude
    h1 :
        e1 * sin(pomega1)
    k1 : 
        e1 * cos(pomega1)
    h2 :
        e2 * sin(pomega2)
    k2 : 
        e2 * cos(pomega2)
    p1 : float
        sin(I1/2) sin(Omega1)
    q1 : float
        sin(I1/2) cos(Omega1)
    p2 : float
        sin(I2/2) sin(Omega2)
    q2 : float
        sin(I2/2) cos(Omega2)

    Returns
    -------
    float :
        Disturbing function value
    """
    _x1,_y1,_u1,_v1 = planar_els2xv(a1,l1,h1,k1,GMstar1)
    _x2,_y2,_u2,_v2 = planar_els2xv(a2,l2,h2,k2,GMstar2)

    x1,y1,z1,u1,v1,w1 = R_inc_mtrx_transform(_x1,_y1,_u1,_v1,p1,q1)
    x2,y2,z2,u2,v2,w2 = R_inc_mtrx_transform(_x2,_y2,_u2,_v2,p2,q2)
    # direct term
    dx = (x2 - x1)
    dy = (y2 - y1)
    dz = (z2 - z1)
    dr2 = dx*dx + dy*dy + dz*dz
    direct = -1 / T.sqrt(dr2)

    # indirect term
    indirect = u1*u2 + v1*v2 + w1*w2
    return direct,indirect
