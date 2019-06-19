from celmech import disturbing_function as df
import numpy as np

# taken from https://en.wikipedia.org/wiki/Farey_sequence
def farey_sequence(n):
    """Return the nth Farey sequence as order pairs of the form (N,D) where `N' is the numerator and `D' is the denominator."""
    a, b, c, d = 0, 1, 1, n
    sequence=[(a,b)]
    while (c <= n):
        k = int((n + b) / d)
        a, b, c, d = c, d, (k*c-a), (k*d-b)
        sequence.append( (a,b) )
    return sequence
def resonant_period_ratios(min_per_ratio,max_per_ratio,order):
    """Return the period ratios of all resonances up to order 'order' between 'min_per_ratio' and 'max_per_ratio' """
    if min_per_ratio < 0.:
        raise AttributeError("min_per_ratio of {0} passed to resonant_period_ratios can't be < 0".format(min_per_ratio))
    if max_per_ratio >= 1.:
        raise AttributeError("max_per_ratio of {0} passed to resonant_period_ratios can't be >= 1".format(max_per_ratio))
    minJ = int(np.floor(1. /(1. - min_per_ratio)))
    maxJ = int(np.ceil(1. /(1. - max_per_ratio)))
    res_ratios=[(minJ-1,minJ)]
    for j in range(minJ,maxJ):
        res_ratios = res_ratios + [ ( x[1] * j - x[1] + x[0] , x[1] * j + x[0]) for x in farey_sequence(order)[1:] ]
    res_ratios = np.array(res_ratios)
    msk = np.array( list(map( lambda x: min_per_ratio < x[0]/float(x[1]) < max_per_ratio , res_ratios )) )
    return res_ratios[msk]
def resonance_jk_list(min_per_ratio,max_per_ratio,order):
    """Return the 'j' and 'k' of resonances up to order 'order' between 'min_per_ratio' and 'max_per_ratio' """
    minJ = int(np.floor(1. /(1. - min_per_ratio)))
    maxJ = int(np.ceil(1. /(1. - max_per_ratio)))
    res_jk=[(minJ,1)]
    for j in range(minJ,maxJ):
        res_jk = res_jk + [ ( x[1] * j + x[0] , x[1]) for x in farey_sequence(order)[1:] ]
    res_jk =np.array(res_jk)
    msk = np.array( list(map( lambda x: min_per_ratio < float(x[0]-x[1])/float(x[0]) < max_per_ratio , res_jk )) )
    return res_jk[msk]
def resonance_pratio_span(mu1,mu2,Z0,res_j,res_k):
    """Compute the span of a resonance in terms of period ratio. 
    Inputs:
        mu1,mu2        --    Planet-star mass raitos of inner and outer planet
        Z0            --    Combined eccentricity of planet pair
        res_j,res_k    --    Specify the resonance as the res_j : (res_j-res_k) MMR
    Returns:
     the minimum and maximum period ratio spanned by resonance."""
    res_pratio = float(res_j - res_k) /float(res_j)
    alpha = res_pratio**(2./3.)
    beta = mu2 / mu1 / np.sqrt(alpha)
    f,g = df.get_fg_coeffs(res_j,res_k)

    # Pendulum approximation: 
    #        H(P,Q) =     (1/2)A P^2 + B cos(Q)
    A = 1.5 * (res_j*res_j + beta * res_j * (res_j-res_k))
    B = 2 * mu1 * np.sqrt(f*f + g*g)**res_k * Z0**res_k

    DeltaP = np.sqrt(4 * B / A )
    return (\
    res_pratio * (1. - 1.5 * res_j * DeltaP ) / (1 + 1.5 * beta * (res_j - res_k) * DeltaP ) , \
    res_pratio * (1. + 1.5 * res_j * DeltaP ) / (1 - 1.5 * beta * (res_j - res_k) * DeltaP )\
    )

def pendulum_approx_coeffs(mu1,mu2,res_j,res_k):
    res_pratio = float(res_j - res_k) /float(res_j)
    alpha = res_pratio**(2./3.)
    beta = mu2 / mu1 / np.sqrt(alpha)
    f,g = df.get_fg_coeffs(res_j,res_k)
    # Pendulum approximation: 
    #        H(P,Q) =     (1/2)A P^2 + B * Z0^k cos(Q)
    A = 1.5 * (res_j*res_j + beta * res_j * (res_j-res_k))
    B = 2 * mu1 * np.sqrt(f*f + g*g)**res_k
    return A,B,beta
    
def two_resonance_intersection(mu1,mu2,res_L,res_R):
    """Compute the intersection of adjacent resonances"""
    jL,kL = res_L
    jR,kR = res_R
    
    if (jL-kL)/float(jL) > (jR-kR)/float(jR):
    #swap left and right resonances
        jR,jL = jL,jR
        kR,kL = kL,kR
    max_k = np.max((kL,kR))
    #
    per_ratio_L = (jL-kL)/float(jL)
    per_ratio_R = (jR-kR)/float(jR)
    #
    aL,bL,betaL = pendulum_approx_coeffs(mu1,mu2,jL,kL)
    aR,bR,betaR = pendulum_approx_coeffs(mu1,mu2,jR,kR)
    #
    coeffL = per_ratio_L * 1.5 * (jL + betaL *(jL - kL)) * np.sqrt(4*bL/aL)
    coeffR = per_ratio_R * 1.5 * (jR + betaR *(jR - kR)) * np.sqrt(4*bR/aR)
    coeff0 = per_ratio_L - per_ratio_R
    #
    coeffs = np.zeros(max_k+1)
    coeffs[0] = coeff0
    coeffs[kL] = coeffL
    coeffs[kR] = coeffR
    roots = np.roots(coeffs[::-1])
    sqrt_e = np.min( roots[np.logical_and(np.imag(roots)==0,np.real(roots)>0)] ) 
    sqrt_e = np.real(sqrt_e)
    
    return (per_ratio_L + coeffL * sqrt_e**kL  , sqrt_e * sqrt_e)

def resonance_intersections_list(mu1,mu2,min_p,max_p,max_order):
    """
        Compute the list or resonance intersections for all resoances between 'min_per_ratio'
        and 'max_per_ratio' up to the specified 'order'. A list of ordered pairs containing 
        the period ratios and combined eccentricity values of the resonances intersections is
        returned. 
    """
    res_jk=resonance_jk_list(min_p,max_p,max_order)
    intersections=[two_resonance_intersection(mu1,mu2,r1,r2) for r1,r2 in zip(res_jk[:-1],res_jk[1:])]
    return np.array(intersections)
    
