from . import disturbingfunction
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
	minJ = int(np.floor(1. /(1. - min_per_ratio)))
	maxJ = int(np.ceil(1. /(1. - max_per_ratio)))
	res_ratios=[(minJ-1,minJ)]
	for j in range(minJ,maxJ):
		res_ratios = res_ratios + [ ( x[1] * j - x[1] + x[0] , x[1] * j + x[0]) for x in farey_sequence(order)[1:] ]
	return res_ratios

def resonance_pratio_halfwidth(mu1,mu2,Z0,res_j,res_k):
	""" Compute the half-width of a resonance in terms of period ratio"""
	res_pratio = float(res_j - res_k) /float(res_j)
	alpha = res_pratio**(2./3.)
	beta = mu2 / mu1 / np.sqrt(alpha)
	f,g = get_fg_coeffs(res_j,res_k)
	
	prefactor_num = 16. * (f*f +g*g)**(res_k/2)
	prefactor_denom = 3. * (res_j*res_j + beta * (res_j)*(res_j-res_k))

	DeltaP = np.sqrt(prefactor_num / prefactor_denom )* np.sqrt(mu1) * Z0**(res_k/2.)
	

def get_fg_coeffs(res_j,res_k):
	"""Get 'f' and 'g' coefficients for approximating the disturbing function coefficients associated with an MMR."""
	res_pratio = float(res_j - res_k) /float(res_j)
	alpha = res_pratio**(2./3.)
	fK = disturbingfunction.general_order_coefficient(res_j, res_k, res_k, a)
	gK = disturbingfunction.general_order_coefficient(res_j, res_k, 0 , a)
	f = -1 * np.abs(fK)**(1./res_k)
	g =      np.abs(gK)**(1./res_k)
	return f,g
