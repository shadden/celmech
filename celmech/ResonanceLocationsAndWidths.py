from celmech import disturbingfunction
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
def get_fg_coeffs(res_j,res_k):
	"""Get 'f' and 'g' coefficients for approximating the disturbing function coefficients associated with an MMR."""
	res_pratio = float(res_j - res_k) /float(res_j)
	alpha = res_pratio**(2./3.)
	Cjkl = disturbingfunction.general_order_coefficient
	fK = Cjkl(res_j, res_k, res_k, alpha)
	gK = Cjkl(res_j, res_k, 0 , alpha)
	# target fn
#	err_sq = lambda x,y: np.total([(( Cjlk(res_j,res_k,l,alpha) - binom(res_k,l)* f**(l) * g**(res_k-l) ) /  Cjlk(res_j,res_k,l,alpha))**2 for l in range(0,res_k+1)])
	f = -1 * np.abs(fK)**(1./res_k)
	g =      np.abs(gK)**(1./res_k)
	return f,g

def resonant_period_ratios(min_per_ratio,max_per_ratio,order):
	"""Return the period ratios of all resonances up to order 'order' between 'min_per_ratio' and 'max_per_ratio' """
	minJ = int(np.floor(1. /(1. - min_per_ratio)))
	maxJ = int(np.ceil(1. /(1. - max_per_ratio)))
	res_ratios=[(minJ-1,minJ)]
	for j in range(minJ,maxJ):
		res_ratios = res_ratios + [ ( x[1] * j - x[1] + x[0] , x[1] * j + x[0]) for x in farey_sequence(order)[1:] ]
	return res_ratios

def resonance_pratio_span(mu1,mu2,Z0,res_j,res_k):
	"""Compute the span of a resonance in terms of period ratio"""
	res_pratio = float(res_j - res_k) /float(res_j)
	alpha = res_pratio**(2./3.)
	beta = mu2 / mu1 / np.sqrt(alpha)
	f,g = get_fg_coeffs(res_j,res_k)
	
	A = 1.5 * (res_j*res_j + beta * res_j * (res_j-res_k))
	Eps = 2 * mu1 * np.sqrt(f*f + g*g)**res_k * Z0**res_k
	DeltaP = np.sqrt(4 * Eps / A )
	return (res_pratio * (1. - 1.5 * res_j * DeltaP ) / (1 + 1.5 * beta * (res_j - res_k) * DeltaP ) , \
	res_pratio * (1. + 1.5 * res_j * DeltaP ) / (1 - 1.5 * beta * (res_j - res_k) * DeltaP ))


