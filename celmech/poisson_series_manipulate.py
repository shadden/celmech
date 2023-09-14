import numpy as np
from scipy.special import factorial
from collections import defaultdict
import sympy as sp

class PSTerm():
    def __init__(self,C,k,kbar,p,q):
        """
        Class representing a single monomial term in a Poisson series.

        Parameters
        ----------
        C : float
            Numerical value of constant coefficient.
        k : ndarray
            1D array of integers denoting powers of complex canonical variables.
        kbar : ndarray
            1D array of integers denoting powers of conjugate complex canonical variables.
        p : ndarray
            1D array of integers denoting powers of action-like variables.
        q : ndarray
            1D array of integers denoting the wave-vector multiplying angle variables.
        """
        self.C = C
        self.k = np.array(k,dtype=int)
        self.kbar = np.array(kbar,dtype=int)
        self.p=np.array(p,dtype=int)
        self.q=np.array(q,dtype=int)
         

    def __call__(self,x,P,Q):
        xbar = np.conj(x)
        k,kbar,p,q = self.k,self.kbar,self.p,self.q
        C = self.C
        val=C * np.prod(x**k) * np.prod(xbar**kbar) * np.prod(P**p) * np.exp(1j * q@Q)
        return val
    # define scalar multiply
    def __mul__(self,val):
        # Scalar multiplication
        return PSTerm(val * self.C,self.k,self.kbar,self.p,self.q)
    # scalar multiply is commutative
    __rmul__ = __mul__

class PoissonSeries():
    def __init__(self,N,M,**kwargs):
        r"""
        A class representing a Poisson series in :math:`N` complex canonically
        conjugate variables and :math:`M` angle-action pairs. Individual terms
        have form 

        .. math::
            C \times \left(\prod_{i=1}^{M} P_i^{p_i}e^{\sqrt{-1}q_iQ_i}\right)
            \left(\prod_{j=1}^{N} x_j^{k_j}\bar{x}_j^{\bar{k}_j}\right)

        The object stores the key-value pairs of :math:`(p,q,k,\bar{k}):C`.

        Parameters
        ----------
        N : int
            Number of complex conjugate variable pairs.
        M : int
            Number of conjugate action-angle variable pairs.
        """
        assert type(N) is int
        assert type(M) is int
        self.N = N
        self.M = M
        # indices of components of p,q,k,kbar 
        self.pi = np.arange(0,M)
        self.qi = np.arange(M,2*M)
        self.ki = np.arange(2*M,2*M+N)
        self.kbari = np.arange(2*M+N,2*M+2*N)


        self._keylength = 2*N+2*M
        self._terms_dict = defaultdict(complex)
        
        if N>0:
            default_cvar_symbols = sp.symbols(
                ",".join(
                    ["z{}".format(i) for i in range(1,N+1)] +\
                    [r"\bar{{z}}_{}".format(i) for i in range(1,N+1)]
                )
            )
            self.cvar_symbols = kwargs.get("cvar_symbols",default_cvar_symbols)
        else:
            self.cvar_symbols = ()
        if M>0:
            default_pvar_symbols = sp.symbols(",".join(["p{}".format(i) for i in range(1,M+1)]) + ",")
            self.pvar_symbols = kwargs.get("pvar_symbols",default_pvar_symbols)

            default_thetavar_symbols = sp.symbols(",".join([r"\theta_{}".format(i) for i in range(1,M+1)])+ ",")
            self.thetavar_symbols = kwargs.get("thetavar_symbols",default_thetavar_symbols)
        else:
            self.pvar_symbols=()
            self.thetavar_symbols = ()
        # save symbol kwargs as dictionary
        self._symbol_kwargs = {
            "cvar_symbols":self.cvar_symbols,
            "thetavar_symbols":self.thetavar_symbols,
            "pvar_symbols":self.pvar_symbols
            }
    def _key_to_PSTerm(self,key):
        C = self[key]
        arr=np.array(key)
        return PSTerm(C,arr[self.ki],arr[self.kbari],arr[self.pi],arr[self.qi])
    
    def _PSTerm_to_key(self,psterm):
        arr =np.concatenate((psterm.p,psterm.q,psterm.k,psterm.kbar))
        return tuple(arr)
    
    @property
    def terms(self):
        return [self._key_to_PSTerm(key) for key in self._terms_dict.keys()]
        
    def add_PSTerm(self,psterm):
        """
        Add a monomial term to the series. If a term with the same
        exponents already exists, the value of the new term's coefficent
        is added to the existing term's.
        """
        key = self._PSTerm_to_key(psterm)
        self[key] += psterm.C
        
    @classmethod
    def from_PSTerms(cls,terms,N=None,M=None,**kwargs):
        """
        Initialize a PoissonSeries from a list of PSTerms.
        """
        if (N is None) or (M is None):
            term = terms[0]
            N = len(term.k)
            M = len(term.p)
        new = cls(N,M,**kwargs)
        for term in terms:
            new.add_PSTerm(term)
        return new
    
    def __setitem__(self, key, value):
        assert len(key)==self._keylength
        self._terms_dict[key] = value
        
    def __getitem__(self,key):
        # Note-- use of 'get' ensures that
        # new items are not needlessly added when
        # a key does not already exist.
        return self._terms_dict.get(key,0.j)
    
    def items(self):
        return self._terms_dict.items()
    
    def __add__(self,ps):
        # Addition between Poisson series objects
        if type(ps)==PoissonSeries:
            new = PoissonSeries(self.N,self.M,**self._symbol_kwargs)
            new._terms_dict = self._terms_dict.copy()
            for key,val in ps._terms_dict.items():
                new._terms_dict[key] += val
            return new
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(PoissonSeries,type(ps)))
    
    # define scalar multiply term-wise
    def __mul__(self,val):
        if len(self.terms)==0:
            return self
        return PoissonSeries.from_PSTerms([term * val for term in self.terms],**self._symbol_kwargs)
        # Scalar multiplication
    # scalar multiplication is commutative
    __rmul__ = __mul__

    def Lie_deriv(self,ps):
        """
        Compute the Lie derivative of a Poisson series expression
        with respect to this Poisson series.
        """
        if type(ps)==PoissonSeries:
            return bracket(ps,self,**self._symbol_kwargs)
        else:
            raise TypeError("unsupported type for Lie_deriv: '{}'".format(type(ps)))

    def as_expression(self):
        """
        Returns the Poisson series as a sympy expression.
        """
        exprn = 0
        cvars = self.cvar_symbols
        pvars = self.pvar_symbols
        thetavars = self.thetavar_symbols
        
        M = self.M
        for key,val in self._terms_dict.items():
            term = val
            arr = np.array(key)
            j = arr[:M]
            k = arr[M:2*M]
            l = arr[2*M:]
            term *= sp.Mul(*[cv**p for cv,p in zip(cvars,l)])
            term *= sp.Mul(*[pv**p for pv,p in zip(pvars,j)])
            term *= sp.exp(sp.I * sp.Add(*[kk*theta for theta,kk in zip(k,thetavars)]))
            exprn += term
        return exprn
    def __call__(self,x,P,Q):
        tot=0
        M = self.M
        N = self.N 
        xbar = np.conj(x)
        for key,val in self._terms_dict.items():
            arr = np.array(key)
            k,kbar,q,p = arr[self.ki],arr[self.kbari],arr[self.qi],arr[self.pi]
            tot += val * np.prod(P**p) * np.prod(x**k) * np.prod(xbar**kbar) * np.exp(1j*q@Q)
        return tot
  
def bracket(PSeries1,Pseries2,**kwargs):
    N,M = PSeries1.N,PSeries1.M
    
    assert Pseries2.N==N and Pseries2.M==M, \
    "Dimensions of poisson series {} and {} do not match!".format(PSeries1,Pseries2)

    result = PoissonSeries(N,M,**kwargs)
    for term1 in PSeries1.terms:
        k = term1.k
        kbar = term1.kbar
        p = term1.p
        q = term1.q
        C1 = term1.C
        for term2 in Pseries2.terms:
            l = term2.k
            lbar = term2.kbar
            r = term2.p
            s = term2.q
            C2 = term2.C
            k_plus_l = k+l
            kbar_plus_lbar = kbar+lbar
            p_plus_r = p + r
            q_plus_s = q + s
            for i,oi in enumerate(np.eye(len(k),dtype=int)):
                pre = kbar[i]*l[i] - k[i]*lbar[i]
                if pre:
                    knew = k_plus_l - oi
                    knew_bar = kbar_plus_lbar - oi
                    Cnew = 1j*pre*C1*C2
                    new_term = PSTerm(Cnew,knew,knew_bar,p_plus_r,q_plus_s)
                    result.add_PSTerm(new_term)
            for j,oj in enumerate(np.eye(len(p),dtype=int)):
                pre = q[j]*r[j] - s[j]*p[j]
                if pre:
                    Cnew = 1j * pre * C1 * C2
                    new_term = PSTerm(Cnew,k_plus_l,kbar_plus_lbar,p_plus_r - oj,q_plus_s)
                    result.add_PSTerm(new_term)
    return result

def Psi_to_chi_and_Hav(omega_vec,Psi,kres):
    chi_terms = []
    Hav_terms = []
    Nkres = len(kres)
    N = len(omega_vec)
    if Nkres>0:
        kres_matrix = np.vstack(kres)
    else:
        kres_matrix = np.zeros(len(omega_vec))
    assert np.linalg.matrix_rank(kres_matrix) == Nkres, "Resonance vectors {} are not linearly independent!".format(kres)
    for term in Psi.terms:
        kvec = term.kbar - term.k
        mat = np.vstack((kres_matrix,kvec))
        r = np.linalg.matrix_rank(mat)
        if r>Nkres:
            omega = kvec @ omega_vec
            amp = -1j * term.C/omega
            chi_terms.append(PSTerm(amp,term.k,term.kbar,term.p,term.q))
        else:
            Hav_terms.append(term)
    chi = PoissonSeries.from_PSTerms(chi_terms,N,0)
    hav=PoissonSeries.from_PSTerms(Hav_terms,N,0)
    return chi,hav

def do_perturbation_theory(omega_vec,H,lmax):
    N = len(omega_vec)
    chi,Phi,Hav = [defaultdict(lambda: PoissonSeries(N,0)) for _ in range(3)]
    Hav[2] += H[2]
    for l in range(2,lmax+1):
        Phi[(0,l)] += H[l]
        Psi = PoissonSeries(N,0)
        Psi+= Phi[(0,l)]        
        for n in range(1,l-1):
            kmax = l+1-n if n>1 else l-n
            for k in range(3,kmax+1):
                Phi[(n,l)]+=chi[k].Lie_deriv(Phi[(n-1,l+2-k)])
            Psi += Phi[(n,l)]*(1/factorial(n))
        if l>2:
            chi[l],Hav[l] = Psi_to_chi_and_Hav(omega_vec,Psi,[])
            Phi[(1,l)]+=chi[l].Lie_deriv(Phi[(0,2)])
    return chi,Hav,Phi

def expL(f,chi,lmax=None):
    kmin = min(chi.keys())
    k1min = min(f.keys())
    Phi = defaultdict(chi.default_factory)
    E = defaultdict(chi.default_factory)
    if not lmax:
        lmax = max(chi.keys()) + k1min - 2
    for l in range(k1min,lmax+1):
        Phi[0,l] += f[l]
        E[l] += Phi[0,l]
        nmax = (l-2) // (kmin-2)
        lmin_n = k1min
        for n in range(1,nmax+1):
            kmax = l+2-lmin_n
            for k in range(kmin,kmax+1):
                Phi[(n,l)]+=chi[k].Lie_deriv(Phi[(n-1,l+2-k)])
            lmin_n = kmin + lmin_n - 2 
            E[l] += Phi[(n,l)] * (1/factorial(n))
    return E

def expLinv(f,chi,lmax=None):
    if not lmax:
        lmax = max(chi.keys())    
    lmin = min(chi.keys())
    nchi = defaultdict(chi.default_factory)
    for key,val in chi.items():
        nchi[key] = val * (-1.)
    return expL(f,nchi,lmax)

def k_nu_l_to_PSindices(Npl,i,j,k_vec,nu_vec,l_vec):
    r"""
    Convert the multi-indicies :math:`k,\nu,l` used in the disturbing
    function expansion to the appropriate Poisson series indices.

    Parameters
    ----------
    Npl : int
        Total number of planets being represented in Poisson series.
    i : int
        Index of inner planet
    j : int
        Index of outer planet
    k_vec : ndarray
        1D array of integers determining the disturbing function cosine argument.
    nu_vec : ndarray
        1D array of integers specifying powers of action-like variables.
    l_vec : ndarray
        1D array of integers specifying powers of :math:`\delta\Lambda`

    Returns
    -------
    tuple 
        Arrays k, kbar, p, q appearing in PSTerm.
    """
    N = 2 * Npl # number of complex K values  
    M = Npl
    nuvec = np.array(nu_vec)
    kvec  = np.array(k_vec)
    lvec = np.array(l_vec)
    nu_indx=np.array((2,3,0,1))
    
    k = nuvec[nu_indx] + (kvec * (kvec>0))[2:]
    kbar = nuvec[nu_indx] - (kvec * (kvec<0))[2:]
    p = lvec
    q = np.array((kvec[1],kvec[0]))

    k_all = np.zeros(N,dtype = int)
    kbar_all = np.zeros(N,dtype = int)

    q_all = np.zeros(M,dtype = int)
    p_all = np.zeros(M,dtype = int)
    
    c_msk = np.array((i,j,Npl + i, Npl + j)) - 1
    r_msk = np.array((i,j)) - 1 
    
    k_all[c_msk] = k
    kbar_all[c_msk] = kbar
    p_all[r_msk] = p
    q_all[r_msk] = q
    return k_all,kbar_all,p_all,q_all

from .poincare import _get_a0_symbol
from .disturbing_function import df_coefficient_C, evaluate_df_coefficient_dict

def DFTerm_as_PSterms(pham,i,j,kvec,nuvec,lvec):
    """
    Generate Poisson series terms representing a specific disturbing function
    term between a pair of planets.

    Parameters
    ----------
    pham : :class:`celmech.poincare.PoincareHamiltonian`
        The Hamiltonian class containing the pair of particles of interest.
    i : int
        index of the inner planet
    j : int
        index of the outer planet
    kvec : ndarray
        1D array specifying cosine argument in the DF
    nuvec : ndarray
        1D array specifying Taylor expansion DF term of cosine coefficient.
    lvec : ndarray
        1D array specifcying Taylor expansion term in powers of delta-Lambda.

    Returns
    -------
    list
        A list of PSTerm objects.
    """
    k,kbar,p,q = k_nu_l_to_PSindices(pham.N-1,i,j,kvec,nuvec,lvec)
    G = pham.state.G
    mi = pham.particles[i].m
    mj = pham.particles[j].m
    
    aj0 = pham.H_params[_get_a0_symbol(j)]
    Lambda_i0 = pham.H_params[pham.Lambda0s[i]]
    Lambda_j0 = pham.H_params[pham.Lambda0s[j]]
    alpha_ij = pham.H_params[sp.symbols(r"\alpha_{{{0}\,{1}}}".format(i,j))]    
    dfcoeff = df_coefficient_C(*kvec,*nuvec,*lvec)    
    Cval = evaluate_df_coefficient_dict(dfcoeff,alpha_ij)
    
    n1 = np.abs(kvec[2]) + np.abs(kvec[4]) + 2 * (nuvec[0] + nuvec[2])
    n2 = np.abs(kvec[3]) + np.abs(kvec[5]) + 2 * (nuvec[1] + nuvec[3])
    m = np.abs(kvec[4]) + np.abs(kvec[5]) + 2 * (nuvec[0] + nuvec[1])
    pwr = 0.5 * (n1 + n2) - m - 1
    prefactor = -G * mi * mj * 2**pwr / aj0
    prefactor *= Lambda_i0**(-0.5 * n1)
    prefactor *= Lambda_j0**(-0.5 * n2)  
    return [PSTerm(prefactor * Cval,k,kbar,p,q),PSTerm(prefactor * Cval,kbar,k,p,-1*q)]


def Perturbation_PSTerm_to_GeneratingFunction_PSTerms(ps_term,omega_vec,domega_vec):
    r"""
    Given a :class:`.PSTerm` object representing a perturbing term in a Hamiltonian,
    create a list of :class:`.PSTerm`s that, when included in a Lie generating function,
    cancel to the perturbing term to leading order, assuming the unperturbed Hamiltonian
    has the form

    .. math:
        H_0 = \sum_{i} \omega_i \delta P_i + \frac{1}{2}\frac{d\omega_i}{d P_i} \delta P_i^2

    as in the case of the Keplerian Hamiltonian.

    Parameters
    ----------
    ps_term : PSTerm
        Represents a term in the perturbing Hamiltonian
    omega_vec : ndarray
        1D array of frequency values
    domega_vec : ndarray
        1D array of the derivative of the i-th frequency w.r.t. the i-th
        momentum.

    Returns
    -------
    list
        List of PSTerm object that together comprise the generating function
        that cancels perturbation term to leading order.
    """
    omega_res = ps_term.q @ omega_vec
    terms = [(- 1j/ omega_res) * ps_term]
    k = ps_term.k
    kbar = ps_term.kbar
    p = ps_term.p
    q = ps_term.q
    M = len(q)
    
    C = ps_term.C
    pre = (1j * C) / omega_res**2
    for i,qi in enumerate(ps_term.q):
        oi = np.zeros(M,dtype=int)
        oi[i] = 1
        if qi!=0:
            Cnew = qi * pre * domega_vec[i]
            term = PSTerm(Cnew,k,kbar,p + oi,q)
            terms.append(term)
    return terms

def PoissonSeries_to_GeneratingFunctionSeries(ps_series,omega_vec,domega_vec):
    new_terms = []
    for term in ps_series.terms:
        new_terms+=Perturbation_PSTerm_to_GeneratingFunction_PSTerms(term,omega_vec,domega_vec)
    return PoissonSeries.from_PSTerms(new_terms,**ps_series._symbol_kwargs)
from .miscellaneous import get_symbol

def get_N_planet_poisson_series_symbols(Npl):
    """
    Create dictionary assigning the symbols used by a 
    :class:`.PoissonSeries` the standard symbols used in the 
    N-planet problem.

    Parameters
    ----------
    Npl : int
        Number of planets

    Returns
    -------
    dict
        symbol assignments
    """
    cvar_symbols = list(sp.symbols(f"x(1:{Npl+1}),y(1:{Npl+1})")) 
    cvar_symbols += [get_symbol(r"\bar{x}",subscript=i) for i in range(1,Npl+1)] 
    cvar_symbols += [get_symbol(r"\bar{y}",subscript=i) for i in range(1,Npl+1)] 
    pvar_symbols = [get_symbol("\delta\Lambda",i,real=True) for i in range(1,Npl+1)]
    thetavar_symbols = sp.symbols(f"lambda(1:{Npl+1})",real=True)
    symbol_kwargs = {
        'cvar_symbols':cvar_symbols,
        'pvar_symbols':pvar_symbols,
        'thetavar_symbols':thetavar_symbols
    }
    return symbol_kwargs
