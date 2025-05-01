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
    def as_series(self,**kwargs):
        """
        Get a :class:`.PoissonSeries` object representing a series comprised of
        a single term.

        Returns
        -------
        PoissonSeries
            Series version of this term.
        """
        return PoissonSeries.from_PSTerms([self],**kwargs)

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
    #define multiplication by another poisson series
    def _Pseries_multiply(self,pseries):
        terms=[PSTerm(t1.C * t2.C, t1.k + t2.k, t1.kbar + t2.kbar,t1.p + t2.p,t1.q + t2.q) for t1 in self.terms for t2 in pseries.terms]
        return PoissonSeries.from_PSTerms(terms)
    # define scalar multiply term-wise
    def __mul__(self,val):
        if type(val)==type(self):
            return self._Pseries_multiply(val)
        if len(self.terms)==0:
            return self
        return PoissonSeries.from_PSTerms([term * val for term in self.terms],**self._symbol_kwargs)
        # Scalar multiplication
    # scalar multiplication is commutative
    __rmul__ = __mul__

    @property
    def conj(self):
        cterms = [PSTerm(np.conj(t.C),t.kbar,t.k,t.p,-1*t.q) for t in self.terms]
        return PoissonSeries.from_PSTerms(cterms)
        
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
    """
    Compute the Poisson bracket of a pair of Poisson series

    Parameters
    ----------
    PSeries1 : PoissonSeries
        First Poisson series
    Pseries2 : PoissonSeries
        Second Poisson series

    Returns
    -------
    PoissonSeries
        Poisson series representing the Poisson bracket of the input series.
    """
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

def birkhoff_normalize(omega_vec,H,lmax,kres = []):
    """
    Given an input frequency vector and Hamiltonian, carry out the Birkhoff
    normalization procedure up to maximum specified order.

    Parameters
    ----------
    omega_vec : 1-d array
        Frequency of 
    H : dict
        Dictionary containing terms of the Hamiltonian grouped by order. The
        keys of the dictionary denote the order of the term in powers of complex
        canonical variables. The values are PoissonSeries objects.
    lmax : int
        Order up to which the Birkhoff normalization should be carried out.
    kres : list, optional
        List of resonant wave vectors to retain in the transformed Hamiltonian.
        Default is none.

    Returns
    -------
    chi : dict
        Dictionary containing terms of the generating function. The keys of the
        dictionary denote the order of the term in powers of complex canonical
        variables. The values are PoissonSeries objects.
    Hav : dict
        Dictionary containing terms of the averaged Hamiltonian. The keys of the
        dictionary denote the order of the term in powers of complex canonical
        variables. The values are PoissonSeries objects.
    """
    N = len(omega_vec)
    chi,Upsilon,Hav = [defaultdict(lambda: PoissonSeries(N,0)) for _ in range(3)]
    Hav[2] += H[2]
    for l in range(2,lmax+1):
        Upsilon[(0,l)] += H[l]
        Psi = PoissonSeries(N,0)
        Psi+= Upsilon[(0,l)]        
        for n in range(1,l-1):
            kmax = l+1-n if n>1 else l-n
            for k in range(3,kmax+1):
                Upsilon[(n,l)]+=chi[k].Lie_deriv(Upsilon[(n-1,l+2-k)])
            Psi += Upsilon[(n,l)]*(1/factorial(n))
        if l>2:
            chi[l],Hav[l] = Psi_to_chi_and_Hav(omega_vec,Psi,kres)
            Upsilon[(1,l)]+=chi[l].Lie_deriv(Upsilon[(0,2)])
    return chi,Hav

def expL(f,chi,lmax=None):
    """
    Calculate a finite-order truncation of the exponential of the Lie derivative
    operator applied to a function.

    Parameters
    ----------
    f : dict
        Dictionary containing terms in the expansion of the target function
        grouped by order. The keys of the dictionary denote the order of the
        term in powers of complex canonical variables. The values are
        PoissonSeries objects.    
    chi : dict 
        Dictionary containing generating function terms grouped by order.
    lmax : int, optional
        maximum order of the finite-order truncation. Defaults to value set by
        maximum order terms appearing in f and chi.

    Returns
    -------
    dict
        Dictionary containing terms of the expansion of the transformed function
        grouped by order.
    """
    kmin = min(chi.keys())
    k1min = min(f.keys())
    Upsilon = defaultdict(chi.default_factory)
    E = defaultdict(chi.default_factory)
    if not lmax:
        lmax = max(chi.keys()) + k1min - 2
    for l in range(k1min,lmax+1):
        Upsilon[0,l] += f[l]
        E[l] += Upsilon[0,l]
        nmax = (l-k1min) // (kmin-2)
        lmin_n = k1min
        for n in range(1,nmax+1):
            kmax = l+2-lmin_n
            for k in range(kmin,kmax+1):
                Upsilon[(n,l)]+=chi[k].Lie_deriv(Upsilon[(n-1,l+2-k)])
            lmin_n = kmin + lmin_n - 2 
            E[l] += Upsilon[(n,l)] * (1/factorial(n))
    return E

def expLinv(f,chi,lmax=None):
    """
    Calculate a finite-order truncation of the exponential of the inverse of the
    Lie derivative operator applied to a function.

    Parameters
    ----------
    f : dict
        Dictionary containing terms in the expansion of the target function
        grouped by order. The keys of the dictionary denote the order of the
        term in powers of complex canonical variables. The values are
        PoissonSeries objects.    
    chi : dict 
        Dictionary containing generating function terms grouped by order.
    lmax : int, optional
        maximum order of the finite-order truncation. Defaults to value set by
        maximum order terms appearing in f and chi.

    Returns
    -------
    dict
        Dictionary containing terms of the expansion of the transformed function
        grouped by order.
    """
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
    Lambda_i0 = float(pham.H_params[pham.Lambda0s[i]])
    Lambda_j0 = float(pham.H_params[pham.Lambda0s[j]])
    alpha_ij = float(pham.H_params[sp.symbols(r"\alpha_{{{0}\,{1}}}".format(i,j))])
    dfcoeff = df_coefficient_C(*kvec,*nuvec,*lvec)    
    Cval = evaluate_df_coefficient_dict(dfcoeff,alpha_ij)
    n1 = np.abs(kvec[2]) + np.abs(kvec[4]) + 2 * (nuvec[0] + nuvec[2])
    n2 = np.abs(kvec[3]) + np.abs(kvec[5]) + 2 * (nuvec[1] + nuvec[3])
    m = np.abs(kvec[4]) + np.abs(kvec[5]) + 2 * (nuvec[0] + nuvec[1])
    pwr = 0.5 * (n1 + n2) - m - 1
    prefactor = -G * mi * mj * 2**pwr / aj0
    prefactor *= Lambda_i0**(-0.5 * n1 - lvec[0])
    prefactor *= Lambda_j0**(-0.5 * n2 - lvec[1])
    return [PSTerm(complex(prefactor * Cval),k,kbar,p,q),PSTerm(complex(prefactor * Cval),kbar,k,p,-1*q)]


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
    pvar_symbols = [get_symbol(r"\delta\Lambda",i,real=True) for i in range(1,Npl+1)]
    thetavar_symbols = sp.symbols(f"lambda(1:{Npl+1})",real=True)
    symbol_kwargs = {
        'cvar_symbols':cvar_symbols,
        'pvar_symbols':pvar_symbols,
        'thetavar_symbols':thetavar_symbols
    }
    return symbol_kwargs

_RT2 = np.sqrt(2)
_RT2_INV = 1/_RT2
class PoissonSeriesHamiltonian:
    r"""
    Represents a Hamiltonian and its corresponding equations of motion expressed as a Poisson series.

    The Hamiltonian and equations of motion are assumed to be functions of the real variables
    .. math::
        y_1, \dots, y_N,\ \theta_1, \dots, \theta_M,\ x_1, \dots, x_N,\ p_1, \dots, p_M

    These are mapped to complex Poisson series variables via
    .. math::
        z_j = \frac{x_j - i y_j}{\sqrt{2}}

    Parameters
    ----------
    hamiltonian_poisson_series : PoissonSeries
        The Hamiltonian expressed as a Poisson series in terms of the complex variables.
    """

    def __init__(self, hamiltonian_poisson_series):
        self.hamiltonian = hamiltonian_poisson_series
        self._cvar_flow = _hamiltonian_series_to_flow_series_list(hamiltonian_poisson_series)
        self._cvar_real_flow_jacobian = _real_vars_jacobian_series(self._cvar_flow)

    @property
    def N(self):
        """Number of canonical (z, z̄) variables (i.e., degrees of freedom related to x, y)."""
        return self.hamiltonian.N

    @property
    def M(self):
        """Number of canonical (Q, P) variables (i.e., angle-action type coordinates)."""
        return self.hamiltonian.M

    def _real_vars_to_cvars(self, real_vars):
        """
        Convert the real-valued state vector to complex canonical variables.

        Parameters
        ----------
        real_vars : array_like
            A 1D array of real variables in the order:
            [y₁, ..., y_N, Q₁, ..., Q_M, x₁, ..., x_N, P₁, ..., P_M]

        Returns
        -------
        z : ndarray
            Complex canonical variables defined as z_j = (x_j - i y_j) / sqrt(2)
        P : ndarray
            Canonical momenta corresponding to Q
        Q : ndarray
            Canonical coordinates corresponding to P
        """
        y = real_vars[:self.N]
        Q = real_vars[self.N:self.N + self.M]
        x = real_vars[self.N + self.M:2 * self.N + self.M]
        P = real_vars[2 * self.N + self.M:2 * self.N + 2 * self.M]
        z =   (x - 1j * y) / _RT2
        return z, P, Q

    def __call__(self, real_vars):
        """
        Evaluate the Hamiltonian at the given point in real variable space.

        Parameters
        ----------
        real_vars : array_like
            Real-valued phase space coordinates.

        Returns
        -------
        float
            Value of the Hamiltonian.
        """
        z, P, Q = self._real_vars_to_cvars(real_vars)
        return self.hamiltonian(z, P, Q)

    def flow(self, real_vars):
        """
        Compute the time derivatives of the real-valued state vector (Hamiltonian flow).

        Parameters
        ----------
        real_vars : array_like
            Real-valued phase space coordinates.

        Returns
        -------
        ndarray
            Time derivatives [dy/dt, dQ/dt, dx/dt, dP/dt] of the real variables.
        """
        z, P, Q = self._real_vars_to_cvars(real_vars)
        zdot = np.array([f(z, P, Q) for f in self._cvar_flow[:self.N]])
        Pdot = np.real([f(z, P, Q) for f in self._cvar_flow[self.N:self.N + self.M]])
        Qdot = np.real([f(z, P, Q) for f in self._cvar_flow[self.N + self.M:]])
        return np.concatenate((-np.imag(zdot) * _RT2, Qdot, np.real(zdot) * _RT2, Pdot))

    def jacobian(self, real_vars):
        """
        Compute the Jacobian matrix of the Hamiltonian flow at the given state.

        Parameters
        ----------
        real_vars : array_like
            Real-valued phase space coordinates.

        Returns
        -------
        ndarray
            The Jacobian matrix of the flow, shape (2N+2M, 2N+2M)
        """
        z, P, Q = self._real_vars_to_cvars(real_vars)
        jj = np.array([[series(z, P, Q) for series in row] for row in self._cvar_real_flow_jacobian])
        dzdot_dvar = jj[:self.N, :]
        dxdot_dvar = _RT2 * np.real(dzdot_dvar)
        dydot_dvar = -_RT2 * np.imag(dzdot_dvar)
        dPdot_dvar = np.real(jj[self.N:self.N + self.M, :])
        dQdot_dvar = np.real(jj[self.N + self.M:, :])
        return np.vstack([dydot_dvar, dQdot_dvar, dxdot_dvar, dPdot_dvar])


def _hamiltonian_series_to_flow_series_list(ham_series):
    I_N = np.eye(ham_series.N,dtype=int)
    zero_N = np.zeros(ham_series.N,dtype=int)
    I_M = np.eye(ham_series.M,dtype=int)
    zero_M = np.zeros(ham_series.M,dtype=int)
    flow_series_list = []
    # dx/dt
    for i in range(ham_series.N):
        var_series = PSTerm(1,I_N[i],zero_N,zero_M,zero_M).as_series()
        dxi_dt = ham_series.Lie_deriv(var_series)
        flow_series_list.append(dxi_dt)
    # dP/dt 
    for i in range(ham_series.M):
        var_series = PSTerm(1,zero_N,zero_N,I_M[i],zero_M).as_series()
        dPi_dt = ham_series.Lie_deriv(var_series)
        flow_series_list.append(dPi_dt)
    # dQ/dt 
    for i in range(ham_series.M):
        var_series = PSTerm(1,zero_N,zero_N,zero_M,I_M[i]).as_series()
        dexp_iQ_dt = ham_series.Lie_deriv(var_series)
        factor = PSTerm(-1j,zero_N,zero_N,zero_M,-I_M[i]).as_series()
        dQi_dt = factor * dexp_iQ_dt
        flow_series_list.append(dQi_dt)
    return flow_series_list
def dseries_dQi(series,i):
    return PoissonSeries.from_PSTerms([1j * term.q[i] * term for term in series.terms if term.q[i]], N=series.N, M=series.M)
def dseries_dPi(series,i):
    one_i = np.eye(series.M)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.p[i], term.k,term.kbar,term.p - one_i,term.q) for term in series.terms if term.p[i]], N=series.N, M=series.M)
def dseries_dzi(series,i):
    one_i = np.eye(series.N)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.k[i], term.k - one_i,term.kbar,term.p,term.q) for term in series.terms if term.k[i]], N=series.N, M=series.M)
def dseries_dzbari(series,i):
    one_i = np.eye(series.N)[i]
    return PoissonSeries.from_PSTerms([PSTerm(term.C * term.kbar[i], term.k,term.kbar - one_i,term.p,term.q) for term in series.terms if term.kbar[i]], N=series.N, M=series.M)

def _real_vars_jacobian_series(complex_flow_series):
    flow0 = complex_flow_series[0]
    N,M = flow0.N,flow0.M
    N_dim = 2 * (N+M)
    jac = [[None for j in range(N_dim)] for i in range(N+2*M)]
    for i,flow_i in enumerate(complex_flow_series):
        for j in range(N):
            df_dz = dseries_dzi(flow_i,j)
            df_dzbar = dseries_dzbari(flow_i,j)
            df_dy = 1j * _RT2_INV * (df_dzbar + -1*df_dz)
            df_dx = _RT2_INV * (df_dzbar + df_dz)
            jac[i][j] = df_dy
            jac[i][N+M+j] = df_dx
        for j in range(M):            
            df_dP = dseries_dPi(flow_i,j)
            df_dQ = dseries_dQi(flow_i,j)
            jac[i][N+j] = df_dQ
            jac[i][2*N+M+j] = df_dP
    return jac

