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
            new = PoissonSeries(self.N,self.M)
            new._terms_dict = self._terms_dict.copy()
            for key,val in ps._terms_dict.items():
                new._terms_dict[key] += val
            return new
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(PoissonSeries,type(ps)))
    def __mul__(self,val):
        # Scalar multiplication
        new = PoissonSeries(self.N,self.M)
        for key,coeff in self._terms_dict.items():
            new[key] = val * coeff
        return new
    
    def Lie_deriv(self,ps):
        """
        Compute the Lie derivative of a Poisson series expression
        with respect to this Poisson series.
        """
        if type(ps)==PoissonSeries:
            return bracket(ps,self)
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
  
def bracket(PSeries1,Pseries2):
    N,M = PSeries1.N,PSeries1.M
    
    assert Pseries2.N==N and Pseries2.M==M, \
    "Dimensions of poisson series {} and {} do not match!".format(PSeries1,Pseries2)

    result = PoissonSeries(N,M)
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
