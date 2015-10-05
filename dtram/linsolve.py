import time
import numpy as np

from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, diags, csr_matrix
from scipy.sparse.linalg import splu, SuperLU

def mydot(A, B):
    r"""Dot-product that can handle dense and sparse arrays

    Parameters
    ----------
    A : numpy ndarray or scipy sparse matrix
        The first factor
    B : numpy ndarray or scipy sparse matrix
        The second factor

    Returns
    C : numpy ndarray or scipy sparse matrix
        The dot-product of A and B

    """
    if issparse(A) :
        return A.dot(B)
    elif issparse(B):
        return (B.T.dot(A.T)).T
    else:
        return np.dot(A, B)

def myfactor(A):
    if issparse(A):
        return splu(A)
    else:
        return lu_factor(A)

def mysolve(LU, b):
    if isinstance(LU, SuperLU):
        return LU.solve(b)
    else:
        return lu_solve(LU, b)

###############################################################################
# Utility functions for block-assembly
###############################################################################

def assemble_aug(DFval, l, s, A, G):
    r"""Assemble augmented system"""
    n = G.shape[1]
    n_I = G.shape[0]
    n_E = A.shape[0]

    """Diagonal of Sigma matrix"""
    # sig = l/s
    SIG = diags(l/s, 0)

    """Augmented system"""
    W = np.zeros((n+n_E, n+n_E))

    """Assemble system"""
    # W[0:n, 0:n] = DFval + mydot(G.T, sig[:,np.newaxis]*G)
    W[0:n, 0:n] = DFval + mydot(G.T, mydot(SIG, G))    
    W[0:n, n:] = A.T
    W[n:, 0:n] = A
    return W

def schur_complement(LU_Wk, B):
    """Compute Wk^{-1} Ak0"""
    WkinvAk0 = lu_solve(LU_Wk, B)
    """Compute Schur complement Ak0^T Wk^{-1} Ak0"""
    Wkc = mydot(B.T, WkinvAk0)
    return Wkc

###############################################################################
# Factor and solve for DTRAM system
###############################################################################

def factor(DFval, z, A0, Ak, Ak0, G):    
    r"""Factor DKKT-system for DTRAM using block-structure.

    Parameters
    ----------
    DFval : (N, n) ndarray
        Jacobians of mappings at each thermodynamic state (stacked).
        DFval[i*n:(i+1)*n, :] is the Jacobian at thermodynamic state 
        alpha = i
    z : (N+P+M+M,) ndarray
        Vector of unknowns
    A0 : (n_E0, n) ndarray
        Equality constraints at alpha=0
    Ak : (n_E, n) ndarray
        Equality constraints at alpha>0
    Ak0 : (n_E, n) ndarray
        Coupling between alpha=0 and alpha>0
    G : (n_I, n) ndarray
        Inequality constraints
    
    Returns
    -------
    LU : tuple
        Tuple containing the LU-factors of the condensed LHS, LU_S =
        LU[0] and a list of LU-factors for the augmented systems W at
        each condition alpha>0, LU_W = LU[1]. LU_W[0] are the
        LU-factors for condition alpha=1

    Notes
    -----
    The factorization is computed for the LHS of the augmented system.
    The augmented system can be obtained from the full DKKT-system via
    elimination of the subvector (dl, ds).

    The factorization uses the special block structure present in the
    DKKT system for the DTRAM equations to speed up the factorization 
    process.
    
    """

    """Dimensions of subproblems"""
    n = A0.shape[1] # Primal point
    p_E0 = A0.shape[0] # Equality constraints alpha=0
    p_E = Ak.shape[0] # Equality constraints alpha>0
    m = G.shape[0] # Inequality constraints

    """Total number of subproblems, alpha=0...K-1"""
    K = (z.shape[0] - (n+p_E0+m+m))/(n+p_E+m+m) + 1

    """Total dimensions of subvectors"""
    N = K*n # Primal
    P = p_E0 + (K-1)*p_E # Multipliers equality
    M = K*m # Multipliers/slacks inequality

    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks for inequality constraints"""
    s = z[N+P+M:]

    """Set up extended block containing coupling to augmented system
    at alpha=0"""
    B = np.zeros((n+p_E, n+p_E0))
    B[n:, 0:n] = Ak0

    """Assembel augmented system for alpha=0"""
    l0 = l[0:m]
    s0 = s[0:m]
    DFval0 = DFval[0:n,:]    
    S = assemble_aug(DFval0, l0, s0, A0, G) #S is the LHS for the condensed system

    """Compute LU-factors of augmented system for alpha>0 and update
    LHS of condensed system"""
    LU_W = []
    for k in range(1, K):
        lk = l[k*m:(k+1)*m]
        sk = s[k*m:(k+1)*m]
        DFvalk = DFval[k*n:(k+1)*n, :]        
        """Assemble augmented system"""
        Wk = assemble_aug(DFvalk, lk, sk, Ak, G)

        """Factor"""
        LU_k = lu_factor(Wk)
        
        """Compute Schur complement"""
        SC_k = schur_complement(LU_k, B)
        """Update LHS of condensed system"""
        S -= SC_k
        
        """Store LU-factors"""
        LU_W.append(LU_k)
        
    """Factor LHS of condensed system"""
    LU_S = lu_factor(S)
    
    return (LU_S, LU_W)

def solve_factorized(KKTval, z, LU, A0, Ak, Ak0, G):
    r"""Compute the Newton increment for a DKKT system with prefactored
    LHS.        
    
    """
    
    """Dimensions of subproblems"""
    n = A0.shape[1] # Primal point
    p_E0 = A0.shape[0] # Equality constraints alpha=0
    p_E = Ak.shape[0] # Equality constraints alpha>0
    m = G.shape[0] # Inequality constraints

    """Total number of subproblems, alpha=0...K-1"""
    K = (z.shape[0] - (n+p_E0+m+m))/(n+p_E+m+m) + 1

    """Total dimensions of subvectors"""
    N = K*n # Primal
    P = p_E0 + (K-1)*p_E # Multipliers equality
    M = K*m # Multipliers/slacks inequality

    """Multiplier for inequality constraints"""
    l = z[N+P:N+P+M]

    """Slacks for inequality constraints"""
    s = z[N+P+M:]

    """Dual residuum"""    
    rd = KKTval[0:N]
    
    """Primal residuum (equalities)"""
    rp1 = KKTval[N:N+P]

    """Primal residuum (inequalities)"""
    rp2 = KKTval[N+P:N+P+M]

    """Complementarity"""
    rc = KKTval[N+P+M:]

    """Unpack decomposition"""
    LU_S, LU_W = LU

    """Set up extended block containing coupling to augmented system
    at alpha=0"""
    B = np.zeros((n+p_E, n+p_E0))
    B[n:, 0:n] = Ak0

    """Diagonal of Sigma matrix"""
    sig = l/s

    """Compute RHS of augmented system for all alpha and assemble RHS
    of condensed system, store intermediate results W_k^{-1}b_k, k>0, for
    later use"""
    Winvb = []

    for k in range(K):
        """Extract subvectors"""
        sk = s[k*m:(k+1)*m]
        sigk = sig[k*m:(k+1)*m]
        rdk = rd[k*n:(k+1)*n]
        rp2k = rp2[k*m:(k+1)*m]
        rck = rc[k*m:(k+1)*m]
        if k==0:
            rp1k = rp1[0:p_E0]
        else:
            rp1k = rp1[p_E0+(k-1)*p_E:p_E0+k*p_E]
        """RHS of augmented system"""
        b1 = rdk + mydot(G.T, sigk*rp2k) - mydot(G.T, rck/sk)
        b2 = rp1k
        b = np.hstack((b1, b2))
        if k==0:
            """RHS of condensed system"""
            rhs0 = b
        else:
            """Compute W_k^{-1}b_k"""
            Winvbk = lu_solve(LU_W[k-1], b)
            """Update RHS of condensed system"""
            rhs0 -= mydot(B.T, Winvbk)
            """Store W_k^{-1}b_k"""
            Winvb.append(Winvbk)

    """Increment subvectors for return"""
    dx = np.zeros(N)
    dnu = np.zeros(P)
    dl = np.zeros(M)
    ds = np.zeros(M)

    # """Solve system at condition alpha=0"""
    # dy0 = lu_solve(LU_S, -rhs0)
    # dx[0:n] = dy0[0:n]
    # dnu[0:p_E0] = dy0[n:]
    # ds[0:m] = -rp2[0:m] - mydot(G, dx[0:n])
    # dl[0:m] = -sig[0:m]*ds[0:m] - rc[0:m]/s[0:m]

    """Compute increment subvectors and assign"""
    for k in range(K):
        if k==0:
            dy0 = lu_solve(LU_S, -rhs0)
            dy = dy0
            # dx[0:n] = dy0[0:n]
            dnu[0:p_E0] = dy0[n:]
        else:
            dy = -Winvb[k-1] - lu_solve(LU_W[k-1], mydot(B, dy0))
            dnu[p_E0+(k-1)*p_E:p_E0+k*p_E] = dy[n:]
        dx[k*n:(k+1)*n] = dy[0:n]
        ds[k*m:(k+1)*m] = -rp2[k*m:(k+1)*m] - mydot(G, dx[k*n:(k+1)*n])
        dl[k*m:(k+1)*m] = -sig[k*m:(k+1)*m]*ds[k*m:(k+1)*m] - rc[k*m:(k+1)*m]/s[k*m:(k+1)*m]

    return np.hstack((dx, dnu, dl, ds))
