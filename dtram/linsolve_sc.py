# -*- coding: utf-8 -*-

# This file is part of dtram.

# Copyright (c) 2016 Benjamin Trendelkamp-Schroer

# dtram is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# dtram is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with dtram.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse, diags, csr_matrix, csc_matrix, bmat
from scipy.sparse.linalg import splu, SuperLU, gmres, minres, LinearOperator

class MyCounter(object):

    def __init__(self):
        self.N = 0

    def count(self, xk):
        self.N += 1
        
class SchurComplement(LinearOperator):

    def __init__(self, W0, LU_Wk, B):
        self.W0 = W0
        self.LU_Wk = LU_Wk
        self.B = B
        self.BT = B.T.tocsr()
        self.shape = W0.shape
        self.dtype = W0.dtype

    def _matvec(self, x):
        W0 = self.W0
        LU_Wk = self.LU_Wk
        B = self.B
        BT = self.BT
        
        y = W0.dot(x)
        for LU in LU_Wk:
            y -= BT.dot(mysolve(LU, B.dot(x)))
        return y                       

# def probe(S):
#     r"""Probe the Schur complement matrix to extract approximate diagonal
#     using only a few matrix vector products.
    
#     """
#     N = S.shape[0]
#     w1 = np.zeros(N)
#     w2 = np.zeros(N)
#     w3 = np.zeros(N)
#     w1[0::3] = 1.0
#     w2[1::3] = 1.0
#     w3[2::3] = 1.0

#     y1 = S.dot(w1)
#     y2 = S.dot(w2)
#     y3 = S.dot(w3)

#     d0 = np.zeros(N)
#     d0[0::3] = y1[0::3]
#     d0[1::3] = y2[1::3]
#     d0[2::3] = y3[2::3]
    
#     return d0

def probe(A, k=1):
    N = A.shape[0]
    d0 = np.zeros(N)
    for i in range(2*k+1):
        v = np.zeros(N)
        v[i::(2*k+1)] = 1.0
        d0[i::(2*k+1)] = A.dot(v)[i::(2*k+1)]
    return d0

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
        return splu(A.tocsc())
    else:
        return lu_factor(A)

def extract_cols(A):
    A = A.tocsc()
    indptr = A.indptr
    """Indices of nonzero cols"""
    cols = np.nonzero(indptr[1:] - indptr[0:-1])[0]
    """Extract nonzero columns"""
    Anz = A[:, cols]
    return Anz.toarray(), cols

def mysolve(LU, b):
    if isinstance(LU, SuperLU):        
        if b.ndim>1:
            bnz, cols = extract_cols(b)
            xnz = LU.solve(bnz)
            # ind = np.any(b != 0.0, axis=0)
            # xnz = LU.solve(b[:, ind])
            x = np.zeros((xnz.shape[0], b.shape[1]))
            x[:, cols] = xnz
            return x        
        else:
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
    H = DFval + (G.T.dot(SIG)).dot(G)    
    W = bmat([[H, A.T], [A, None]], format='csr')        
    return W

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
    B = bmat([[None, csr_matrix((n, p_E0))], [Ak0, None]], format='csr')    

    """Assembel augmented system for alpha=0"""
    l0 = l[0:m]
    s0 = s[0:m]
    DFval0 = DFval[0]
    W0 = assemble_aug(DFval0, l0, s0, A0, G) #S is the LHS for the condensed system
    """Compute LU-factors of augmented system for alpha>0 and update
    LHS of condensed system"""
    LU_W = []
    for k in range(1, K):
        lk = l[k*m:(k+1)*m]
        sk = s[k*m:(k+1)*m]
        DFvalk = DFval[k]        
        """Assemble augmented system"""
        Wk = assemble_aug(DFvalk, lk, sk, Ak, G)

        """Factor"""
        LU_k = myfactor(Wk)

        """Store LU-factors"""
        LU_W.append(LU_k)

    """Set up Schur-complement as LinearOperator"""
    S = SchurComplement(W0, LU_W, B)
    return (S, LU_W)

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
    S, LU_W = LU

    """Set up extended block containing coupling to augmented system
    at alpha=0"""
    B = bmat([[None, csr_matrix((n, p_E0))], [Ak0, None]], format='csr')
    
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
            Winvbk = mysolve(LU_W[k-1], b)
            """Update RHS of condensed system"""
            rhs0 -= mydot(B.T, Winvbk)
            """Store W_k^{-1}b_k"""
            Winvb.append(Winvbk)

    """Increment subvectors for return"""
    dx = np.zeros(N)
    dnu = np.zeros(P)
    dl = np.zeros(M)
    ds = np.zeros(M)

    """Set up the symmetrization matrix"""
    T0 = diags(np.hstack((np.ones(n/2), -1.0*np.ones(n/2+p_E0))), 0)    
    rhs0sym = T0.dot(rhs0)

    """Set up symmetric Schur complement as LinearOperator"""
    def matvec(x):
        return T0.dot(S.dot(x))    
    Ssym = LinearOperator(S.shape, matvec=matvec)

    """Set up Jacobi preconditioning for Ssym"""
    dSsym = np.abs(probe(Ssym, k=1))
    dPc = np.ones(Ssym.shape[0])
    ind = (dSsym > 0.0)
    dPc[ind] = 1.0/dSsym[ind]
    Pc = diags(dPc, 0)    
    mycounter = MyCounter()

    """Compute increment subvectors and assign"""
    for k in range(K):
        if k==0:
            dy0, info = minres(Ssym, -rhs0sym, tol=1e-13, M=Pc, callback=mycounter.count)
            # print mycounter.N
            dy = dy0
            dnu[0:p_E0] = dy0[n:]
        else:
            dy = -Winvb[k-1] - mysolve(LU_W[k-1], mydot(B, dy0))
            dnu[p_E0+(k-1)*p_E:p_E0+k*p_E] = dy[n:]
        dx[k*n:(k+1)*n] = dy[0:n]
        ds[k*m:(k+1)*m] = -rp2[k*m:(k+1)*m] - mydot(G, dx[k*n:(k+1)*n])
        dl[k*m:(k+1)*m] = -sig[k*m:(k+1)*m]*ds[k*m:(k+1)*m] - rc[k*m:(k+1)*m]/s[k*m:(k+1)*m]

    return np.hstack((dx, dnu, dl, ds))
