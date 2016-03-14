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
from scipy.sparse import issparse, csr_matrix, diags
from scipy.sparse.linalg import minres, aslinearoperator, LinearOperator, gmres

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

class LinEq(LinearOperator):

    def __init__(self, A0, Ak, Ak0, K):
        self.K = K
        self.A0 = A0
        self.Ak = Ak
        self.Ak0 = Ak0

        p0 = A0.shape[0]
        self.p0 = p0
        p = Ak.shape[0]
        self.p = p
        n = A0.shape[1]
        self.n = n

        P = p0+(K-1)*p
        self.P = P
        N = K*n
        self.N = N

        self.shape = (P, N)
        self.dtype = A0.dtype

    def _matvec(self, x):
        K = self.K
        P = self.P
        p0 = self.p0
        p = self.p
        n = self.n
        y = np.zeros(P)
        x0 = x[0:n]
        y[0:p0] = self.A0.dot(x0)
        
        for i in range(1, K):
            xi = x[i*n:(i+1)*n]
            y[p0+(i-1)*p:p0+i*p] = self.Ak0.dot(x0) + self.Ak.dot(xi)
        return y

class LinEqT(LinearOperator):
    def __init__(self, A0, Ak, Ak0, K):
        self.K = K
        self.A0T = A0.T.tocsr()
        self.AkT = Ak.T.tocsr()
        self.Ak0T = Ak0.T.tocsr()

        p0 = A0.shape[0]
        self.p0 = p0
        p = Ak.shape[0]
        self.p = p
        n = A0.shape[1]
        self.n = n

        P = p0+(K-1)*p
        self.P = P
        N = K*n
        self.N = N

        self.shape = (N, P)
        self.dtype = A0.dtype

    def _matvec(self, nu):
        K = self.K
        N = self.N
        P = self.P
        p0 = self.p0
        p = self.p
        n = self.n
        y = np.zeros(N)
        
        nu0 = nu[0:p0]
        y[0:n] = self.A0T.dot(nu0)
        for i in range(1, K):
            nui = nu[p0+(i-1)*p:p0+i*p]
            y[i*n:(i+1)*n] = self.AkT.dot(nui)
            y[0:n] += self.Ak0T.dot(nui)
        return y        

class AugmentedSystem(LinearOperator):

    def __init__(self, DFval, A0, Ak, Ak0, G, sig):
        r"""Augmented system for DTRAM

        Parameters
        ----------
        DFval : list
            List of Jacobians, DFval[i] is a sparse matrix
        A0 : sparse matrix
            Linear equality constraints condition zero
        Ak : sparse matrix
            Linear equality constraints k>0
        Ak0 : sparse matrix
            Linear couplings between unbiased and biased condition
        G : sparse matrix
            Linear inequality constraints
        sig : ndarray
            Diagonal of sigma matrix        
        
        """
        self.DFval = DFval
        """Number of subproblems"""
        K = len(DFval)
        self.K = K
        """Dimension of the subproblems"""
        n = A0.shape[1] # Primal point
        self.n = n
        p0 = A0.shape[0] # Equality constraints alpha=0
        self.p0 = p0
        p = Ak.shape[0] # Equality constraints alpha>0
        self.p = p
        m = G.shape[0] # Inequality constraints
        self.m = m

        """Total dimensions of subvectors"""
        N = K*n # Primal
        self.N = N
        P = p0 + (K-1)*p # Multipliers equality
        self.P = P
        M = K*m # Multipliers/slacks inequality
        self.M = M

        self.shape=(N+P, N+P)
        self.dtype = A0.dtype
        
        """Store input"""
        self.A0 = csr_matrix(A0)
        self.Ak = csr_matrix(Ak)
        self.Ak0 = csr_matrix(Ak0)
        self.G = csr_matrix(G)
        self.sig = sig

        self.J = []
        self.diag = np.zeros(N+P)

        for i in range(K):
            SIGi = diags(sig[i*m:(i+1)*m], 0)
            Ji = DFval[i] + (G.T.dot(SIGi)).dot(G)
            self.J.append(Ji)
            self.diag[i*n:(i+1)*n] = Ji.diagonal()

        self.A = LinEq(self.A0, self.Ak, self.Ak0, self.K)
        self.AT = LinEqT(self.A0, self.Ak, self.Ak0, self.K)

    def _mymatvecJ(self, x):
        K = self.K
        n = self.n
        N = self.N
        y = np.zeros(N)
        for i in range(K):
            y[i*n:(i+1)*n] = self.J[i].dot(x[i*n:(i+1)*n])
        return y

    def _matvec(self, z):
        K = self.K
        n = self.n
        p = self.p
        p0 = self.p0

        N = self.N
        P = self.P

        x = z[0:N]
        nu = z[N:]

        y = np.zeros(N+P)
        y[0:N] = self._mymatvecJ(x) + self.AT.dot(nu)
        y[N:] = self.A.dot(x)
        return y                   

    def diagonal(self):
        return self.diag    

def factor(DFval, z, A0, Ak, Ak0, G):    
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

    sig = l/s

    DFval_sp = []
    for i in range(K):
        DPhi = DFval[i*n:(i+1)*n, :]
        DPhi_sp = csr_matrix(DPhi)
        DFval_sp.append(DPhi_sp)

    return AugmentedSystem(DFval_sp, A0, Ak, Ak0, G, sig)    

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

    """Diagonal of Sigma matrix"""
    sig = l/s

    """Dual residuum"""    
    rd = KKTval[0:N]
    
    """Primal residuum (equalities)"""
    rp1 = KKTval[N:N+P]

    """Primal residuum (inequalities)"""
    rp2 = KKTval[N+P:N+P+M]

    """Complementarity"""
    rc = KKTval[N+P+M:]

    """Decmposition is linear operator"""
    W = LU

    """Compute the RHS for augmented system"""
    b1 = np.zeros(N)
    b2 = np.zeros(P)

    for k in range(K):
        sk = s[k*m:(k+1)*m]
        sigk = sig[k*m:(k+1)*m]
        rdk = rd[k*n:(k+1)*n]
        rp2k = rp2[k*m:(k+1)*m]
        rck = rc[k*m:(k+1)*m]
        b1[k*n:(k+1)*n] = rdk + mydot(G.T, sigk*rp2k) - mydot(G.T, rck/sk)

    b2 = rp1
    b = np.hstack((b1, b2))

    """Set up preconditioner"""
    dW = np.abs(W.diagonal())
    dPc = np.ones(W.shape[0])
    ind = (dW > 0.0)
    dPc[ind] = 1.0/dW[ind]
    Pc = diags(dPc, 0)    
    
    dz, info = gmres(W, -b, tol=1e-5)
    dx = dz[0:N]
    dnu = dz[N:]    
    dl = np.zeros(M)
    ds = np.zeros(M)

    for k in range(K):
        ds[k*m:(k+1)*m] = -rp2[k*m:(k+1)*m] - mydot(G, dx[k*n:(k+1)*n])
        dl[k*m:(k+1)*m] = -sig[k*m:(k+1)*m]*ds[k*m:(k+1)*m] - rc[k*m:(k+1)*m]/s[k*m:(k+1)*m]

    return np.hstack((dx, dnu, dl, ds))
    
