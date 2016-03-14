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
from scipy.sparse import csr_matrix, diags, bmat
from scipy.sparse.construct import _compressed_sparse_stack
from scipy.sparse.linalg import LinearOperator

cimport cython
cimport numpy as np

from libc.math cimport exp

ctypedef np.int32_t DTYPE_INT_t
ctypedef np.float_t DTYPE_FLOAT_t

def convert_solution(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs):
    cdef size_t M, k, l, j
    cdef double cs_kj, ekj
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] x, y
    cdef np.ndarray[DTYPE_INT_t, ndim=1] indices, indptr
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] data, data_P, diag_P, nu

    if not isinstance(Cs, csr_matrix):
        """Convert to csr_matrix"""
        Cs = csr_matrix(Cs)

    M = Cs.shape[0]
    x = z[0:M]
    y = z[M:]    

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    data_P = np.zeros_like(data)
    diag_P = np.zeros(M)
    nu = np.zeros(M)

    """Loop over rows of Cs"""
    for k in range(M):
        nu[k] = exp(y[k])
        """Loop over nonzero entries in row of Cs"""
        for l in range(indptr[k], indptr[k+1]):
            """Column index of current element"""
            j = indices[l]
            if k != j:
                """Current element of Cs at (k, j)"""
                cs_kj = data[l]
                """Exponential of difference"""
                ekj = exp(y[k]-y[j])
                """Compute off diagonal element"""
                data_P[l] = cs_kj/(x[k] + x[j]*ekj)
                """Update diagonal element"""
                diag_P[k] -= data_P[l]
        diag_P[k] += 1.0

    P = csr_matrix((data_P, indices, indptr), shape=(M, M)) + diags(diag_P, 0)
    return nu/nu.sum(), P    

def F(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs, np.ndarray[DTYPE_FLOAT_t, ndim=1] c):
    r"""Monotone mapping for the reversible MLE problem.

    Parameters
    ----------
    z : (2*M,) ndarray
        Point at which to evaluate mapping, z=(x, y)
    C : (M, M) scipy.sparse matrix
        Count matrix of reversible chain

    Returns
    -------
    Fval : (2*M,) ndarray
        Value of the mapping at z
    
    """    
    cdef size_t M, k, l, j
    cdef double cs_kj, ekj
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] x, y
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] data, Fval
    cdef np.ndarray[DTYPE_INT_t, ndim=1] indices, indptr

    if not isinstance(Cs, csr_matrix):
        """Convert to csr_matrix"""
        Cs = csr_matrix(Cs)    

    M = Cs.shape[0]

    x = z[0:M]
    y = z[M:]

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    Fval = np.zeros(2*M,)

    """Loop over rows of Cs"""
    for k in range(M):
        Fval[k] += 1.0
        Fval[k+M] -= c[k]

        """Loop over nonzero entries in row of Cs"""
        for l in range(indptr[k], indptr[k+1]):
            """Column index of current element"""
            j = indices[l]
            """Current element of Cs at (k, j)"""
            cs_kj = data[l]
            """Exponential of difference"""
            ekj = exp(y[k]-y[j])
            """Update Fx"""
            Fval[k] += -cs_kj/(x[k]+x[j]*ekj)
            """Update Fy"""
            Fval[k+M] -= -cs_kj*x[j]/(x[k]/ekj + x[j])               
    return Fval

def _DF(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs, np.ndarray[DTYPE_FLOAT_t, ndim=1] c):
    r"""Jacobian of the monotone mapping.

    Parameters
    ----------
    z : (2*M,) ndarray
        Point at which to evaluate mapping, z=(x, y)
    C : (M, M) scipy.sparse matrix
        Count matrix of reversible chain

    Returns
    -------
    DFval : (2*M, 2*M) scipy.sparse matrix
        Value of the Jacobian at z
    
    """
    cdef size_t M, k, l, j
    cdef double cs_kj, ekj, tmp1, tmp2
    cdef np.ndarray[np.float_t, ndim=1] x, y
    cdef np.ndarray[np.float_t, ndim=1] data, data_Hxx, data_Hyy, data_Hyx
    cdef np.ndarray[np.float_t, ndim=1] diag_Dxx, diag_Dyy, diag_Dyx
    cdef np.ndarray[np.int32_t, ndim=1] indices, indptr

    if not isinstance(Cs, csr_matrix):
        """Convert to csr_matrix"""
        Cs = csr_matrix(Cs)    

    M = Cs.shape[0]

    x = z[0:M]
    y = z[M:]

    data = Cs.data
    indptr = Cs.indptr
    indices = Cs.indices

    """All subblocks DF_ij can be written as follows, DF_ij = H_ij +
    D_ij. H_ij has the same sparsity structure as C+C.T and D_ij is a
    diagonal matrix, i, j \in {x, y}

    """
    data_Hxx = np.zeros_like(data)
    data_Hyx = np.zeros_like(data)
    data_Hyy = np.zeros_like(data)

    diag_Dxx = np.zeros(M)
    diag_Dyx = np.zeros(M)
    diag_Dyy = np.zeros(M)

    """Loop over rows of Cs"""
    for k in range(M):
        """Loop over nonzero entries in row of Cs"""
        for l in range(indptr[k], indptr[k+1]):
            """Column index of current element"""
            j = indices[l]
            """Current element of Cs at (k, j)"""
            cs_kj = data[l]

            ekj = exp(y[k]-y[j])

            tmp1 = cs_kj/((x[k]+x[j]*ekj)*(x[k]/ekj+x[j]))
            tmp2 = cs_kj/(x[k] + x[j]*ekj)**2

            data_Hxx[l] = tmp1
            diag_Dxx[k] += tmp2

            data_Hyy[l] = tmp1*x[k]*x[j]
            diag_Dyy[k] -= tmp1*x[k]*x[j]

            data_Hyx[l] = -tmp1*x[k]
            diag_Dyx[k] += tmp1*x[j]

    Hxx = csr_matrix((data_Hxx, indices, indptr), shape=(M, M))
    Dxx = diags(diag_Dxx, 0)
    DFxx = Hxx + Dxx

    Hyy = csr_matrix((data_Hyy, indices, indptr), shape=(M, M))
    Dyy = diags(diag_Dyy, 0)
    DFyy = Hyy + Dyy

    Hyx = csr_matrix((data_Hyx, indices, indptr), shape=(M, M))
    Dyx = diags(diag_Dyx, 0)
    DFyx = Hyx + Dyx

    return DFxx, DFyx, DFyy

def DF(np.ndarray[DTYPE_FLOAT_t, ndim=1] z, Cs, np.ndarray[DTYPE_FLOAT_t, ndim=1] c):
    r"""Jacobian of the monotone mapping.

    Parameters
    ----------
    z : (2*M,) ndarray
        Point at which to evaluate mapping, z=(x, y)
    C : (M, M) scipy.sparse matrix
        Count matrix of reversible chain

    Returns
    -------
    DFval : (2*M, 2*M) scipy.sparse matrix
        Value of the Jacobian at z
    
    """
    DFxx, DFyx, DFyy = _DF(z, Cs, c)
    """The call to bmat is really expensive, but I don't know how to avoid it
    if a sparse matrix is desired"""
    DFval = bmat([[DFxx, DFyx.T], [-1.0*DFyx, -1.0*DFyy]]).tocsr()    
    return DFval

def F_dtram(x, Cs, c):
    r"""Mapping for DTRAM.

    Parameters
    ----------
    x : (K*N,) ndarray
        Vector of unknowns
    Cs : list
        List of scipy.sparse matrices. Cs[i]=C[i]+C[i].T, C[i] is the
        countmatrix at at thermodynamic condition i
    c : list
        List of ndarrays. c[i] contains the array of state counts at
        thermodynamic condition i

    Returns
    -------
    Fval : (K*N,) ndarray
        Array of 'gradient' values, F[N*i:N*(i+1)] is the value of the
        reversible MLE mapping at thermodynamic state i
    
    """
    K = len(Cs)
    M = Cs[0].shape[0]
    N = 2*M
    Fval = np.zeros(K*N,)
    for i in range(K):
        xi = x[i*N:(i+1)*N]
        Csi = Cs[i]
        ci = c[i]        
        Fval[i*N:(i+1)*N] = F(xi, Csi, ci)
    return Fval

def DF_dtram(x, Cs, c):
    r"""Jacobian for DTRAM.

    Parameters
    ----------
    x : (K*N,) ndarray
        Vector of unknowns
    Cs : list
        List of scipy.sparse matrices. Cs[i]=C[i]+C[i].T, C[i] is the
        countmatrix at at thermodynamic condition i
    c : list
        List of ndarrays. c[i] contains the array of state counts at
        thermodynamic condition i

    Returns
    -------
    DFval : list
        List of scipy.sparse matrices. DFval[i] is the Jacobian at
        at thermodynamic condition i        
    
    """
    K = len(Cs)
    M = Cs[0].shape[0]
    N = 2*M
    DFval = []
    for i in range(K):
        xi = x[i*N:(i+1)*N]
        Csi = Cs[i]
        ci = c[i]
        DFval.append(DF(xi, Csi, ci))
    return DFval
