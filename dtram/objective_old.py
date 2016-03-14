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

r"""Gradient and Hessian for DTRAM"""
import numpy as np

###############################################################################
# Reversible MLE
###############################################################################
r"""Gradient and Hessian for reversible MLE"""

def F(z, C):
    r"""Gradient for reversible MLE"""
    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]
    q=np.exp(y)
    C_sym=C+C.transpose()
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Z=W+W.transpose()   
    Fx=-1.0*np.sum(C_sym*q[np.newaxis, :]/Z, axis=1)+1.0
    Fy= -1.0*np.sum(C_sym*W.transpose()/Z, axis=1)+np.sum(C, axis=0)
    return np.hstack((Fx, -1.0*Fy))

def DF(z, C):
    r"""Hessian for reversible MLE"""
    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]
    
    q=np.exp(y)

    C_sym=C+C.transpose()
    W=x[:,np.newaxis]*q[np.newaxis,:]
    Wt=W.transpose()
    Z=W+Wt

    Z2=Z**2
    Q=q[:,np.newaxis]*q[np.newaxis,:]

    dxx=np.sum(C_sym*(q**2)[np.newaxis,:]/Z2, axis=1)
    DxDxf= np.diag(dxx)+C_sym*Q/Z2

    dxy=np.sum(C_sym*(x*q)[:,np.newaxis]*q[np.newaxis,:]/Z2, axis=0)
    DyDxf=-1.0*C_sym*q[np.newaxis,:]/Z + C_sym*(W*q[np.newaxis,:])/Z2+np.diag(dxy)
    
    DxDyf=DyDxf.transpose()
    
    Dyy1=-1.0*C_sym*W/Z
    Dyy2=C_sym*W**2/Z2
    dyy=np.sum(Dyy1, axis=0)+np.sum(Dyy2, axis=0)
    
    DyDyf=np.diag(dyy)+C_sym*W*Wt/Z2

    J=np.zeros((N, N))
    J[0:N/2, 0:N/2]=DxDxf
    J[0:N/2, N/2:]=DyDxf
    J[N/2:, 0:N/2]=-1.0*DxDyf
    J[N/2:, N/2:]=-1.0*DyDyf
    return J

def convert_solution(z, C):
    N=z.shape[0]
    x=z[0:N/2]
    y=z[N/2:]

    w=np.exp(y)
    pi=w/w.sum()

    # X=pi*np.transpose(x)
    X=pi[:,np.newaxis]*x[np.newaxis,:]
    Y=X+np.transpose(X)
    denom=Y
    enum=(C+np.transpose(C))*np.transpose(pi)
    P=enum/denom
    ind=np.diag_indices(C.shape[0])
    P[ind]=0.0
    rowsums=P.sum(axis=1)
    P[ind]=1.0-rowsums
    return pi, P

###############################################################################
# DTRAM
###############################################################################

def F_dtram(x, C):
    r"""'Gradient' for DTRAM.

    Parameters
    ----------
    x : (M, N) ndarray
        Vector of unknowns.
    C : (M, N, N) ndarray
        Array of count matrices, C[i, :, :] is the count-matrix at
        thermodynamic state i

    Returns
    -------
    Fval : (M, N) ndarray
        Array of 'gradient' values, F[i, :] is the reversible MLE
        'gradient' at thermodynamic state i.
    
    """
    K = C.shape[0]
    M = C.shape[1]
    N = 2*M
    Fval = np.zeros(K*N,)
    for i in range(K):
        xi = x[i*N:(i+1)*N]
        Ci = C[i, :, :]
        Fval[i*N:(i+1)*N] = F(xi, Ci)
    return Fval

def DF_dtram(x, C):
    r"""Jacobian for DTRAM.

    Parameters
    ----------
    z : (M, N) ndarray
        Vector of unknowns.
    C : (M, N, N) ndarray
        Array of count matrices, C[i, :, :] is the count-matrix at
        thermodynamic state i

    Returns
    -------
    DFval : (M, N, N) ndarray
        Array of Jacobian values, DFval[i, :, :] is the reversible MLE
        Jacobian at thermodynamic state i.
    
    """
    K = C.shape[0]
    M = C.shape[1]
    N = 2*M
    DFval = np.zeros((K*N, N))
    for i in range(K):
        xi = x[i*N:(i+1)*N]
        Ci = C[i, :, :]
        DFval[i*N:(i+1)*N, :] = DF(xi, Ci)
    return DFval
