import numpy as np

from objective import F_dtram, DF_dtram

if __name__=="__main__":
    # C0 = 1.0*np.load('C0.npy')
    # C1 = 1.0*np.load('C1.npy')

    C0 = np.array([[9985.0, 5.0, 0], [5.0, 0, 9.0], [0, 9.0, 85.0]])
    C1 = np.array([[166.0, 166.0, 0], [165.0, 0.0, 177.0], [0, 177.0, 148.0]])

    N = C0.shape[1]
    C = np.zeros((2, N, N))
    C[0, :, :] = C0/C0.max()
    C[1, :, :] = C1/C1.max()

    z0 = np.zeros(2*N)
    z0[0:N] = 1.0
    z1 = np.zeros(2*N)
    z1[0:N] = 1.0
    z = np.hstack((z0, z1))

    """Inequality constraints"""
    G = np.zeros((2, N, 2*N))
    G[:, np.arange(N), np.arange(N)] = -1.0
    h = np.zeros((2, N))

    A0 = np.zeros((1, 2*N))
    A0[0, N]=1.0
    b0 = np.array([0.0])

    Id_prime = np.zeros((N, 2*N))
    Id_prime[0:N,N:] = np.eye(N)

    A1 = Id_prime
    A10 = -Id_prime

    Fval = F_dtram(z, C)
    DFval = DF_dtram(z, C)
    
    from scipy.linalg import eigvals, lu_factor, lu_solve

    """Slacks"""
    G0 = G[0, :, :]
    h0 = h[0, :]
    s0 = -1.0*(np.dot(G0, z0)-h0)
    l0 = np.ones(s0.shape)    
    sig0 = l0/s0
    H0 = DFval[0, :, :] + np.dot(G0.T, sig0[:,np.newaxis]*G0)
    W0 = np.zeros((2*N+1, 2*N+1))
    W0[0:2*N,0:2*N] = H0
    W0[0:2*N,2*N:] = A0.T
    W0[2*N:,0:2*N] = A0
    print np.linalg.cond(W0)    

    G1 = G[0, :, :]
    h1 = h[0, :]
    s1 = -1.0*(np.dot(G1, z1)-h1)
    l1 = np.ones(s1.shape)    
    sig1 = l1/s1
    H1 = DFval[1, :, :] + np.dot(G1.T, sig0[:,np.newaxis]*G1)
    
    W1 = np.zeros((2*N+N, 2*N+N))
    W1[0:2*N,0:2*N] = H1
    W1[0:2*N,2*N:] = A1.T
    W1[2*N:,0:2*N] = A1

    print np.linalg.cond(W1)

    print H0.shape, A0.shape
    print H1.shape, A1.shape     
    
    B = np.zeros((2*N+N, 2*N+1))
    B[2*N:,0:2*N] = A10

    LU_1 = lu_factor(W1)
    H1invB = lu_solve(LU_1, B)
    H1_sc = np.dot(B.T, H1invB)

    S = W0 - H1_sc
    print np.linalg.cond(S)
    lu_factor(S)
