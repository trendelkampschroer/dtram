import numpy as np
import scipy.sparse

from objective_sparse import F_dtram, DF_dtram, convert_solution
from linsolve_new import factor, solve_factorized, mydot

__all__=['solve_dtram',]

"""Parameters for primal-dual iteration"""
GAMMA_MIN = 0.0001
GAMMA_MAX = 0.01
GAMMA_BAR = 0.49
KAPPA = 0.01
TAU = 0.5
RHO = min(0.2, min((0.5*GAMMA_BAR)**(1.0/TAU), 1.0-KAPPA))
SIGMA = 0.1
BETA = 100000

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper

def primal_dual_solve(func, x0, Dfunc, A0, b0, Ak, bk, Ak0, Gk, hk,
                      args=(),
                      tol=1e-10,
                      maxiter=100, show_progress=True):
    """Wrap calls to function and Jacobian"""
    fcalls, func = wrap_function(func, args)
    Dfcalls, Dfunc = wrap_function(Dfunc, args)

    """Dimensions of subproblems"""
    n = A0.shape[1] # Primal point
    p_E0 = A0.shape[0] # Equality constraints alpha=0
    p_E = Ak.shape[0] # Equality constraints alpha>0
    m = Gk.shape[0] # Inequality constraints

    """Total number of subproblems, alpha=0...K-1"""
    # K = (z.shape[0] - (n+p_E0+m+m))/(n+p_E+m+m) + 1
    K = x0.shape[0]/n

    """Total dimensions of subvectors"""
    N = K*n # Primal
    P = p_E0 + (K-1)*p_E # Multipliers equality
    M = K*m # Multipliers/slacks inequality
    
    def gap(z):
        r"""Gap-function"""
        l = z[N+P:N+P+M]
        s = z[N+P+M:]
        return mydot(l, s)/M

    def centrality(z):
        r"""Centrality function"""
        l = z[N+P:N+P+M]
        s = z[N+P+M:]
        return np.min(l*s)        

    def KKT(z):
        r"""KKT system for DTRAM"""

        """Dimensions of subproblems"""
        n = A0.shape[1] # Primal point
        p_E0 = A0.shape[0] # Equality constraints alpha=0
        p_E = Ak.shape[0] # Equality constraints alpha>0
        m = Gk.shape[0] # Inequality constraints

        """Total number of subproblems, alpha=0...K-1"""
        K = (z.shape[0] - (n+p_E0+m+m))/(n+p_E+m+m) + 1

        """Total dimensions of subvectors"""
        N = K*n # Primal
        P = p_E0 + (K-1)*p_E # Multipliers equality
        M = K*m # Multipliers/slacks inequality

        """Primal variable"""
        x = z[0:N]

        """Multiplier for equality constraints"""
        nu = z[N:N+P]

        """Multiplier for inequality constraints"""
        l = z[N+P:N+P+M]

        """Slacks"""
        s = z[N+P+M:]

        """Evaluate system"""
        Fval = func(x)

        """Dual residual"""
        rd = np.zeros(N)

        """Primal residual (equalitites)"""
        rp1 = np.zeros(P)

        """Primal residual (inequalities)"""
        rp2 = np.zeros(M)

        """Complementarity"""
        rc = np.zeros(M)

        """Compute residuals at condition, alpha=0"""
        x0 = x[0:n]
        nu0 = nu[0:p_E0]
        l0 = l[0:m]
        s0 = s[0:m]
        F0 = Fval[0:n]
        rd[0:n] = F0 + mydot(A0.T, nu0) + mydot(Gk.T, l0)
        rp1[0:p_E0] = mydot(A0, x0) - b0
        rp2[0:m] = mydot(Gk, x0) - hk + s0
        rc[0:m] = l0*s0

        for i in range(1, K):
            """Extract subvectors"""
            xi = x[i*n:(i+1)*n]
            li = l[i*m:(i+1)*m]
            si = s[i*m:(i+1)*m]
            Fi = Fval[i*n:(i+1)*n]                    
            bi = bk[(i-1)*p_E:i*p_E]
            nui = nu[p_E0+(i-1)*p_E:p_E0+i*p_E]

            """Compute residuals"""
            rd[i*n:(i+1)*n] = Fi + mydot(Ak.T, nui) + mydot(Gk.T, li)
            rp1[p_E0+(i-1)*p_E:p_E0+i*p_E] = mydot(Ak, xi) + mydot(Ak0, x0) - bi
            rp2[i*m:(i+1)*m] = mydot(Gk, xi) - hk + si
            rc[i*m:(i+1)*m] = li*si

            """Update dual residual at condition alpha=0"""
            rd[0:n] += mydot(Ak0.T, nui)

        return np.hstack((rd, rp1, rp2, rc))                                        

    def step_fast(z, KKTval, LU, mu, beta, gamma, alpha0):        
        r"""Affine scaling step."""
        dz_dtram = solve_factorized(KKTval, z, LU, A0, Ak, Ak0, Gk)        

        """Reduce step length until slacks s and multipliers l are positive"""
        alpha = 1.0*alpha_0
        k = 0
        while True:
            z_new = z + alpha*dz_dtram
            if np.all( z_new[N+P:] > 0.0 ):
                break
            alpha *= 0.5
            k += 1
            if k>10:
                raise RuntimeError("Maximum steplength reduction reached")

        """Reduce step length until iterates lie in correct neighborhood"""
        k=0
        while True:
            z_new = z + alpha*dz_dtram
            KKTval_new = KKT(z_new)
            dual = np.linalg.norm(KKTval_new[0:N])
            prim = np.linalg.norm(KKTval_new[N:N+P+M])        
            mu_new = gap(z_new)
            cent_new = centrality(z_new)

            if (dual <= beta*mu_new and prim <= beta*mu_new and
                cent_new >= gamma*mu_new):
                break
            alpha *= 0.5
            # alpha *= 0.95
            k += 1
            if k>10:
                raise RuntimeError("Maximum steplength reduction reached")
        return z_new, mu_new

    def step_safe(z, KKTval, LU, mu, beta, gamma):
        r"""Centering step."""
        dz_dtram = solve_factorized(KKTval, z, LU, A0, Ak, Ak0, Gk)                

        """Reduce step length until slacks s and multipliers l are positive"""
        alpha = 1.0
        k = 0
        while True:
            z_new = z + alpha*dz_dtram
            if np.all( z_new[N+P:] > 0.0 ):
                break
            alpha *= 0.5
            k += 1
            if k>10:
                raise RuntimeError("Maximum steplength reduction (pos.) reached")

        """Reduce step length until iterates lie in correct neighborhood
        and mu fulfills Armijo condition"""
        k=0
        while True:
            z_new = z+alpha*dz_dtram
            KKTval_new = KKT(z_new)
            dual = np.linalg.norm(KKTval_new[0:N])
            prim = np.linalg.norm(KKTval_new[N:N+P+M])
            mu_new = gap(z_new)
            cent_new = centrality(z_new)

            if (dual <= beta*mu_new and prim <= beta*mu_new and
                cent_new >= gamma*mu_new and
                mu_new<=(1.0-KAPPA*alpha*(1.0-SIGMA))*mu):
                break
            alpha *= 0.5
            k += 1
            if k>10:
                raise RuntimeError("Maximum steplength reduction reached")
        return z_new, mu_new

    """INITIALIZATION"""
    
    """Initial Slacks for inequality constraints"""
    s0 = np.zeros(M)
    for i in range(K):
        s0[i*m:(i+1)*m] = -1.0*(mydot(Gk, x0[i*n:(i+1)*n]) - hk)

    """Initial multipliers for inequality constraints"""
    l0 = 1.0*np.ones(M)

    """Initial multipliers for equality constraints"""
    nu0 = np.zeros(P)
    
    """Initial point"""
    z0 = np.hstack((x0, nu0, l0, s0))

    """Initial KKT-values"""
    KKTval0 = KKT(z0)
    mu0 = gap(z0)
    dual0 = np.linalg.norm(KKTval0[0:N])
    prim0 = np.linalg.norm(KKTval0[N:N+P+M])

    """Initial neighborhood"""
    beta = BETA * np.sqrt(dual0**2 + prim0**2)/mu0
    gamma = 1.0 * GAMMA_MAX

    """Number of fast steps"""
    t = 0

    """Number of iterations"""
    niter = 0

    """Dummy variable for step type"""
    step_type = " "

    if show_progress:
        print "%s %s %s %s %s %s %s" %("iter", "gap", "dual", "primal",
                                       "min", "max", "step")
    """MAIN LOOP"""
    z = z0
    x = z0[0:N]
    KKTval = KKTval0
    dual = dual0
    prim = prim0
    mu = mu0
    Dfunc_val = Dfunc(x)
    LU = factor(Dfunc_val, z, A0, Ak, Ak0, Gk)
    while True:
        if show_progress:
            l=z[N+P:N+P+M]
            s=z[N+P+M:]
            print "%i %.6e %.6e %.6e %.6e %.6e %s" %(niter+1, mu, dual, prim,
                                                     np.min(l*s), np.max(l*s),
                                                     step_type)
        """Attempt fast step"""
        beta_new = (1.0 + GAMMA_BAR**(t+1)) * beta
        gamma_new = GAMMA_MIN + GAMMA_BAR**(t+1)*(GAMMA_MAX - GAMMA_MIN)
        alpha_0 = 1.0 - np.sqrt(mu)/GAMMA_BAR**t

        if alpha_0 > 0.0:
            z_new, mu_new = step_fast(z, KKTval, LU, mu,
                                      beta_new, gamma_new, alpha_0)
            if mu_new < RHO * mu:
                """Fast successful"""
                z = z_new
                mu = mu_new
                beta = beta_new
                gamma = gamma_new
                t += 1
                step_type = "f"
            else:
                """Perturbed right-had side"""
                KKTval_pert = 1.0*KKTval
                KKTval_pert[N+P+M:] -= SIGMA * mu
                z, mu = step_safe(z, KKTval_pert, LU, mu, beta, gamma)
                step_type = "s"
        else:
            """Perturbed right-hand side"""
            KKTval_pert = 1.0*KKTval
            KKTval_pert[N+P+M:] -= SIGMA * mu
            z, mu = step_safe(z, KKTval_pert, LU, mu, beta, gamma)
            step_type = "s"

        """Compute new iterates"""
        KKTval = KKT(z)
        dual = np.linalg.norm(KKTval[0:N])
        prim = np.linalg.norm(KKTval[N:N+P+M])
        x = z[0:N]
        Dfunc_val = Dfunc(x)
        LU = factor(Dfunc_val, z, A0, Ak, Ak0, Gk)        
        niter += 1
        if (mu < tol and dual < tol and prim < tol):
            break
        if niter > maxiter:
            raise RuntimeError("Maximum number of iterations reached")
        
    if show_progress:
            l=z[N+P:N+P+M]
            s=z[N+P+M:]
            print "%i %.6e %.6e %.6e %.6e %.6e %s" %(niter+1, mu, dual, prim,
                                                     np.min(l*s), np.max(l*s),
                                                     step_type)
    return z[0:N]                                      

def solve_dtram(C, u, tol=1e-10, maxiter=100, show_progress=True):
    K = C.shape[0] # Number of thermodynamic states
    M = C.shape[1] # Number of discrete states

    """Set up parameter"""
    c = []
    Cs = []
    for i in range(K):
        Ci = C[i,:, :]
        ci = Ci.sum(axis=0)
        Csi = scipy.sparse.csr_matrix(Ci+Ci.T)
        c.append(ci)
        Cs.append(Csi)    

    """Inequality constraints"""
    # Gk = np.zeros((M, 2*M))
    # Gk[:,0:M] = -1.0 * np.eye(M)
    Gk = -1.0*scipy.sparse.eye(M, n=2*M, k=0)
    hk = np.zeros((M,))

    """Equality constraints at alpha=0"""
    A0 = np.zeros((1, 2*M))
    A0[0, M] = 1.0
    b0 = np.array([0.0])

    A0 = scipy.sparse.csr_matrix(A0)

    """Equality constraints at alpha>0"""
    Ak = np.zeros((M, 2*M))
    Ak[:, M:] = np.eye(M)

    Ak = scipy.sparse.csr_matrix(Ak)

    """Coupling constraints"""
    Ak0 = np.zeros((M, 2*M))
    Ak0[:,M:] = -1.0*np.eye(M)

    Ak0 = scipy.sparse.csr_matrix(Ak0)

    """RHS for equality constraints, alpha>0"""
    bk = np.zeros((K-1)*M)
    for i in range(K-1):
        bk[i*M:(i+1)*M] = u[i, :]

    """Initial guess (primal variables)"""
    xstarti = np.zeros((2*M))
    xstarti[0:M] = 1.0

    """Repeat same initial guess K-times"""
    xstart = np.tile(xstarti, K)

    x = primal_dual_solve(F_dtram, xstart, DF_dtram, A0, b0,
                          Ak, bk, Ak0, Gk, hk, args=(Cs, c), tol=tol,
                          maxiter=maxiter, show_progress=show_progress)

    """Extract optimal point at alpha=0"""
    x0 = x[0:2*M]

    # """Count matrix at alpha=0"""
    # C0 = C[0, :, :]
    """Parameters at alpha=0"""
    Cs0 = Cs[0]
    c0 = c[0]

    """Compute pi0 and P0"""
    pi0, P0 = convert_solution(x0, Cs0)

    return pi0, P0


    
    

    



    
