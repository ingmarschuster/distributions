from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.stats import chi2
import scipy.stats as stats
from .linalg import pdinv


#some functions taken from https://gist.github.com/jfrelinger/2638485


def invwishart_prec_rv(K_0i, nu):
    return inv(wishart_rv(K_0i, nu))

def invwishart_rv(K_0, nu):
    return inv(wishart_rv(inv(K_0), nu))

def invwishart_logpdf(X, S, nu):
    """Compute logpdf of inverse-wishart distribution
    Args:
        X (p x p matrix): rv value for which to compute logpdf
        S (p x p matrix): scale matrix parameter of inverse-wishart distribution (psd Matrix)
        nu: degrees of freedom  nu > p - 1 where p is the dimension of S
    Returns:
        float: logpdf of X given parameters S and nu
    """
    #pdf is \frac{\left|{\mathbf{S}}\right|^{\frac{\nu}{2}}}{2^{\frac{\nu p}{2}}\Gamma_p(\frac{\nu}{2})} \left|\mathbf{X}\right|^{-\frac{\nu+p+1}{2}}e^{-\frac{1}{2}\operatorname{tr}({\mathbf{S}}\mathbf{X}^{-1})}
    p = S.shape[0]    
    assert(len(S.shape) == 2 and
           S.shape[0] == S.shape[1] and
           nu > p - 1)
    if (len(X.shape) != 2 or
        X.shape[0] != X.shape[1] or
        X.shape[0] != S.shape[0]):
        return -np.inf
    nu_h = nu/2
    
    log_S_term = (nu_h * (log(np.linalg.det(S)) - p * log(2))
                   - multigammaln(nu_h, p))
    log_X_term = -(nu + p + 1) / 2 * log(np.linalg.det(X))
    log_e_term = - np.dot(S.T.flat, inv(X).flat) / 2 #using an efficient formula for trace(dot(.,.))
    return log_S_term + log_X_term + log_e_term

def wishart_rv(Ki, nu):
    dim = Ki.shape[0]
    chol = cholesky(Ki)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.arange(1,dim+1)
    foo = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = npr.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

def norm_invwishart_rv(K_0, nu, mu_0, kappa):
    K = invwishart_rv(K_0, nu)
    mu = stats.multivariate_normal.rvs(mu_0, K / kappa)
    return (mu, K)

def norm_invwishart_logpdf(mu, K, K0, nu0, mu0, kappa0):
    K0det = np.linalg.det(K0)
    Kdet = np.linalg.det(K)
    diff = mu - mu0
    diff.shape = (np.prod(diff.shape), 1)
    
    return (-((nu0 + np.max(mu.shape))/2+1) * log(Kdet) +
             (-np.dot(K0.T.flat, Ki.flat) - kappa0 * diff.T.dot(Ki).dot(diff))/2)



class norm_invwishart(object):
    def __init__(self, K0, nu0, mu0, kappa0):
        K0 = np.array(K0)
        mu0 = np.array(mu0)
        assert(len(K0.shape) == 2 and
               K0.shape[0] == K0.shape[1] and
               K0.shape[0] == np.max(mu0.shape))
        (self.K0, self.nu0, self.mu0, self.kappa0) = (K0, nu0, mu0, kappa0)
    
    def rv(self, stacked = False):
        rval =  norm_invwishart_rv(self.K0, self.nu0, self.mu0, self.kappa0)
        if stacked:
            return np.vstack(rval)
        else:
            return rval
            
    def logpdf(self, X):
        if len(args) == 1:
            #mu is stacked upon K
            m = args[0]
            assert(m.shape[0] == m.shape[1] + 1)
            (mu, K) = (m[0,:], m[1:,:])
        elif len(args) == 2:
            (mu, K) = args
        else:
            raise(Exception("expecting arguments mu, K or a matr"))
        return norm_invwishart_logpdf(mu, K, self.K0, self.nu0, self.mu0, self.kappa0)

        
        
class invwishart(object):
    def __init__(self, K0, nu0):
        K0 = np.atleast_2d(K0)
        nu0 = np.array(nu0)
        assert(len(K0.shape) == 2 and
               K0.shape[0] == K0.shape[1])
        (self.K0, self.nu0) = (K0, nu0)
    
    def rv(self):
        return invwishart_rv(self.K0, self.nu0)
            
    def logpdf(self, x):
        return invwishart_logpdf(x, self.K0, self.nu0)