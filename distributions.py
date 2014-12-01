from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.random as npr
from numpy.linalg import inv, cholesky, det
from scipy.special import multigammaln
from scipy.stats import chi2
import scipy.stats as stats
from linalg import pdinv

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


class mvnorm(object):
    def __init__(self, mu, K):        
        assert(np.prod(mu.shape) == K.shape[0] )
        assert(K.shape[0] == K.shape[1])
        self.mu = mu
        self.K = K
        (self.Ki, self.L, self.Li, self.logdet) = pdinv(K)
        self.freeze = stats.multivariate_normal(mu, K)
        
    def ppf(self, component_cum_prob):
        #this is a pointwise ppf
        std_norm = stats.norm(0, 1)
        rval = []
        for r in range(component_cum_prob.shape[0]):
            rval.append(self.mu + self.L.dot(std_norm.ppf(component_cum_prob[r, :])))
        return np.array(rval)
    
    def logpdf(self, x):
        return self.freeze.logpdf(x)
    
    def logpdf_grad(self, x):
        return self.Ki.dot(np.atleast_1d(x) - self.mu)
    
    def rvs(self, *args, **kwargs):
        return self.freeze.rvs(*args, **kwargs)
    
    @classmethod
    def fit(cls, samples): # observations expected in rows
        mu = samples.mean(0)
        return (mu, ensure_2d(np.cov(samples, rowvar = 0)))


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
        K0 = np.array(K0)
        nu0 = np.array(nu0)
        assert(len(K0.shape) == 2 and
               K0.shape[0] == K0.shape[1])
        (self.K0, self.nu0) = (K0, nu0)
    
    def rv(self):
        return invwishart_rv(self.K0, self.nu0)
            
    def logpdf(self, x):
        return invwishart_logpdf(x, self.K0, self.nu0)
        
#################################
        
def test_invwishart_logpdf():
    # values from R-package bayesm, function lndIWishart(6.1, a, a)
    a = 4 * np.eye(5)
    assert(abs(invwishart_logpdf(a,a,6.1) + 40.526062) < 1*10**-5)
    
    a = np.eye(5) + np.ones((5,5))
    assert(abs(invwishart_logpdf(a,a,6.1) + 25.1069258) < 1*10**-6)
    
    a = 2 * np.eye(5)
    assert(abs(invwishart_logpdf(a,a,6.1) + 30.12885519) < 1*10**-7)

    
if __name__ == '__main__':
    npr.seed(1)
    nu = 5
    a = np.array([[1,0.5,0],[0.5,1,0],[0,0,1]])
    #print invwishart_rv(nu,a)
    x = np.array([ invwishart_rv(nu,a) for i in range(20000)])
    nux = np.array([invwishart_prec_rv(nu,a) for i in range(20000)])
    print(x.shape)
    print(np.mean(x,0),"\n", inv(np.mean(nux,0)))
    #print inv(a)/(nu-a.shape[0]-1)