from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.stats import chi2
import scipy.stats as stats
from linalg import pdinv


class mvnorm(object):
    def __init__(self, mu, K): 
        mu = np.atleast_1d(mu)
        K = np.atleast_2d(K)       
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
        sh = (np.prod(self.mu.shape), 1)
        return -self.Ki.dot(x.reshape(sh) - self.mu.reshape(sh)).flat[:]
    
    def rvs(self, *args, **kwargs):
        return self.freeze.rvs(*args, **kwargs)
    
    @classmethod
    def fit(cls, samples, return_instance = False): # observations expected in rows
        mu = samples.mean(0)
        var = np.atleast_2d(np.cov(samples, rowvar = 0))
        if return_instance:
            return mvnorm(mu, var)
        else:
            return (mu, var)
        

class mvt(object):
    def __init__(self, mu, K, df):
        mu = np.atleast_1d(mu)
        K = np.atleast_2d(K)
        assert(np.prod(mu.shape) == K.shape[0] )
        assert(K.shape[0] == K.shape[1])
        self.mu = mu
        self.K = K
        self.df = df
        self.dim = K.shape[0]
        (self.Ki, self.L, self.Li, self.logdet) = pdinv(K)
        self.freeze_mvn = stats.multivariate_normal(mu, K)
        self.freeze_chi2 = stats.chi2(self.df)
        self.lpdf_const = (gammaln((self.df + self.dim) / 2)
                           -(gammaln(self.df/2)
                             + (log(self.df)+log(np.pi)) * self.dim*0.5
                             + self.logdet * 0.5)
                           )
        
    def ppf(self, component_cum_prob):
        #this is a pointwise ppf
        std_norm = stats.norm(0, 1)
        rval = []
        for r in range(component_cum_prob.shape[0]):
            samp_mvn_0mu = self.L.dot(std_norm.ppf(component_cum_prob[r, :-1]))
            samp_chi2 = self.freeze_chi2.ppf(component_cum_prob[r, -1])
            samp_mvt_0mu = samp_mvn_0mu * np.sqrt(self.df / samp_chi2)
            rval.append(self.mu + samp_mvt_0mu)
        return np.array(rval)
    
    def logpdf(self, x):
        diff = (x - self.mu).reshape((np.prod(self.mu.shape), 1))
        c = diff.T.dot(self.Ki).dot(diff)
        return self.lpdf_const - 0.5*(self.df+self.dim)*log(1+c/self.df)
    
    def logpdf_grad(self, x):
        diff = (x - self.mu).reshape((np.prod(self.mu.shape), 1))
        var_diff = self.Ki.dot(diff)/self.df
        return -(self.df + self.dim)/(1+diff.T.dot(var_diff)) * var_diff.flat[:]
    
    def rvs(self, n = 1):
        return self.ppf(stats.uniform.rvs(size = (n, self.dim+1)))
    
    @classmethod
    def fit(cls, samples, return_instance = False): # observations expected in rows
        raise(RuntimeError())
        mu = samples.mean(0)
        return (mu, np.atleast_2d(np.cov(samples, rowvar = 0)))


###############


def test_mvt_mvn_logpdf_n_grad():
    import scipy.optimize as opt
    # values from R-package bayesm, function dmvt(6.1, a, a)
    for (mu, var, df, lpdf) in [(np.array((1,1)), np.eye(2),   3, -1.83787707) ,
                                (np.array((1,2)), np.eye(2)*3, 3, -2.93648936)]:
        for dist in [ mvnorm(mu,var), mvt(mu,var,df)]:
            ad = np.abs(dist.logpdf(mu) -lpdf )   
            assert(ad < 10**-8)
            assert(np.all(opt.check_grad(lambda x: dist.logpdf(x), lambda x: dist.logpdf_grad(x), mu-1) < 10**-17))
