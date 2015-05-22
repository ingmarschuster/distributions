from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.stats import chi2
import scipy.stats as stats
from .linalg import pdinv, diag_dot

import sys

__all__ = ["mvnorm", "mvt"]

def shape_match_1d(y, x):
    """
    match the shape of y to that of x and return the new y object.
    """
    #assert(len(y.shape) == 1)
    sh = (np.array(x.shape) == np.array(y.shape))
    if sh.sum() != 1:
        print("There have been " + str(sh.sum()) + " matches in shape instead of exactly 1. Using first match only.", file=sys.stderr)
        sh[np.argmax(sh)+1:] = False
    
    new_sh = np.ones(sh.shape)
    new_sh[sh] = y.shape
    return np.reshape(y, new_sh)

class mvnorm(object):
    def __init__(self, mu, K, Ki = None, logdet_K = None, L = None): 
        mu = np.atleast_1d(mu).flatten()
        K = np.atleast_2d(K) 
        assert(np.prod(mu.shape) == K.shape[0] )
        assert(K.shape[0] == K.shape[1])
        
        self.mu = mu
        self.K = K
        self.dim = K.shape[0]        
        (self.Ki, self.L, self.Li, self.logdet) = pdinv(K)
        
        self.lpdf_const = -0.5 *np.float(self.dim * np.log(2 * np.pi)
                                           + self.logdet)
                                           
    def set_mu(self, mu):
        self.mu = np.atleast_1d(mu).flatten()
        
    def ppf(self, component_cum_prob):
        assert(component_cum_prob.shape[1] == self.dim)
        #this is a pointwise ppf
        std_norm = stats.norm(0, 1)
        rval = []
        for r in range(component_cum_prob.shape[0]):
            rval.append(self.mu + self.L.dot(std_norm.ppf(component_cum_prob[r, :])))
        return np.array(rval)
    
    def logpdf(self, x):
        return self.log_pdf_and_grad(x, pdf = True, grad = False)
    
    def logpdf_grad(self, x):
        return self.log_pdf_and_grad(x, pdf = False, grad = True)
    
    def log_pdf_and_grad(self, x, pdf = True, grad = True):
        assert(pdf or grad)
        
        x = np.atleast_2d(x)
        if x.shape[1] == self.mu.size:
            x = x.T
        else:
            assert(np.sum(np.array(x.shape) == self.mu.size)>=1)
        
        d = x - np.atleast_2d(self.mu).T

        Ki_d = self.Ki.dot(d)        #vector
        
        if pdf:
            # vector times vector
            res_pdf = (self.lpdf_const - 0.5 * diag_dot(d.T, Ki_d)).T
            if res_pdf.size == 1:
                res_pdf = np.float(res_pdf)
            if not grad:
                return res_pdf
        if grad:
            # nothing
            res_grad = - Ki_d.T #.flat[:]
            if res_grad.shape[0] <= 1:
                res_grad = res_grad.flatten()
            if not pdf:
                return res_grad
        return (res_pdf, res_grad)    
    
    def rvs(self, n=1):
        rval = self.ppf(stats.uniform.rvs(size = (n, self.dim)))
        if n == 1:
            return rval.flatten()
        else:
            return rval
    
    @classmethod
    def fit(cls, samples, return_instance = False): # observations expected in rows
        mu = samples.mean(0)
        var = np.atleast_2d(np.cov(samples, rowvar = 0))
        if return_instance:
            return mvnorm(mu, var)
        else:
            return (mu, var)

class mvt(object):
    def __init__(self, mu, K, df, Ki = None, logdet_K = None, L = None):
        mu = np.atleast_1d(mu).flatten()
        K = np.atleast_2d(K)
        assert(np.prod(mu.shape) == K.shape[0] )
        assert(K.shape[0] == K.shape[1])
        self.mu = mu
        self.K = K
        self.df = df
        self.dim = K.shape[0]
        self._df_dim = self.df + self.dim
        (self.Ki, self.L, self.Li, self.logdet) = pdinv(K)
        
        self._freeze_chi2 = stats.chi2(self.df)
        self.lpdf_const = np.float(gammaln((self.df + self.dim) / 2)
                                   -(gammaln(self.df/2)
                                     + (log(self.df)+log(np.pi)) * self.dim*0.5
                                     + self.logdet * 0.5)
                                   )
    
    def set_mu(self, mu):
        self.mu = np.atleast_1d(mu).flatten()
        
    def ppf(self, component_cum_prob):
        #this is a pointwise ppf
        assert(component_cum_prob.shape[1] == self.dim + 1)
        std_norm = stats.norm(0, 1)
        rval = []
        for r in range(component_cum_prob.shape[0]):
            samp_mvn_0mu = self.L.dot(std_norm.ppf(component_cum_prob[r, :-1]))
            samp_chi2 = self._freeze_chi2.ppf(component_cum_prob[r, -1])
            samp_mvt_0mu = samp_mvn_0mu * np.sqrt(self.df / samp_chi2)
            rval.append(self.mu + samp_mvt_0mu)
        return np.array(rval)
    
    def logpdf(self, x):
        return self.log_pdf_and_grad(x, pdf = True, grad = False)
    
    def logpdf_grad(self, x):
        return self.log_pdf_and_grad(x, pdf = False, grad = True)
    
    def log_pdf_and_grad(self, x, pdf = True, grad = True):
        assert(pdf or grad)
        
        x = np.atleast_2d(x)
        if x.shape[1] == self.mu.size:
            x = x.T
        else:
            assert(x.shape[1] == self.mu.size
                   or x.shape[0] == self.mu.size)
        
        d = x - np.atleast_2d(self.mu).T
        Ki_d_scal = self.Ki.dot(d) / self.df          #vector
        d_Ki_d_scal_1 = diag_dot(d.T, Ki_d_scal) + 1. #scalar
        
        if pdf:
            # purely scalar multiplication
            res_pdf = (self.lpdf_const 
                       - 0.5 * self._df_dim * log(d_Ki_d_scal_1)).flatten() 
            if res_pdf.size == 1:
                res_pdf = np.float(res_pdf)
            if not grad:
                return res_pdf
        if grad:
            #scalar times vector
            res_grad = -self._df_dim/np.atleast_2d(d_Ki_d_scal_1).T * Ki_d_scal.T
            
            if res_grad.shape[0] <= 1:
                res_grad = res_grad.flatten()
            if not pdf:
                return res_grad
        return (res_pdf, res_grad)
    
    def rvs(self, n = 1):
        rval =  self.ppf(stats.uniform.rvs(size = (n, self.dim+1)))
        if n == 1:
            return rval.flatten()
        else:
            return rval
    
    def __getstate__(self):
        from copy import copy
        rval = copy(self.__dict__)
        rval.pop("_freeze_chi2")
        return rval
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._freeze_chi2 = stats.chi2(self.df)
    
    @classmethod
    def fit(cls, samples, return_instance = False): # observations expected in rows
        raise(NotImplementedError())
        mu = samples.mean(0)
        return (mu, np.atleast_2d(np.cov(samples, rowvar = 0)))
