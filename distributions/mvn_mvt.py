from __future__ import division, print_function, absolute_import
import autograd.numpy as np
from autograd.numpy import log, exp
import autograd.numpy.random as npr
from autograd.numpy.linalg import inv, cholesky
from autograd.scipy.special import multigammaln, gammaln
from scipy.stats import chi2
import scipy.stats as stats
from .linalg import pdinv, diag_dot

import sys

__all__ = ["mvnorm", "mvdiagnorm", "mvt", "mvt_logpdf", "mvt_rvs", "mvt_ppf", 'mvnorm_logpdf', 'mvnorm_logpdf_theano', 'mvt_logpdf_theano']

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


def mvnorm_logpdf_theano(x, mu = None, Li = None):
    """
    Parameters
    ++++++++++
    mu - mean of MVN, if not given assume zero mean
    Li - inverse of lower cholesky
    """
    
    import theano.tensor as T
    dim = Li.shape[0]
    Ki = Li.T.dot(Li)
    #determinant is just multiplication of diagonal elements of cholesky
    logdet = 2*T.log(1./T.diag(Li)).sum()
    lpdf_const = -0.5 * T.flatten(dim * T.log(2 * np.pi) + logdet)[0]
    if mu is None:
        d = T.reshape(x, (dim, 1))
    else:
        d = (x - mu.reshape((1 ,dim))).T

    Ki_d = T.dot(Ki, d)        #vector
    
    res_pdf = (lpdf_const - 0.5 * diag_dot(d.T, Ki_d)).T
    if res_pdf.size == 1:
        res_pdf = T.float(res_pdf)
    return res_pdf 

def mvnorm_logpdf(x, mu = None, Li = None):
    """
    Parameters
    ++++++++++
    mu - mean of MVN, if not given assume zero mean
    Li - inverse of lower cholesky
    """
    
    import autograd.numpy as T
    dim = Li.shape[0]
    Ki = np.dot(Li.T, Li)
    #determinant is just multiplication of diagonal elements of cholesky
    logdet = 2*T.log(1./T.diag(Li)).sum()
    lpdf_const = -0.5 * (dim * T.log(2 * np.pi) + logdet)
    if mu is None:
        d = T.reshape(x, (dim, 1))
    else:
        d = (x - mu.reshape((1 ,dim))).T

    Ki_d = T.dot(Ki, d)        #vector
    
    res_pdf = (lpdf_const - 0.5 * diag_dot(d.T, Ki_d)).T
    if res_pdf.size == 1:
        res_pdf = res_pdf[0]
    return res_pdf 
    
def mvnorm_diag_logpdf_theano(x, mu = None, Li = None):
    """
    Parameters
    ++++++++++
    mu - mean of MVN, if not given assume zero mean
    Li - inverse of lower cholesky
    """
    
    import theano.tensor as T
    dim = Li.size
    Ki = Li**2
    #determinant is just multiplication of diagonal elements of cholesky
    logdet = 2*T.log(1./Li).sum()
    lpdf_const = -0.5 * T.flatten(dim * T.log(2 * np.pi) + logdet)[0]
    if mu is None:
        d = T.reshape(x, (dim, 1))
    else:
        d = (x - mu.reshape((1 ,dim))).T

    Ki_d = T.dot(Ki, d)        #vector
    
    res_pdf = (lpdf_const - 0.5 * diag_dot(d.T, Ki_d)).T
    if res_pdf.size == 1:
        res_pdf = T.float(res_pdf)
    return res_pdf     

class mvdiagnorm(object):
    def __init__(self, mu, var):
        self.norm_const = - 0.5*np.log(2*np.pi)
        self.mu = np.atleast_1d(mu).flatten()
        self.var = np.atleast_1d(var).flatten() 
        self.dim = np.prod(self.var.shape)
        assert(self.mu.shape == self.var.shape)
        self.std = np.sqrt(var)
        self.logstd = np.log(self.std)

    def get_num_unif(self):
        return self.dim
                                       
    def set_mu(self, mu):
        self.mu = np.atleast_1d(mu).flatten()
    
    def logpdf(self, x):
        xcent2d2 = (x-self.mu)**2/2
        return self.norm_const - self.logstd - xcent2d2/self.var
    
class mvnorm(object):
    def __init__(self, mu, K, Ki = None, logdet_K = None, L = None): 
        mu = np.atleast_1d(mu).flatten()
        K = np.atleast_2d(K) 
        assert(np.prod(mu.shape) == K.shape[0] )
        assert(K.shape[0] == K.shape[1])
        
        self.mu = mu
        self.K = K
        (val, vec) = np.linalg.eigh(K)
        idx = np.arange(mu.size-1,-1,-1)
        (self.eigval, self.eigvec) = (np.diag(val[idx]), vec[:,idx])
        self.eig = self.eigvec.dot(np.sqrt(self.eigval))
        self.dim = K.shape[0]
        #(self.Ki, self.logdet) = (np.linalg.inv(K), np.linalg.slogdet(K)[1])
        (self.Ki, self.L, self.Li, self.logdet) = pdinv(K)
        
        self.lpdf_const = -0.5 *np.float(self.dim * np.log(2 * np.pi)
                                           + self.logdet)
#    def get_theano_logp(self, X):
#        import theano.tensor as T
#        T.matrix("log")
#        d = x - np.atleast_2d(self.mu).T
#        return (self.lpdf_const - 0.5 *d.dot(Ki.dot(d))).T

    def get_num_unif(self):
        return self.dim
                                       
    def set_mu(self, mu):
        self.mu = np.atleast_1d(mu).flatten()
        
    def ppf(self, component_cum_prob, eig = False):
        assert(component_cum_prob.shape[1] == self.get_num_unif())
        #this is a pointwise ppf
        std_norm_dist = stats.norm(0, 1)
        rval = []
        for r in range(component_cum_prob.shape[0]):
            std_norm = std_norm_dist.ppf(component_cum_prob[r, :])
            if  eig:
                rval.append(self.mu + (self.eig.dot(std_norm)))
                #assert()
            else:
                rval.append(self.mu + self.L.dot(std_norm))
            
        return np.array(rval)
    
    def logpdf(self, x, theano_expr = False):
        if not theano_expr:
            return self.log_pdf_and_grad(x, pdf = True, grad = False)
        else:
            import theano.tensor as T
            return self.log_pdf_and_grad(x, pdf = True, grad = False, T=T)
    
    def logpdf_grad(self, x):
        return self.log_pdf_and_grad(x, pdf = False, grad = True)
    
    def log_pdf_and_grad(self, x, pdf = True, grad = True, T = np):
        assert(pdf or grad)
        
        if T == np:
            x = np.atleast_2d(x)
            if x.shape[1] != self.mu.size:
                x = x.T
            assert(np.sum(np.array(x.shape) == self.mu.size)>=1)
        
        d = (x - self.mu.reshape((1 ,self.mu.size))).T

        Ki_d = T.dot(self.Ki, d)        #vector
        
        if pdf:
            # vector times vector
            res_pdf = (self.lpdf_const - 0.5 * diag_dot(d.T, Ki_d)).T
            if res_pdf.size == 1:
                res_pdf = res_pdf.reshape(res_pdf.size)[0]
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
    
    def rvs(self, n=1, eig=False):
        rval = self.ppf(stats.uniform.rvs(size = (n, self.dim)), eig=eig)
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


def mvt_logpdf(x, mu, Li, df):
    dim = Li.shape[0]
    Ki = np.dot(Li.T, Li)

    #determinant is just multiplication of diagonal elements of cholesky
    logdet = 2*log(1./np.diag(Li)).sum()
    lpdf_const = (gammaln((df + dim) / 2)
                                   -(gammaln(df/2)
                                     + (log(df)+log(np.pi)) * dim*0.5
                                     + logdet * 0.5)
                                   )

    x = np.atleast_2d(x)
    if x.shape[1] != mu.size:
        x = x.T
    assert(x.shape[1] == mu.size
               or x.shape[0] == mu.size)
    
    d = (x - mu.reshape((1 ,mu.size))).T
    
    Ki_d_scal = np.dot(Ki, d) /df          #vector
    d_Ki_d_scal_1 = diag_dot(d.T, Ki_d_scal) + 1. #scalar
    

    res_pdf = (lpdf_const 
               - 0.5 * (df + dim) * np.log(d_Ki_d_scal_1)).flatten() 
    if res_pdf.size == 1:
        res_pdf = np.float(res_pdf)
    return res_pdf

def mvt_ppf(component_cum_prob, mu, L, df):
    from scipy.stats import norm, chi2
    mu = np.atleast_1d(mu).flatten()
    assert(component_cum_prob.shape[1] == mu.size+1)
    L = np.atleast_2d(L)
    rval = []
    for r in range(component_cum_prob.shape[0]):
        samp_mvn_0mu = L.dot(norm.ppf(component_cum_prob[r, :-1]))
        samp_chi2 = chi2.ppf(component_cum_prob[r, -1], df)
        samp_mvt_0mu = samp_mvn_0mu * np.sqrt(df / samp_chi2)
        rval.append(mu + samp_mvt_0mu)
    return np.array(rval)

def mvt_rvs(n, mu, L, df):
    from scipy.stats import uniform
    mu = np.atleast_1d(mu).flatten()
    return mvt_ppf(uniform.rvs(size = (n, mu.size+1)), mu, L, df)

def mvt_logpdf_theano(x, mu, Li, df):
    import theano.tensor as T
    dim = Li.shape[0]
    Ki = Li.T.dot(Li)
    #determinant is just multiplication of diagonal elements of cholesky
    logdet = 2*T.log(1./T.diag(Li)).sum()
    lpdf_const = (T.gammaln((df + dim) / 2)
                       -(T.gammaln(df/2)
                         + (T.log(df)+T.log(np.pi)) * dim*0.5
                         + logdet * 0.5)
                       )

    d = (x - mu.reshape((1 ,mu.size))).T
    Ki_d_scal = T.dot(Ki, d) / df          #vector
    d_Ki_d_scal_1 = diag_dot(d.T, Ki_d_scal) + 1. #scalar
    
    res_pdf = (lpdf_const 
               - 0.5 * (df+dim) * T.log(d_Ki_d_scal_1)).flatten() 
    if res_pdf.size == 1:
        res_pdf = T.float(res_pdf)
    return res_pdf 

class mvt(object):
    def __init__(self, mu, K, df, Ki = None, logdet_K = None, L = None):
        mu = np.atleast_1d(mu).flatten()
        K = np.atleast_2d(K)
        assert(np.prod(mu.shape) == K.shape[0] )
        assert(K.shape[0] == K.shape[1])
        self.mu = mu
        self.K = K
        self.df = df
        self._freeze_chi2 = stats.chi2(df)
        self.dim = K.shape[0]
        self._df_dim = self.df + self.dim
        #(self.Ki,  self.logdet) = (np.linalg.inv(K), np.linalg.slogdet(K)[1])
        (self.Ki, self.L, self.Li, self.logdet) = pdinv(K)
        
        
        self.lpdf_const = np.float(gammaln((self.df + self.dim) / 2)
                                   -(gammaln(self.df/2)
                                     + (log(self.df)+log(np.pi)) * self.dim*0.5
                                     + self.logdet * 0.5)
                                   )
    def get_num_unif(self):
        return self.dim + 1
        
    def set_mu(self, mu):
        self.mu = np.atleast_1d(mu).flatten()
        
    def set_df(self, df):
        self.df = df
        self._freeze_chi2 = stats.chi2(df)
        
    def ppf(self, component_cum_prob):
        #this is a pointwise ppf
        assert(component_cum_prob.shape[1] == self.get_num_unif())
        std_norm = stats.norm(0, 1)
        rval = []
        for r in range(component_cum_prob.shape[0]):
            samp_mvn_0mu = self.L.dot(std_norm.ppf(component_cum_prob[r, :-1]))
            samp_chi2 = self._freeze_chi2.ppf(component_cum_prob[r, -1])
            samp_mvt_0mu = samp_mvn_0mu * np.sqrt(self.df / samp_chi2)
            rval.append(self.mu + samp_mvt_0mu)
        return np.array(rval)
    
    def logpdf(self, x, theano_expr = False):
        if not theano_expr:
            return self.log_pdf_and_grad(x, pdf = True, grad = False)
        else:
            import theano.tensor as T
            return self.log_pdf_and_grad(x, pdf = True, grad = False, T=T)
    
    def logpdf_grad(self, x):
        return self.log_pdf_and_grad(x, pdf = False, grad = True)
    
    def log_pdf_and_grad(self, x, pdf = True, grad = True, T = np):
        assert(pdf or grad)
        
        if T == np:
            x = T.atleast_2d(x)
            if x.shape[1] != self.mu.size:
                x = x.T
            assert(x.shape[1] == self.mu.size
                       or x.shape[0] == self.mu.size)
        
        d = (x - self.mu.reshape((1 ,self.mu.size))).T
        Ki_d_scal = T.dot(self.Ki, d) / self.df          #vector
        d_Ki_d_scal_1 = diag_dot(d.T, Ki_d_scal) + 1. #scalar
        
        if pdf:
            # purely scalar multiplication
            res_pdf = (self.lpdf_const 
                       - 0.5 * self._df_dim * T.log(d_Ki_d_scal_1)).flatten() 
            if res_pdf.size == 1:
                res_pdf = res_pdf.flat[0]
            if not grad:
                return res_pdf
        if grad:
            #scalar times vector
            res_grad = -self._df_dim/T.atleast_2d(d_Ki_d_scal_1).T * Ki_d_scal.T
            
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
