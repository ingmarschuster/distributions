from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.stats import chi2
import scipy.stats as stats
from .linalg import pdinv, diag_dot


class mvnorm(object):
    def __init__(self, mu, K, Ki = None, logdet_K = None, L = None): 
        mu = np.atleast_1d(mu).flatten()
        K = np.atleast_2d(K) 
        assert(np.prod(mu.shape) == K.shape[0] )
        assert(K.shape[0] == K.shape[1])
        
        self.mu = mu
        self.K = K
        self.dim = K.shape[0]        
        #(self.Ki, self.L, self.Li, self.logdet) = pdinv(K)
        if Ki is None:
            self.Ki = np.linalg.inv(K)
        else:
            self.Ki = Ki
        if logdet_K is None:
            self.logdet = np.linalg.slogdet(K)[1]
        else:
            self.logdet = logdet_K
        if L is None:
            self.L = np.linalg.cholesky(K)
        else:
            self.L = L
        
        self.lpdf_const = -0.5 *np.float(self.dim * np.log(2 * np.pi)
                                           + self.logdet)
        
    def ppf(self, component_cum_prob):
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
            assert(x.shape[1] == self.mu.size
                   or x.shape[0] == self.mu.size)
        
        d = x - np.atleast_2d(self.mu).T
        Ki_d = self.Ki.dot(d)        
        
        if pdf:
            res_pdf = (self.lpdf_const - 0.5 * diag_dot(d.T, Ki_d)).T
            if res_pdf.size == 1:
                res_pdf = np.float(res_pdf)
            if not grad:
                return res_pdf
        if grad:
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
        #(self.Ki, self.L, self.Li, self.logdet) = pdinv(K)
        if Ki is None:
            self.Ki = np.linalg.inv(K)
        else:
            self.Ki = Ki
        if logdet_K is None:
            self.logdet = np.linalg.slogdet(K)[1]
        else:
            self.logdet = logdet_K
        if L is None:
            self.L = np.linalg.cholesky(K)
        else:
            self.L = L
        
        self.freeze_mvn = stats.multivariate_normal(mu, K)
        self.freeze_chi2 = stats.chi2(self.df)
        self.lpdf_const = np.float(gammaln((self.df + self.dim) / 2)
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
        Ki_d_scal = self.Ki.dot(d) / self.df
        d_Ki_d_scal_1 = diag_dot(d.T, Ki_d_scal) + 1.
        
        if pdf:
            res_pdf = (self.lpdf_const 
                       - 0.5 * self._df_dim * log(d_Ki_d_scal_1)).flatten()
            if res_pdf.size == 1:
                res_pdf = np.float(res_pdf)
            if not grad:
                return res_pdf
        if grad:
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
        for dist in [mvt(mu,var,df), mvnorm(mu,var)]:
            ad = np.abs(dist.logpdf(mu) -lpdf )   
            assert(ad < 10**-8)
            assert(np.all(opt.check_grad(dist.logpdf, dist.logpdf_grad, mu-1) < 10**-7))
    
            al = [(5,4), (3,3), (1,1)]
            (cpdf, cgrad) = dist.log_pdf_and_grad(al)
            (spdf, sgrad) = zip(*[dist.log_pdf_and_grad(m) for m in al])
            (spdf, sgrad) = (np.array(spdf), np.array(sgrad)) 
            assert(np.all(cpdf == spdf) and np.all(cpdf == spdf))
            assert(sgrad.shape == cgrad.shape)
            
    mu = np.array([ 11.56966913,   8.66926112])
    obs = np.array([[ 1.31227875, -2.88454287],[ 2.14283061, -2.97526902]])
    var = np.array([[ 1.44954579, -1.43116137], [-1.43116137,  3.6207941 ]])
    dist = mvnorm(mu, var)
    
    assert(np.all(dist.logpdf(obs) - stats.multivariate_normal(mu, var).logpdf(obs) < 10**-7))
