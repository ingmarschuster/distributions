from __future__ import division, print_function, absolute_import
import autograd.numpy as np
from  autograd.numpy import log, exp
import numpy.random as npr
from  autograd.numpy.linalg import inv, cholesky
from  autograd.scipy.special import multigammaln, gammaln
from  autograd.scipy.misc import logsumexp
from  autograd.scipy import stats as stats
from .linalg import pdinv, diag_dot
from .cat_dirichlet import categorical
from .mvn_mvt import mvnorm, mvt

import sys

__all__ = ["mixt", "NPGMM", "GMM", "TMM"]

class mixt(object):
    def __init__(self, dim, comp_dists, comp_weights, comp_w_in_logspace = False):
        self.dim = dim
        self.comp_dist = comp_dists
        self.dist_cat = categorical(comp_weights, comp_w_in_logspace)

    def get_num_unif(self):
        return self.dim + 1
    
    def ppf(self, component_cum_prob, eig=True):
        assert(component_cum_prob.shape[1] == self.get_num_unif())
        rval = []
        for i in range(component_cum_prob.shape[0]):
            r = component_cum_prob[i,:]
            comp = self.dist_cat.ppf(r[0])
            rval.append(self.comp_dist[comp].ppf(np.atleast_2d(r[1:]), eig=True))
        return np.array(rval).reshape((component_cum_prob.shape[0], self.dim))
    
    def logpdf(self, x):
        comp_logpdf = np.array([self.dist_cat.logpdf(i)+ self.comp_dist[i].logpdf(x)
                              for i in range(len(self.comp_dist))])
        rval = logsumexp(comp_logpdf, 0)
        if len(comp_logpdf.shape) > 1:
            rval = rval.reshape((rval.size, 1))
        return rval
     
    def logpdf_grad(self, x):
        rval = np.array([exp(self.dist_cat.logpdf(i))* self.comp_dist[i].logpdf_grad(x)
                              for i in range(len(self.comp_dist))])

        rval = logsumexp(rval, 0)
        
        return rval
    
    def rvs(self, num_samples=1):
        rval = np.array([self.comp_dist[i].rvs(1) for i in self.dist_cat.rvs(num_samples)])
        if num_samples == 1 and len(rval.shape) > 1:
            return rval[0]
        else:
            return rval

class SmoothMixt(object):
    def __init__(self, dim, comp_dists, weightfunc):
        self.dim = dim
        self.comp_dist = comp_dists
        self.weightfunc = weightfunc

    def get_num_unif(self):
        return self.dim
    
    def ppf(self, component_cum_prob, eig=True):
        assert(component_cum_prob.shape[1] == self.get_num_unif())
        rval = []
        for i in range(component_cum_prob.shape[0]):
            r = component_cum_prob[i,:]
            comp = self.dist_cat.ppf(r[0])
            rval.append(self.comp_dist[comp].ppf(np.atleast_2d(r[1:]), eig=True))
        return np.array(rval).reshape((component_cum_prob.shape[0], self.dim))
    
    def logpdf(self, x):
        comp_logpdf = np.array([self.dist_cat.logpdf(i)+ self.comp_dist[i].logpdf(x)
                              for i in range(len(self.comp_dist))])
        rval = logsumexp(comp_logpdf, 0)
        if len(comp_logpdf.shape) > 1:
            rval = rval.reshape((rval.size, 1))
        return rval
     
    def logpdf_grad(self, x):
        rval = np.array([exp(self.dist_cat.logpdf(i))* self.comp_dist[i].logpdf_grad(x)
                              for i in range(len(self.comp_dist))])

        rval = logsumexp(rval, 0)
        
        return rval
    
    def rvs(self, num_samples=1):
        rval = np.array([self.comp_dist[i].rvs(1) for i in self.dist_cat.rvs(num_samples)])
        if num_samples == 1 and len(rval.shape) > 1:
            return rval[0]
        else:
            return rval

class NPGMM(object):
    def __init__(self, dim, samples = None):
        self.dim = dim
        if samples is not None:
            self.fit(samples)

    
    def fit(self, samples, scale = 1):
        import sklearn.mixture
        m = sklearn.mixture.DPGMM(covariance_type="full")
        m.fit(samples)
        self.num_components = len(m.weights_)
        self.comp_lprior = log(m.weights_)
        self.dist_cat = categorical(exp(self.comp_lprior))
        self.comp_dist = [mvnorm(m.means_[i], np.linalg.inv(m.precs_[i])* scale) for i in range(self.comp_lprior.size)]
        self.dim = m.means_[0].size
        
    def get_num_unif(self):
        return self.dim + 1
        
    def ppf(self, component_cum_prob, eig=True):
        assert(component_cum_prob.shape[1] == self.get_num_unif())
        rval = []
        for i in range(component_cum_prob.shape[0]):
            r = component_cum_prob[i,:]
            comp = self.dist_cat.ppf(r[0])
            rval.append(self.comp_dist[comp].ppf(np.atleast_2d(r[1:]), eig=True))
        return np.array(rval).reshape((component_cum_prob.shape[0], self.dim))
    
    def logpdf(self, x):
        rval = np.array([self.comp_lprior[i]+ self.comp_dist[i].logpdf(x)
                              for i in range(self.comp_lprior.size)])
        rval = logsumexp(rval, 0).flatten()
        return rval
    
    def rvs(self, num_samples=1):
        rval = self.ppf(stats.uniform.rvs(0, 1, (num_samples, self.dim + 1)))
        if num_samples == 1 and len(rval.shape) > 1:
            return rval[0]
    

class GMM(object):
    def __init__(self, num_components, dim, samples = None):
        self.num_components = num_components
        self.dim = dim
        if samples is not None:
            self.fit(samples)
            
    def get_num_unif(self):
        return self.dim + 1
    
    def fit(self, samples, scale = 1):
        import sklearn.mixture
        m = sklearn.mixture.GMM(self.num_components, "full")
        m.fit(samples)
        self.comp_lprior = log(m.weights_)
        self.dist_cat = categorical(exp(self.comp_lprior))
        self.comp_dist = [mvnorm(m.means_[i], m.covars_[i] * scale) for i in range(self.comp_lprior.size)]
        self.dim = m.means_[0].size
        #self._e_step()
        if False:        
            old = -1
            i = 0
            while not np.all(old == self.resp):
                i += 1
                old = self.resp.copy()
                self._e_step()
                self._m_step()
                print(np.sum(old == self.resp)/self.resp.size)
            #print("Convergence after",i,"iterations")
            self.dist_cat = categorical(exp(self.comp_lprior))
    
    def _m_step(self):
        assert(self.resp.shape[0] == self.num_samp)
        pseud_lcount = logsumexp(self.resp, axis = 0).flat
        r = exp(self.resp)        
        
        self.comp_dist = []
        for c in range(self.num_components):
            norm = exp(pseud_lcount[c])
            mu = np.sum(r[:,c:c+1] * self.samples, axis=0) / norm
            diff = self.samples - mu
            scatter_matrix = np.zeros([self.samples.shape[1]]*2)
            for i in range(diff.shape[0]):
                scatter_matrix += r[i,c:c+1] *diff[i:i+1,:].T.dot(diff[i:i+1,:])
            scatter_matrix /= norm
            self.comp_dist.append(mvnorm(mu, scatter_matrix))
        self.comp_lprior = pseud_lcount - log(self.num_samp)
            
    
    def _e_step(self):
        lpdfs = np.array([d.logpdf(self.samples).flat[:] 
                              for d in self.comp_dist]).T + self.comp_lprior
        self.resp = lpdfs - logsumexp(lpdfs, axis = 1).reshape((self.num_samp, 1))
    
    def ppf(self, component_cum_prob, eig=True):
        assert(component_cum_prob.shape[1] == self.get_num_unif())
        rval = []
        for i in range(component_cum_prob.shape[0]):
            r = component_cum_prob[i,:]
            comp = self.dist_cat.ppf(r[0])
            rval.append(self.comp_dist[comp].ppf(np.atleast_2d(r[1:]), eig=True))
        return np.array(rval).reshape((component_cum_prob.shape[0], self.dim))
    
    def logpdf(self, x):
        rval = np.array([self.comp_lprior[i]+ self.comp_dist[i].logpdf(x)
                              for i in range(self.comp_lprior.size)])
        rval = logsumexp(rval, 0).flatten()
        return rval
    
    def rvs(self, num_samples=1):
        rval = self.ppf(stats.uniform.rvs(0, 1, (num_samples, self.dim + 1)))
        if num_samples == 1 and len(rval.shape) > 1:
            return rval[0]


class TMM(object):
    def __init__(self, num_components, dim, df, samples = None):
        self.num_components = num_components
        self.dim = dim
        self.df = df
        if samples is not None:
            self.fit(samples)
            
    def get_num_unif(self):
        return self.dim + 2
    
    def fit(self, samples):
        import sklearn.mixture
        m = sklearn.mixture.GMM(self.num_components, "full")
        m.fit(samples)
        self.comp_lprior = log(m.weights_)
        self.dist_cat = categorical(exp(self.comp_lprior))
        self.comp_dist = [mvt(m.means_[i], m.covars_[i], self.df) for i in range(self.comp_lprior.size)]
        self.dim = m.means_[0].size
        #self._e_step()
        if False:        
            old = -1
            i = 0
            while not np.all(old == self.resp):
                i += 1
                old = self.resp.copy()
                self._e_step()
                self._m_step()
                print(np.sum(old == self.resp)/self.resp.size)
            #print("Convergence after",i,"iterations")
            self.dist_cat = categorical(exp(self.comp_lprior))
       
    def ppf(self, component_cum_prob, eig=True):
        assert(component_cum_prob.shape[1] == self.get_num_unif())
        rval = []
        for i in range(component_cum_prob.shape[0]):
            r = component_cum_prob[i,:]
            comp = self.dist_cat.ppf(r[0])
            rval.append(self.comp_dist[comp].ppf(np.atleast_2d(r[1:])))
        return np.array(rval).reshape((component_cum_prob.shape[0], self.dim))
    
    def logpdf(self, x):
        rval = np.array([self.comp_lprior[i]+ self.comp_dist[i].logpdf(x)
                              for i in range(self.comp_lprior.size)])
        rval = logsumexp(rval, 0).flatten()
        return rval
    
    def rvs(self, num_samples=1):
        rval = self.ppf(stats.uniform.rvs(0, 1, (num_samples, self.get_num_unif())))
        if num_samples == 1 and len(rval.shape) > 1:
            return rval[0]
