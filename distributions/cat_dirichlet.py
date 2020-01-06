from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.special import logsumexp
from scipy.stats import chi2
import scipy.stats as stats

from scipy.special import gammaln

__all__ = ["categorical", "dirichlet"]

class categorical(object):
    def __init__(self, p, p_in_logspace = False):
        if not p_in_logspace:
            self.lp = log(p).flatten()
        else:
            self.lp = np.array(p).flatten()
        if np.abs(1-exp(logsumexp(self.lp))) >= 10**-7:
            raise ValueError("the probability vector does not sum to 1")
        self.cum_lp = np.array([logsumexp(self.lp[:i]) for i in range(1, len(self.lp)+1)])
    
    def get_num_unif(self):
        return 1    
        
    def ppf(self, x, indic = False, x_in_logspace = False):
        if not x_in_logspace:
            x = log(x)
        idc = (self.cum_lp >= x)
        if indic:
            idc[np.argmax(idc)+1:] = False
            return idc
        else:
            return np.argmax(idc)
    
    def logpdf(self, x, indic = False):
        if indic:
            x = np.argmax(x, 1)
        return self.lp[x]
            
    def rvs(self, size = 1, indic = False):
        assert(size >= 0)
        return np.array([self.ppf(stats.uniform.rvs(), indic = indic)
                             for _ in range(size)])


class dirichlet(object):
    def __init__(self, p):
        p = np.array(p).flatten()
        assert(np.all(p > 0))
        self.p = p
        self.normalizing_const = gammaln(self.p).sum() - gammaln(self.p.sum())
    
    def logpdf(self, x):
        x = np.array(x).flatten()
        assert(x.size == self.p.size)
        return np.sum(log(x)*(self.p-1)) - self.normalizing_const
            
    def rvs(self, size = 1, indic = False):
        return np.random.dirichlet(self.p, size = size)