from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.misc import logsumexp
import scipy.stats as stats
from .linalg import pdinv, diag_dot
from .cat_dirichlet import categorical

import sys

__all__ = ["mixt"]

class mixt(object):
    def __init__(self, dim, comp_dists, comp_weights, comp_w_in_logspace = False):
        self.dim = dim
        self.comp_dist = comp_dists
        self.cat_dist = categorical(comp_weights, comp_w_in_logspace)

    
    def ppf(self, component_cum_prob):
        assert(component_cum_prob.shape[1] == self.dim + 1)
        rval = []
        for i in range(component_cum_prob.shape[0]):
            r = component_cum_prob[i,:]
            comp = self.dist_cat.ppf(r[0])
            rval.append(self.comp_dist[comp].ppf(np.atleast_2d(r[1:])))
        return np.array(rval).reshape((component_cum_prob.shape[0], self.dim))
    
    def logpdf(self, x):
        rval = np.array([self.cat_dist.logpdf(i)+ self.comp_dist[i].logpdf(x)
                              for i in range(len(self.comp_dist))])
        rval = logsumexp(rval, 0).flatten()
        return rval
    
    def rvs(self, num_samples):
        rval = np.array([self.comp_dist[i].rvs(1) for i in self.cat_dist.rvs(num_samples)])
        return rval