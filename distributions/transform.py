# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:24:40 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv


__all__ = ["softplus"]

class softplus(object):
    def __init__(self, wrapped_distribution, transform_components_at_index):        
        assert(len(transform_components_at_index) > 0)
        self.wrapped = wrapped_distribution
        self.idx = transform_components_at_index
        
    def rvs(self, n = 1):
        rval = self.wrapped.rvs(n)
        rval[:,self.idx] = log(1+exp(rval[:,self.idx]))
        return rval
    
    def logpdf(self, rvs_transf):
        rvs_orig = np.copy(rvs_transf)
        rvs_orig[:,self.idx] = log(exp(rvs_transf[:,self.idx]) - 1)
        rval = self.wrapped.logpdf(rvs_orig)
        correction = np.sum(-log(exp(rvs_transf[:,self.idx]) - 1) + rvs_transf[:,self.idx], 1)
        assert(len(correction) == len(rval))
        if len(correction.shape) < len(rval.shape):
            correction = correction[:, np.newaxis]
        return rval + correction

