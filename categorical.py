from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.stats import chi2
import scipy.stats as stats
from linalg import pdinv


class categorical(object):
    def __init__(self, p):
        assert(np.abs(1-np.sum(p)) < 10**-7)
        self.p = np.array(p).flat[:]
        self.cum_p = np.array([self.p[:i].sum() for i in range(1, len(self.p)+1)])
        
    def ppf(self, cum_prob, indic = False):
        idc = (self.cum_p >= cum_prob)
        if indic:
            idc[np.argmax(idc)+1:] = False
            return idc
        else:
            return np.argmax(idc)