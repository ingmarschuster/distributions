from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.misc import logsumexp
from scipy.stats import chi2
import scipy.stats as stats


class categorical(object):
    def __init__(self, p, p_in_logspace = False):
        if not p_in_logspace:
            self.lp = log(p).flatten()
        else:
            self.lp = np.array(p).flatten()
        assert(np.abs(1-exp(logsumexp(self.lp))) < 10**-7)
        self.cum_lp = np.array([logsumexp(self.lp[:i]) for i in range(1, len(self.lp)+1)])
        
    def ppf(self, x, indic = False, x_in_logspace = False):
        if not x_in_logspace:
            x = log(x)
        idc = (self.cum_lp >= x)
        if indic:
            idc[np.argmax(idc)+1:] = False
            return idc
        else:
            return np.argmax(idc)
            
    def rvs(self, indic = False):
        return self.ppf(stats.uniform.rvs(), indic = indic)

def test_categorical():
    dist = categorical(np.array((0.5, 0.3,0.1,0.1)))
    #assert(dist.ppf(0.4) == 0 and dist.ppf(0.6) == 1 and dist.ppf(0.95) == 3)
    for (p, idx) in [(0.4, 0), (0.5, 0), (0.6, 1), (0.95, 3)]:
        assert(dist.ppf(p) == idx)
        assert(np.argmax(dist.ppf(p, indic = True)) == idx)
    print(dist.rvs())