from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.misc import logsumexp
from scipy.stats import chi2
import scipy.stats as stats

from scipy.special import gammaln

__all__ = ["logcumsumexp", "stream_categorical"]

def logcumsumexp(x):
    #FIXME: this could be more efficient maybe
    x = np.array(x)
    assert(len(x.shape) <= 1)
    assert(not np.any(np.isinf(x)))
    assert(not np.any(np.isnan(x)))
    lcs = np.zeros_like(x)
    lcs[0] = x[0]
    for i in range(1, x.size):
        lcs[i] = np.logaddexp(lcs[i-1], x[i])
    return lcs

class stream_categorical(object):
    def __init__(self, unnorm_initial_logprobs, planned_size):
        try:
            unnorm_initial_logprobs = unnorm_initial_logprobs.flatten()
        except:
            pass
        
        assert(len(unnorm_initial_logprobs) > 0)
        assert(planned_size >= len(unnorm_initial_logprobs))
        
        self.cum_lp = np.zeros(planned_size)
        self.cur_size = len(unnorm_initial_logprobs)
        self.cum_lp[:len(unnorm_initial_logprobs)] = logcumsumexp(unnorm_initial_logprobs)
        
    def get_num_unif(self):
        return 1
        
    def ppf(self, x, indic = False, x_in_logspace = False):
        if not x_in_logspace:
            x = log(x)
        x = x + self.cum_lp[self.cur_size-1]
        
        # binary search, very efficient
        idx = np.searchsorted(self.cum_lp[:self.cur_size], x) 
        if not indic:
            return idx
        else:
            rval = np.zeros(self.cur_size, dtype=bool)
            rval[idx] = True
            return rval
    
    def extend(self, unnorm_logpr):
        try:
            unnorm_logpr = unnorm_logpr.flatten()
        except:
            pass
        assert(len(self.cum_lp) >= self.cur_size + len(unnorm_logpr))
        
        #FIXME: this could be more efficient maybe
        for i in unnorm_logpr:
            self.cum_lp[self.cur_size] = np.logaddexp(self.cum_lp[self.cur_size-1], i)
            self.cur_size = self.cur_size + 1


    def logpdf(self, x, indic = False):
        if indic:
            x = np.argmax(x, 1)
        return self.lp[x] - self.cum_lp[self.cur_size-1]

           
    def rvs(self, size = 1, indic = False):
        assert(size >= 0)
        return np.array([self.ppf(stats.uniform.rvs(), indic = indic)
                             for _ in range(size)])

def test_stream():
    up = 10
    lp = log(np.arange(1,up))
    
    sc = stream_categorical(lp, up+1)
    sc_ext = stream_categorical(lp[:up//2], up+1)
    sc_ext.extend(lp[up//2:])
    
    rvs = np.random.rand(100)
    a = np.array([sc.ppf(i) for i in rvs])
    a_ext = np.array([sc_ext.ppf(i) for i in rvs])
    assert(np.all(a == a_ext))