from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.random as npr

from numpy import exp, log
from numpy.linalg import inv, cholesky, det
from scipy.special import multigammaln
from scipy.stats import chi2
import scipy.stats as stats

from .. import *
from .util import assert_equal


def test_categorical():
    dist = categorical(np.array((0.5, 0.3,0.1,0.1)))
    #assert(dist.ppf(0.4) == 0 and dist.ppf(0.6) == 1 and dist.ppf(0.95) == 3)
    for (p, idx) in [(0.4, 0), (0.5, 0), (0.6, 1), (0.95, 3)]:
        assert(dist.ppf(p) == idx)
        assert(np.argmax(dist.ppf(p, indic = True)) == idx)
    print(dist.rvs())
    
        
def test_invwishart_logpdf():
    # values from R-package bayesm, function lndIWishart(6.1, a, a)
    a = 4 * np.eye(5)
    assert_equal(invwishart_logpdf(a,a,6.1), -40.526062, 1*10**-5)
    
    a = np.eye(5) + np.ones((5,5))
    assert_equal(invwishart_logpdf(a,a,6.1), -25.1069258, 1*10**-6)
    
    a = 2 * np.eye(5)
    assert_equal(invwishart_logpdf(a,a,6.1), -30.12885519, 1*10**-7)