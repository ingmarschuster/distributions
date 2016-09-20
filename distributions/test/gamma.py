# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 07:16:16 2014

@author: Ingmar Schuster
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log, exp
import numpy.random as npr
from numpy.linalg import inv, cholesky
from scipy.special import multigammaln, gammaln
from scipy.stats import chi2
import scipy.stats as stats

from .util import assert_equal
from distributions import gamma, invgamma


def test_gamma():
    # values from R-package bayesm, function dmvt(6.1, a, a)
    for (x, shape, rate, lpdf_gamm, lpdf_invg) in [
                                   (1, 2, 5, -1.7811241751317992143555102302343584597110748291015625, -3.4188758248682003) ,
                                   (1, 1, 4, -2.613705638880109205501867108978331089019775390625, -1.6362943611198906),
                                   (1, 100, 4, -224.50476925758630386553704738616943359375, -498.01364148156449) ]:
        for (dist, lpdf) in [(gamma(shape, rate),lpdf_gamm),
                             (invgamma(shape, rate),lpdf_invg)]:
            ad = np.mean(np.abs(dist.logpdf(x) - lpdf))
            assert(ad < 10**-5)
#        assert(np.all(opt.check_grad(dist.logpdf, dist.logpdf_grad, x) < 10**-4))