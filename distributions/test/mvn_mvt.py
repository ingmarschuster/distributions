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
from .. import mvt, mvnorm

import scipy.optimize as opt

def test_mvt_mvn_logpdf_n_grad():
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
    
def test_mvnorm_fit():
    mu = np.array((2,3,4))
    cov = np.array([(20, 3, 2),
                    (3, 10, 1),
                    (2,  1, 7)])
    
    param_fit = mvnorm.fit(stats.multivariate_normal(mu, cov).rvs(2000000))
    #print(param_fit)
    assert_equal(param_fit[0], mu, 0.2)
    assert_equal(param_fit[1].diagonal(), cov.diagonal(), 0.1)