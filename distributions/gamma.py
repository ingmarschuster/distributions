# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:28:49 2016

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import autograd.numpy as np
import autograd.scipy as sp
import autograd.scipy.stats as stats

from autograd.numpy import exp, log, sqrt
from autograd.scipy.misc import logsumexp
from autograd.numpy.linalg import inv
import theano.tensor as T #FIXME: shouldnt be a requirement
import theano


__all__ = ["gamma_logpdf_theano", "gamma_logpdf", "gamma", "invgamma_logpdf_theano", "invgamma"]

def gamma_logpdf(x, shape, rate):    
    if np.any(x <= 0):
        return -np.infty
    return (  np.log(rate) * shape
            - sp.special.gammaln(shape)
            + np.log(x) * (shape-1)
            - rate * x)
def gamma_logpdf_theano(x, shape, rate):    
    return (  T.log(rate) * shape
            - T.gammaln(shape)
            + T.log(x) * (shape-1)
            - rate * x)

def invgamma_logpdf_theano(x, shape, rate):  
    assert()#not properly tested
    return (  T.log(rate) * shape
            - T.gammaln(shape)
            + T.log(x) * (-shape-1)
            - rate / x)

class gamma(object):
    def __init__(self, shape, rate): 
        self.shape = shape
        self.rate = rate
        x = T.dscalar()
        shape = T.dscalar()
        rate = T.dscalar()
        self.logpdf_expr = gamma_logpdf_theano(x, self.shape, self.rate)
        self.logpdf_func = theano.function([x], self.logpdf_expr)
        self.logpdf_grad_x_func = theano.function([x], T.grad(self.logpdf_expr, [x]))
#        self.logpdf_hess_x_func = theano.function([x], T.hessian(self.logpdf_expr, [x]))
    
    def logpdf(self, x, theano_expr = False):
        if not theano_expr:
            return self.logpdf_func(x)
        else:
            return gamma_logpdf_theano(x, self.shape, self.rate)
    
    def logpdf_grad(self, x):
        return self.logpdf_grad_x_func(x)  
    
    def rvs(self, n=1, eig=False):
        stats.gamma.rvs(n, shape=self.shape, rate=self.rate)

class invgamma(object):
    def __init__(self, shape, rate): 
        assert()#not properly tested
        self.shape = shape
        self.rate = rate
        x = T.dscalar()
        shape = T.dscalar()
        rate = T.dscalar()
        self.logpdf_expr = invgamma_logpdf_theano(x, self.shape, self.rate)
        self.logpdf_func = theano.function([x], self.logpdf_expr)
        self.logpdf_grad_x_func = theano.function([x], T.grad(self.logpdf_expr, [x]))
#        self.logpdf_hess_x_func = theano.function([x], T.hessian(self.logpdf_expr, [x]))
    
    def logpdf(self, x, theano_expr = False):
        if not theano_expr:
            return self.logpdf_func(x)
        else:
            return invgamma_logpdf_theano(x, self.shape, self.rate)
    
    def logpdf_grad(self, x):
        return self.logpdf_grad_x_func(x)  
    
    def rvs(self, n=1, eig=False):
        stats.invgamma.rvs(n, shape=self.shape, rate=self.rate)