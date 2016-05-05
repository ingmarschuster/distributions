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


__all__ = ["CompTransform", "softplus", "Power", "Shift", "NegateDim"]


class CompTransform(object):
    def __init__(self, wrapped_distribution, transform_components_at_index, transform_forw, transform_backw, transform_backw_grad):        
        assert(len(transform_components_at_index) > 0)
        self.wrapped = wrapped_distribution
        self.idx = transform_components_at_index
        self.transform_forw = transform_forw
        self.transform_backw = transform_backw
        self.transform_backw_grad = transform_backw_grad
        
    def ppf(self, component_cum_prob, eig=True):
        rval = self.wrapped.ppf(component_cum_prob)
        rval[:,self.idx] = self.transform_forw(rval[:,self.idx])
        return rval
        
    def rvs(self, n = 1):
        rval =  self.ppf(stats.uniform.rvs(size = (n, self.wrapped.get_num_unif())))
        if n == 1:
            return rval.flatten()
        else:
            return rval
    
    def get_num_unif(self):
        return self.wrapped.get_num_unif()    
    
    def logpdf(self, rvs_transf):
        rvs_transf = np.atleast_2d(rvs_transf)
        rvs_orig = np.copy(rvs_transf)
        rvs_orig[:,self.idx] = self.transform_backw(rvs_transf[:,self.idx])
        rval = np.atleast_1d(self.wrapped.logpdf(rvs_orig))
        correction = np.log(np.abs(self.transform_backw_grad(rvs_transf[:,self.idx]))).sum(1)
        assert(len(correction) == len(rval))
        if len(correction.shape) < len(rval.shape):
            correction = correction[:, np.newaxis]
        return rval + correction


class Softplus(object):
    def __init__(self, wrapped_distribution, transform_components_at_index):        
        self.transf = CompTransform(wrapped_distribution, transform_components_at_index,
                                    lambda x: log(1+exp(x)),
                                    lambda y: log(exp(y) - 1),
                                    lambda y: 1./(1. - 1./exp(y))
                                   )
    def get_num_unif(self):
        return self.transf.get_num_unif()
                                           
    def ppf(self, component_cum_prob, eig=True):
        return self.transf.ppf(component_cum_prob)
    
    def rvs(self, n = 1):
        return self.transf.rvs(n)
    
    def logpdf(self, rvs_transf):
        return self.transf.logpdf(rvs_transf)

class NegateDim(object):
    def __init__(self, wrapped_distribution, transform_components_at_index):        
        self.transf = CompTransform(wrapped_distribution, transform_components_at_index,
                                    lambda x: -x,
                                    lambda y: -y,
                                    lambda y: -1
                                   )
    def get_num_unif(self):
        return self.transf.get_num_unif()
                                           
    def ppf(self, component_cum_prob, eig=True):
        return self.transf.ppf(component_cum_prob)
    
    def rvs(self, n = 1):
        return self.transf.rvs(n)
    
    def logpdf(self, rvs_transf):
        return self.transf.logpdf(rvs_transf)


class Power(object): # you can get the banana transform by choosing power == 2
    def __init__(self, wrapped_distribution, transform_components_at_index, b, shift, power):
        if np.any(transform_components_at_index == 0):
            #first coordinate is untransformed
            assert()
        assert(len(transform_components_at_index) > 0)
        assert(power > 1)
        self.power = power
        self.shift = shift
        self.b = b
        self.wrapped = wrapped_distribution
        self.idx = transform_components_at_index
        
        #the following functions work on one random variable
        self.transform_forw = lambda rv: np.r_[rv[:1], rv[1:] + self.b*(rv[0]**self.power-self.shift)]
        self.transform_backw = lambda rv: np.r_[rv[:1], rv[1:] - self.b*(rv[0]**self.power-self.shift)] # y_0 = x_0

    def get_num_unif(self):
        return self.wrapped.get_num_unif()
        
    def ppf(self, component_cum_prob, eig=True):
        rval = self.wrapped.ppf(component_cum_prob)
        return np.apply_along_axis(self.transform_forw, 1, rval)
        
    def rvs(self, n = 1):
        rval =  self.ppf(stats.uniform.rvs(size = (n, self.wrapped.get_num_unif())))
        if n == 1:
            return rval.flatten()
        else:
            return rval
    
    def log_abs_det_jacobian(self, one_rv):
        return 0
        jacobian = np.zeros([one_rv.size, one_rv.size])
        jacobian[:,0] = -self.power*self.b*one_rv.flat[0]**(self.power - 1)
        np.fill_diagonal(jacobian, 1)
        det = np.linalg.det(jacobian)
        print(jacobian, det)
        return np.log(np.abs(det))
    
    def logpdf(self, rvs_transf):
        rvs_transf = np.atleast_2d(rvs_transf)
        rvs_orig = np.apply_along_axis(self.transform_backw, 1, rvs_transf)
        rval = np.atleast_1d(self.wrapped.logpdf(rvs_orig))
        
        # abs determinant of Jacobian is 1: 1s on diagonal,
        # only first column of does not contain 0s
        # => determinant is product of diagonal elements, 1
        # => no further correction necessary
        return rval 
        
        
class Shift(object):
    def __init__(self, wrapped_distribution, offset):  
        assert(len(offset.shape) == 1)
        #FIXME: transform at all indices
        self.wrapped = wrapped_distribution
        self.offset = np.atleast_2d(offset)
        
    def get_num_unif(self):
        return self.wrapped.get_num_unif()
        
    def ppf(self, component_cum_prob, eig=True):
        return self.wrapped.ppf(component_cum_prob) + np.repeat(self.offset, len(component_cum_prob), 0)
        
    def rvs(self, n = 1):
        rval =  self.ppf(stats.uniform.rvs(size = (n, self.wrapped.get_num_unif())))
        if n == 1:
            return rval.flatten()
        else:
            return rval

    
    def logpdf(self, rvs_transf):
        rvs_transf = np.atleast_2d(rvs_transf)
        return self.wrapped.logpdf(rvs_transf - np.repeat(self.offset, len(rvs_transf), 0))

