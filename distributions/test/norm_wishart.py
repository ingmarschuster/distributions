# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:29:35 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

import distributions




def test_wishart_invwishart_pdf():
    m1 = np.array([[ 2. , -0.3], [-0.3,  4. ]])
    m2 = np.array([[ 1. ,  0.3],[ 0.3,  1. ]])
    assert(np.abs(distributions.wishart_logpdf(m1, m2, 3) + 5.7851626233668245) < 0.5)
    assert(np.abs(distributions.invwishart_logpdf(m2, m1, 3) +2.5415049314906186) < 0.5)