from __future__ import division, print_function, absolute_import
import numpy as np


def is_equal(a, b, tolerance = 1e-14):
    return (np.abs(a - b) <= tolerance).all()

def assert_equal(a, b, tolerance = 1e-14):
    assert(is_equal(a, b, tolerance = tolerance))