# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""
import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi_n = lambda x_n: [x_n**d for d in range(degree + 1)]
    return np.array(list(map(phi_n, x)))
