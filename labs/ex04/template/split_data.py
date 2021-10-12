# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # split the data based on the given ratio
    n = x.shape[0]
    split_idx = int(n * ratio)
    indices = np.random.permutation(n)
    training_idx, test_idx = indices[:split_idx], indices[split_idx:]
    training_x, test_x = x[training_idx], x[test_idx]
    training_y, test_y = y[training_idx], y[test_idx]
    return training_x, test_x, training_y, test_y
