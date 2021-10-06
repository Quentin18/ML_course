# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np


def compute_loss(y, tx, w, mae=False):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = len(y)
    e = y - (tx @ w)
    if mae:
        return np.sum(abs(e)) / N
    return (e @ e) / (2 * N)
