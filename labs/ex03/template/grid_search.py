# -*- coding: utf-8 -*-
""" Grid Search"""
import numpy as np
import costs


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    nrows, ncols = len(w0), len(w1)
    losses = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            w = np.array([w0[i], w1[j]])
            losses[i, j] = costs.compute_loss(y, tx, w)
    return losses
