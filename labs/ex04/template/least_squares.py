# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    n = len(y)
    w_star = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - (tx @ w_star)
    loss_star = (e.T @ e) / (2 * n)
    return loss_star, w_star
