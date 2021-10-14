# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    n, d = tx.shape
    a = (tx.T @ tx) + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w_star = np.linalg.solve(a, b)
    e = y - (tx @ w_star)
    loss_star = (e.T @ e) / (2 * n)
    return loss_star, w_star
