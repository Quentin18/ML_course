# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    n, m = tx.shape
    w_star = np.linalg.solve((tx.T @ tx) + lambda_ * np.eye(m), tx.T @ y)
    e = y - (tx @ w_star)
    loss_star = (e.T @ e) / (2 * m)
    return loss_star, w_star
