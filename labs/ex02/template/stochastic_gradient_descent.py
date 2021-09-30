# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import costs
import gradient_descent
import helpers


def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient from just few examples n
    and their corresponding y_n labels.
    """
    return gradient_descent.compute_gradient(y, tx, w)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in helpers.batch_iter(y, tx, batch_size):
            grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = costs.compute_loss(y, tx, w)
            w = w - gamma * grad
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
