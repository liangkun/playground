#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as pp


def sigmoid(z):
    """ Sigmoid function.
    """

    return 1 / (1 + np.exp(-z))


def log_likelihood(xs, ys, theta):
    """ Compute the log likelihood cost and the gradient w.r.t. theta.
    """

    xs = np.mat(xs)
    ys = np.mat(ys).T
    theta = np.mat(theta).T
    m = xs.shape[0]

    h_xs = sigmoid(xs * theta)
    cost = (ys.T * np.log(h_xs) + (1 - ys).T * np.log(1 - h_xs)) / m
    gradient = (xs.T * (ys - h_xs)) / m

    return cost[0,0], np.asarray(gradient).flatten()


def load_data(fname):
    """ Load data from file, returns as numpy arrays.

    Note: xs do not contains the intercept term.
    """
    xs = []
    ys = []
    with open(fname, 'r') as f:
        for line in f:
            elements = line.strip().split()
            xs.append([float(x) for x in elements[:-1]])
            ys.append(float(elements[-1]))

    return np.array(xs), np.array(ys)


def plot_data(xs, ys, theta=None):
    """ Plot the data points.
    """

    neg = 1 - ys
    pp.plot(ma.masked_array(xs[:, 0], ys),
            ma.masked_array(xs[:, 1], ys),
            'rx')
    pp.plot(ma.masked_array(xs[:, 0], neg),
            ma.masked_array(xs[:, 1], neg),
            'bo')

    lengends = ['positive', 'negative']

    if theta is not None and len(theta) == 3:
        min_x0 = np.min(xs[:, 0])
        min_x1 = (theta[0] + theta[1] * min_x0) / (- theta[2])
        max_x0 = np.max(xs[:, 0])
        max_x1 = (theta[0] + theta[1] * max_x0) / (- theta[2])
        pp.plot([min_x0, max_x0], [min_x1, max_x1], 'k-')

    pp.legend(lengends, 'lower right')
    pp.show()


def gradient_ascend(xs, ys, alpha = 0.1, max_iter = 1000, epslon = 1E-4, theta = None, plot = False):
    """ Run gradient acend to get the best fitted theta.
    """
    m, n = xs.shape
    xs = np.hstack((np.ones((m, 1)), xs))
    if theta is None:
        theta = np.zeros((n + 1))
    likelihoods = []

    # gradient acend
    last_likelihood = 1
    for i in range(max_iter):
        likelihood, gradient = log_likelihood(xs, ys, theta)
        likelihoods.append(likelihood)
        if abs(likelihood - last_likelihood) < epslon:
            break
        last_likelihood = likelihood

        theta += alpha * gradient

    if plot:
        pp.plot(np.arange(len(likelihoods)), likelihoods)
        pp.legend(['likelihood trend'])
        pp.show()

    return theta


if __name__ == '__main__':
    xs, ys = load_data('data/Ch05/testSet.txt')
    theta = gradient_ascend(xs, ys, plot=True)
    plot_data(xs, ys, theta)