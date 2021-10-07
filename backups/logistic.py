#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as pp
import random


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


def normalize(xs):
    """ Normalize all the feature into (0, 1)
    """

    max_values = np.max(xs, axis=0)
    min_values = np.min(xs, axis=0)

    return (xs - min_values) / (max_values - min_values), max_values, min_values


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


def gradient_ascend(xs, ys, alpha=0.1, max_iter=5000, epslon=1E-4, theta=None, plot=False):
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


def stochastic_gradient_ascend(xs, ys, alpha=0.1, max_iter=20, epslon=1E-4, theta=None, plot=False):
    """ Run stocchastic gradient ascend.
    """

    m, n = xs.shape
    xs = np.hstack((np.ones((m,1)), xs))
    if theta is None:
        theta = np.zeros((n + 1))
    likelihoods = []

    last_likelihood = 1
    for i in range(max_iter):
        idxes = list(range(m))
        for j in range(m):
            new_alpha = 4 / (i + j+ 1) + alpha
            randidx = random.randint(0, len(idxes)-1)
            sample_idx = idxes[randidx]
            likelihood, gradient = log_likelihood(xs[sample_idx, :], ys[sample_idx], theta)
            del idxes[randidx]

            likelihoods.append(likelihood)
            # this does not apply because of the `big' variance of likelihood
            #if abs(likelihood - last_likelihood) < epslon:
            #    break
            #last_likelihood = likelihood

            theta += new_alpha * gradient

    if plot:
        pp.plot(np.arange(len(likelihoods)), likelihoods)
        pp.legend(['likelihood trend'])
        pp.show()

    return theta


def classify(theta, input):
    if theta.dot(input) >= 0:
        return 1
    else:
        return 0


def colic_test(stochastic=False):
    train_xs, train_ys = load_data('data/Ch05/horseColicTraining.txt')
    train_xs, max_values, min_values = normalize(train_xs)

    if stochastic:
        theta = stochastic_gradient_ascend(train_xs, train_ys)
    else:
        theta = gradient_ascend(train_xs, train_ys)

    test_xs, test_ys = load_data('data/Ch05/horseColicTest.txt')
    test_xs = (test_xs - min_values) / (max_values - min_values)
    test_count = len(test_ys)
    test_xs = np.hstack((np.ones((test_count, 1)), test_xs))
    test_error = 0
    for i in range(test_count):
        label = classify(theta, test_xs[i, :])
        if label != test_ys[i]:
            test_error += 1

    error_rate = test_error / test_count
    print('Error rate on test set is %{}'.format(error_rate * 100))
    return error_rate


def multiple_test(n=10, stochasitic=False):
    if not stochasitic:
        colic_test(stochasitic)
    else:
        error_rates = 0.0
        for i in range(n):
            error_rates += colic_test(stochasitic)

        print('\n{} times tests, average error rate is %{}'.format(n, error_rates / n * 100))


if __name__ == '__main__':
    #xs, ys = load_data('data/Ch05/testSet.txt')
    #xs, ys = load_data('data/Ch05/horseColicTraining.txt')
    #theta = gradient_ascend(xs, ys, plot=True)
    #plot_data(xs, ys, theta)
    #theta = stochastic_gradient_ascend(xs, ys, plot=True)
    #plot_data(xs, ys, theta)
    multiple_test(stochasitic=True)
    multiple_test()