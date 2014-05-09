#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as pp


def load_data_set(fname):
    data = []
    labels = []
    with open(fname, 'r') as f:
        for line in f:
            elements = line.strip().split('\t')
            data.append([float(elem) for elem in elements[:-1]])
            labels.append(float(elements[-1]))

    return np.array(data), np.array(labels)


def select_j_random(i, m):
    j = i
    while j == i:
        j = random.randint(0, m - 1)

    return j


def clip_value(a, low, high):
    if a > high:
        a = high
    if a < low:
        a = low
    return a


def plot_line(w, b, min_x0, max_x0):
    min_x1 = (-(w[0] * min_x0 + b) / w[1])[0, 0]
    max_x1 = (-(w[0] * max_x0 + b) / w[1])[0, 0]

    pp.plot([min_x0, max_x0], [min_x1, max_x1], 'k-')


def plot_boundry(os):
    min_x0 = np.min(os.xs, axis=0)[0, 0]
    max_x0 = np.max(os.xs, axis=0)[0, 0]
    min_x1 = np.min(os.xs, axis=0)[0, 1]
    max_x1 = np.max(os.xs, axis=0)[0, 1]

    x0s = np.arange(min_x0, max_x0, (max_x0 - min_x0) / 200)
    x1s = np.arange(min_x1, max_x1, (max_x1 - min_x1) / 200)

    x0_grid, x1_grid = np.meshgrid(x0s, x1s)
    m, n = x0_grid.shape
    h_grid = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            h_grid[i, j] = svm_classify(os, [x0_grid[i, j], x1_grid[i, j]])

    pp.contourf(x0_grid, x1_grid, h_grid, alpha=0.2)


def plot_data(data, labels, os=None):
    pos = data[labels == 1]
    neg = data[labels == -1]

    pp.scatter(pos[:, 0], pos[:, 1], marker='x', c='r')
    pp.scatter(neg[:, 0], neg[:, 1], marker='o', c='c', linewidths=0)
    if os is not None:
        pp.scatter(os.svxs.A[os.svys.A.flatten() == 1][:, 0].flatten(),
                   os.svxs.A[os.svys.A.flatten() == 1][:, 1].flatten(), marker='s', c='r')
        pp.scatter(os.svxs.A[os.svys.A.flatten() == -1][:, 0].flatten(),
                   os.svxs.A[os.svys.A.flatten() == -1][:, 1].flatten(), marker='s', c='c')
        if os.kernel == linear_kernel:
            plot_line(calc_w_linear_kernel(os), os.b,
                      np.min(data, axis=0)[0] + 2, np.max(data, axis=0)[0] - 2)
        else:
            plot_boundry(os)
    pp.show()


def linear_kernel(x1, x2):
    """ Input must be two numpy matrix.
    """
    return x1 * x2


def create_gaussian(sigma=1):
    def guassian(x1, x2):
        """ Input should be two numpy matrix.
        """
        return np.mat(np.exp(-np.sum((x1 - x2.T).A ** 2, axis=1) / (2 * sigma**2))).T

    return guassian


def simple_smo(data, labels, c=0.6, epsilon=0.001, max_iter=40, kernel=linear_kernel):
    xs = np.mat(data)
    ys = np.mat(labels).T
    b = 0
    m, n = xs.shape
    alphas = np.mat(np.zeros((m, 1)))

    iter = 0
    while iter < max_iter:
        alphas_changed = 0
        for i in range(m):
            f_xi = np.multiply(alphas, ys).T * kernel(xs, xs[i, :].T) + b
            ei = f_xi - ys[i]
            if ((ys[i] * ei < -epsilon) and (alphas[i] < c)) or \
                    ((ys[i] * ei > epsilon) and (alphas[i] > 0)):

                j = select_j_random(i, m)
                f_xj = np.multiply(alphas, ys). T * kernel(xs, xs[j, :].T) + b
                ej = f_xj - ys[j]

                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                if ys[i] != ys[j]:
                    low = max(0, alphas[j] - alphas[i])
                    high = min(c, c + alphas[j] - alphas[i])
                else:
                    low = max(0, alphas[j] + alphas[i] - c)
                    high = min(c, alphas[j] + alphas[i])

                if low == high:
                    #print('DEBUG: low == high, skip this pair')
                    continue

                eta = 2 * kernel(xs[i, :], xs[j, :].T) - \
                    kernel(xs[i, :], xs[i, :].T) - \
                    kernel(xs[j, :], xs[j, :].T)

                if eta >= 0:
                    #print('DEBUG: eta >= 0, skip this pair')
                    continue

                alphas[j] -= ys[j] * (ei - ej) / eta
                alphas[j] = clip_value(alphas[j], low, high)
                if abs(alpha_j_old - alphas[j]) < 0.00001:
                    #print('DEBUG: j not moving enough, skip')
                    continue
                alphas[i] += ys[j] * ys[i] * (alpha_j_old - alphas[j])

                b1 = b - ei - ys[i] * (alphas[i] - alpha_i_old) * kernel(xs[i, :], xs[i, :].T) - \
                    ys[j] * (alphas[j] - alpha_j_old) * kernel(xs[i, :], xs[j, :].T)
                b2 = b - ej - ys[i] * (alphas[i] - alpha_i_old) * kernel(xs[i, :], xs[j, :].T) - \
                    ys[j] * (alphas[j] - alpha_j_old) * kernel(xs[j, :], xs[j, :].T)

                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphas_changed += 1

        if alphas_changed == 0:
            iter += 1
        else:
            iter = 0

    return b, alphas


class opt_struct:
    def __init__(self, data, labels, c=0.6, epsilon=0.001, kernel=linear_kernel):
        self.xs = data
        self.ys = labels
        self.c = c
        self.epsilon = epsilon
        self.kernel = kernel
        self.m = data.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))


def calc_ek(os, k):
    f_xk = np.multiply(os.alphas, os.ys).T * os.kernel(os.xs, os.xs[k, :].T) + os.b
    return f_xk - os.ys[k]


def select_j(os, i, ei):
    max_k = -1
    max_delta = 0
    ej = 0

    os.e_cache[i] = [1, ei]
    valid_caches = np.nonzero(os.e_cache[:, 0].A)[0]
    if len(valid_caches) > 1:
        for k in valid_caches:
            if k == i:
                continue
            ek = calc_ek(os, k)
            delta = abs(ek - ei)
            if delta > max_delta:
                max_delta = delta
                max_k = k
                ej = ek
        return max_k, ej
    else:
        j = select_j_random(i, os.m)
        ej = calc_ek(os, j)
        return j, ej


def update_ek(os, k):
    ek = calc_ek(os, k)
    os.e_cache[k] = [1, ek]


def inner_loop(os, i):
    ei = calc_ek(os, i)
    if (os.ys[i] * ei < -os.epsilon and os.alphas[i] < os.c) or \
            (os.ys[i] * ei > os.epsilon and os.alphas[i] > 0):
        j, ej = select_j(os, i, ei)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()
        if os.ys[i] != os.ys[j]:
            low = max(0, os.alphas[j] - os.alphas[i])
            high = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            low = max(0, os.alphas[j] + os.alphas[i] - os.c)
            high = min(os.c, os.alphas[j] + os.alphas[i])
        if low == high:
            #print('DEBUG: low == high, skip this pair')
            return 0
        eta = 2 * os.kernel(os.xs[i, :], os.xs[j, :].T) - \
              os.kernel(os.xs[i, :], os.xs[i, :].T) - \
              os.kernel(os.xs[j, :], os.xs[j, :].T)
        if eta >=0:
            #print('DEBUG: eta >=0, skip this pair')
            return 0
        os.alphas[j] -= os.ys[j] * (ei - ej) / eta
        os.alphas[j] = clip_value(os.alphas[j], low, high)
        update_ek(os, j)
        if abs(os.alphas[j] - alpha_j_old) < 0.00001:
            #print('DEBUG: j not moving enough, skip')
            return 0
        os.alphas[i] += os.ys[j] * os.ys[i] * (alpha_j_old - os.alphas[j])
        update_ek(os, i)
        b1 = os.b - ei - os.ys[i] * (os.alphas[i] - alpha_i_old) * \
                         os.kernel(os.xs[i, :], os.xs[i, :].T) - \
                         os.ys[j] * (os.alphas[j] - alpha_j_old) * \
                         os.kernel(os.xs[i, :], os.xs[j, :].T)
        b2 = os.b - ej - os.ys[i] * (os.alphas[i] - alpha_i_old) * \
                         os.kernel(os.xs[i, :], os.xs[j, :].T) - \
                         os.ys[j] * (os.alphas[j] - alpha_j_old) * \
                         os.kernel(os.xs[j, :], os.xs[j, :].T)
        if 0 < os.alphas[i] < os.c:
            os.b = b1
        elif 0 < os.alphas[j] < os.c:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2

        return 1
    return 0


def smo(data, labels, c=1, epsilon=0.01, max_iter=5000, kernel=linear_kernel):
    os = opt_struct(np.mat(data), np.mat(labels).T, c, epsilon, kernel)
    iter = 0
    full_set = True
    alphas_changed = 0
    while iter < max_iter and (alphas_changed > 0 or full_set):
        alphas_changed = 0
        if full_set:
            for i in range(os.m):
                alphas_changed += inner_loop(os, i)
            iter += 1
        else:
            non_boundis = np.nonzero((os.alphas.A > 0) * (os.alphas.A < os.c))[0]
            for i in non_boundis:
                alphas_changed += inner_loop(os, i)
            iter += 1

        if full_set:
            full_set = False
        elif alphas_changed == 0:
            full_set = True

    msk = (os.alphas > 0).A[:, 0]
    os.svxs = os.xs[msk]
    os.svys = os.ys[msk]
    os.svas = os.alphas[msk]
    return os


def calc_w_linear_kernel(os):
    m, n = os.svxs.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += os.svas[i, 0] * os.svys[i, 0] * os.svxs[i, :].T
    return w


def svm_classify(os, x):
    kernal_value = os.kernel(os.svxs, np.mat(x).T)
    f_x = kernal_value.T * np.multiply(os.svys, os.svas) + os.b
    if f_x >= 0:
        return 1.0
    else:
        return -1.0


def get_error_rate(os, test_xs, test_ys):
    test_count = test_xs.shape[0]
    test_error = 0
    for i in range(test_count):
        if test_ys[i] != svm_classify(os, test_xs[i, :]):
            test_error += 1

    return test_error / test_count

def test_rbf(sigma):
    train = 'data/Ch06/testSetRBF.txt'
    test = 'data/Ch06/testSetRBF2.txt'

    train_xs, train_ys = load_data_set(train)
    os = smo(train_xs, train_ys, c=200, epsilon=0.0001, max_iter=10000, kernel=create_gaussian(sigma))
    plot_data(train_xs, train_ys, os)

    test_xs, test_ys = load_data_set(test)
    print('SVM rbf error rate on train set: %{}'.format(get_error_rate(os, train_xs, train_ys) * 100))
    print('SVM rbf error rate on test set: %{}'.format(get_error_rate(os, test_xs, test_ys) * 100))


if __name__ == '__main__':
    data, labels = load_data_set('data/Ch06/testSet.txt')
    os = smo(data, labels)
    plot_data(data, labels, os)

    test_rbf(1.3)

