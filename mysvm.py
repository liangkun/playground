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

    return data, labels


def select_j_random(i, m):
    j = i
    while j == i:
        j = random.randint(0, m - 1)

    return j


def clip_value(a, low, high):
    if a > high:
        return high
    if a < low:
        return low
    return a


def plot_data(data, labels, svmask):
    data = np.array(data)
    labels = np.array(labels)

    pos = data[labels == 1]
    neg = data[labels == -1]

    pp.scatter(pos[:, 0], pos[:, 1], marker='x', c='r', s=15)
    pp.scatter(neg[:, 0], neg[:, 1], marker='o', c='g', s=15)
    pp.scatter(data[svmask][:, 0], data[svmask][:, 1], marker='o', c='k', s=40)
    pp.show()


def linear_kernel(X1, X2):
    return X1 * X2


def simple_smo(data, labels, c, epsilon, max_iter, kernel=linear_kernel):
    data = np.mat(data)
    labels = np.mat(labels).T
    b = 0
    m, n = data.shape
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0

    while iter < max_iter:
        alphas_changed = 0
        for i in range(m):
            f_xi = np.multiply(alphas, labels).T * kernel(data, data[i, :].T) + b
            ei = f_xi - labels[i]
            if (labels[i] * ei < -epsilon) and (alphas[i] < c) or \
                    (labels[i] * ei > epsilon) and (alphas[i] > 0):

                j = select_j_random(i, m)
                f_xj = np.multiply(alphas, labels). T * kernel(data, data[j, :].T) + b
                ej = f_xj - labels[j]

                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if labels[i] != labels[j]:
                    low = max(0, alphas[j] - alphas[i])
                    high = min(c, c + alphas[j] - alphas[i])
                else:
                    low = max(0, alphas[j] + alphas[i] - c)
                    high = min(c, alphas[j] + alphas[i])

                if low == high:
                    print('DEBUG: low == high, skip this pair')
                    continue

                eta = 2 * kernel(data[i, :], data[j, :].T) - \
                    kernel(data[i, :], data[i, :].T) - \
                    kernel(data[j, :], data[j, :].T)

                if eta >= 0:
                    print('DEBUG: eta >= 0, skip this pair')
                    continue

                alphas[j] -= labels[j] * (ei - ej) / eta
                alphas[j] = clip_value(alphas[j], low, high)
                alphas[i] += labels[j] * labels[i] * (alpha_j_old - alphas[j])

                b1 = b - ei - labels[i] * (alphas[i] - alpha_i_old) * kernel(data[i, :], data[i, :].T) - \
                    labels[j] * (alphas[j] - alpha_j_old) * kernel(data[i, :], data[j, :].T)
                b2 = b - ej - labels[i] * (alphas[i] - alpha_i_old) * kernel(data[i, :], data[j, :].T) - \
                    labels[j] * (alphas[j] - alpha_j_old) * kernel(data[j, :], data[j, :].T)

                if 0 < alphas[i] < c: b = b1
                elif 0 < alphas[j] < c: b = b2
                else: b = (b1 + b2) / 2
                alphas_changed += 1

        if alphas_changed == 0:
            iter += 1
        else:
            iter = 0

    return b, alphas


class SVM:
    def __init__(self, data, labels, c=0.6, epsilon=0.001, kernel=linear_kernel):
        self.xs = np.mat(data)
        self.m, self.n = self.xs.shape
        self.ys = np.mat(labels).T
        self.c = c
        self.epsilon = epsilon
        self.kernel = kernel
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))

    def calc_ek(self, k):
        f_xk = np.multiply(self.alphas, self.ys).T * self.kernel(self.xs, self.xs[k, :].T) + self.b
        ek = f_xk - self.ys[k]
        return ek

    def select_j(self, i, ei):
        max_k = -1
        max_delta_e = 0
        ej = 0

        self.e_cache[i] = [1, ei]
        valid_caches = np.nonzero(self.e_cache[:, 0].A)[0]
        if len(valid_caches) > 1:
            for k in valid_caches:
                if k == i:
                    continue
                ek = self.calc_ek(k)
                delta = abs(ek - ei)
                if delta > max_delta_e:
                    max_k = k
                    max_delta_e = delta
                    ej = ek
            return max_k, ej
        else:
            j = select_j_random(i, self.m)
            ej = self.calc_ek(j)
            return j, ej

    def update_ek(self, k):
        ek = self.calc_ek(k)
        self.e_cache[k] = [1, ek]
        return ek

    def inner_loop(self, i):
        ei = self.calc_ek(i)
        if (self.ys[i] * ei < -self.epsilon) and (self.alphas[i] < self.c) or \
                (self.ys[i] * ei > self.epsilon) and (self.alphas[i] > 0):
            j, ej = self.select_j(i, ei)
            alpha_i_old = self.alphas[i].copy()
            alpha_j_old = self.alphas[j].copy()
            if self.ys[i] != self.ys[j]:
                low = max(0, self.alphas[j] - self.alphas[i])
                high = min(self.c, self.c + self.alphas[j] - self.alphas[i])
            else:
                low = max(0, self.alphas[j] + self.alphas[i] - self.c)
                high = min(self.c, self.alphas[j] + self.alphas[i])

            if low == high:
                print('DEBUG: low == high, skip this pair')
                return 0

            eta = 2.0 * self.kernel(self.xs[i, :], self.xs[j, :].T) - \
                self.kernel(self.xs[i, :], self.xs[i, :].T) - \
                self.kernel(self.xs[j, :], self.xs[j, :].T)
            if eta >= 0:
                print('DEBUG: eta >= 0, skip this pair')
                return 0

            self.alphas[j] -= self.ys[j] * (ei - ej) / eta
            self.alphas[j] = clip_value(self.alphas[j], low, high)
            self.update_ek(j)

            self.alphas[i] += self.ys[j] * self.ys[i] * (alpha_j_old - self.alphas[j])
            self.update_ek(i)

            b1 = self.b - ei - self.ys[i] * (self.alphas[i] - alpha_i_old) * \
                    self.kernel(self.xs[i, :], self.xs[i, :].T) - \
                    self.ys[j] * (self.alphas[j] - alpha_j_old) * self.kernel(self.xs[i, :], self.xs[j, :].T)
            b2 = self.b - ej - self.ys[i] * (self.alphas[i] - alpha_i_old) * \
                    self.kernel(self.xs[i, :], self.xs[j, :].T) - \
                    self.ys[j] * (self.alphas[j] - alpha_j_old) * self.kernel(self.xs[j, :], self.xs[j, :].T)

            if 0 < self.alphas[i] < self.c:
                self.b = b1
            elif 0 < self.alphas[j] < self.c:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

            return 1
        return 0

    def smo(self, max_iter):
        iter = 0
        full_set = True
        alphas_changed = 0

        while iter < max_iter and (alphas_changed > 0 or full_set):
            alphas_changed = 0
            if full_set:
                for i in range(self.m):
                    alphas_changed += self.inner_loop(i)
                print('DEBUG: full set iteration, iter {}, pairs changed {}'.format(iter, alphas_changed))
                iter += 1
            else:
                non_bounds = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.c))[0]
                for i in non_bounds:
                    alphas_changed += self.inner_loop(i)
                print('DEBUG: non-bounds, iter {}, pairs changed {}'.format(iter, alphas_changed))
                iter += 1

            if full_set:
                full_set = False
            elif alphas_changed == 0:
                full_set = True


if __name__ == '__main__':
    data, labels = load_data_set('data/Ch06/testSet.txt')
    #b, alphas = simple_smo(data, labels, 100, 0.001, 40)
    #plot_data(data, labels, (alphas>0).A[:, 0])
    svm = SVM(data, labels, c=1000)
    svm.smo(80)
    plot_data(data, labels, (svm.alphas>0).A[:, 0])
