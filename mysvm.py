#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as pp
from os import listdir
from concurrent.futures import ProcessPoolExecutor
from grogress import begin_progress, progress, end_progress
import cProfile
#import winsound
import pca


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


def plot_boundry(data, os):
    min_x0 = np.min(data, axis=0)[0]
    max_x0 = np.max(data, axis=0)[0]
    min_x1 = np.min(data, axis=0)[1]
    max_x1 = np.max(data, axis=0)[1]

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
    if data.shape[1] > 2:
        return

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
            plot_boundry(data, os)
    pp.show()


def linear_kernel(x1, x2):
    """ Input must be two numpy matrix.
    """
    return x1 * x2


def gaussian(x1, x2, sigma):
    return np.mat(np.exp(-np.sum((x1 - x2.T).A ** 2, axis=1) / (2 * sigma**2))).T


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


def create_k_cache(data, kernel):
    xs = np.mat(data)
    m = xs.shape[0]
    k_cache = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(m):
            k_cache[i, j] = (kernel(xs[i, :], xs[j, :].T)[0, 0]).real

    return k_cache


class opt_struct:
    def __init__(self, data, labels, c=0.6, epsilon=0.001, kernel=linear_kernel, k_cache=None):
        self.xs = data
        self.ys = labels
        self.c = c
        self.epsilon = epsilon
        self.kernel = kernel
        self.m = data.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))

        if k_cache is None:
            self.k_cache = create_k_cache(self.xs, self.kernel)
        else:
            assert(len(k_cache) == self.m)
            self.k_cache = k_cache

        self.used = np.zeros(self.m)

    def compack(self):
        self.xs = None
        self.ys = None
        self.m = None
        self.alphas = None
        self.e_cache = None
        self.k_cache = None
        self.used = None


def calc_ek(os, k):
    if os.e_cache[k, 0] == 1:
        ek = os.e_cache[k, 1]
    else:
        f_xk = np.multiply(os.alphas, os.ys).T * os.k_cache[:, k] + os.b
        ek = f_xk - os.ys[k]
        os.e_cache[k] = [1, ek]

    return ek


def update_usage(os, uses=None):
    if uses is None:
        os.used = 0
    else:
        os.used[uses] = 1


def invalid_e_cache(os, k=None):
    if k is None:
        os.e_cache[:, 0] = 0
    else:
        os.e_cache[k, 0] = 0


def select_j(os, i, ei):
    max_k = -1
    max_delta = 0
    ej = 0

    used = np.nonzero(os.used)[0]
    if len(used) > 0:
        for k in used:
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


def inner_loop(os, i):
    ei = calc_ek(os, i)
    if (os.ys[i] * ei < -os.epsilon and os.alphas[i] < os.c) or \
            (os.ys[i] * ei > os.epsilon and os.alphas[i] > 0):
        j, ej = select_j(os, i, ei)
        alpha_i_old = os.alphas[i, 0]
        alpha_j_old = os.alphas[j, 0]
        if os.ys[i] != os.ys[j]:
            low = max(0, os.alphas[j] - os.alphas[i])
            high = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            low = max(0, os.alphas[j] + os.alphas[i] - os.c)
            high = min(os.c, os.alphas[j] + os.alphas[i])
        if low == high:
            #print('DEBUG: low == high, skip this pair')
            return 0
        eta = 2 * os.k_cache[i, j] - os.k_cache[i, i] - os.k_cache[j, j]
        if eta >= 0:
            #print('DEBUG: eta >=0, skip this pair')
            return 0
        alpha_j_new = os.alphas[j] - os.ys[j] * (ei - ej) / eta
        alpha_j_new = clip_value(alpha_j_new, low, high)
        if abs(alpha_j_new - alpha_j_old) < 0.00001:
            #print('DEBUG: j not moving enough, skip')
            return 0
        os.alphas[j] = alpha_j_new
        os.alphas[i] += os.ys[j] * os.ys[i] * (alpha_j_old - os.alphas[j])
        invalid_e_cache(os)
        update_usage(os, [i, j])
        b1 = os.b - ei - os.ys[i] * (os.alphas[i] - alpha_i_old) * os.k_cache[i, i] - \
                         os.ys[j] * (os.alphas[j] - alpha_j_old) * os.k_cache[i, j]
        b2 = os.b - ej - os.ys[i] * (os.alphas[i] - alpha_i_old) * os.k_cache[i, j] - \
                         os.ys[j] * (os.alphas[j] - alpha_j_old) * os.k_cache[j, j]
        if 0 < os.alphas[i] < os.c:
            os.b = b1
        elif 0 < os.alphas[j] < os.c:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2

        return 1
    return 0


def smo(data, labels, c=1, epsilon=0.01, max_iter=5000, kernel=linear_kernel, k_cache=None):
    os = opt_struct(np.mat(data), np.mat(labels).T, c, epsilon, kernel, k_cache)
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
    os.compack()
    return os


def calc_w_linear_kernel(os):
    m, n = os.svxs.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += os.svas[i, 0] * os.svys[i, 0] * os.svxs[i, :].T
    return w


def svm_classify(os, x):
    if svm_predict(os, x) >= 0:
        return 1.0
    else:
        return -1.0


def svm_predict(os, x):
    kernal_value = os.kernel(os.svxs, np.mat(x).T)
    f_x = kernal_value.T * np.multiply(os.svys, os.svas) + os.b
    return f_x


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


def image2vector(fname):
    vec = []
    with open(fname, 'r') as f:
        for line in f:
            vec.extend([float(x) for x in line[:-1]])

    return vec


def load_digits(dirname, skip_rate=0.9):
    fnames = listdir(dirname)
    data = []
    labels = []
    skip = -1
    for fname in fnames:
        skip += 1
        if skip < 10 * skip_rate:
            continue
        labels.append(float(fname.split('_')[0]))
        data.append(image2vector(dirname + '/' + fname))
        if skip > 9:
            skip = -1
        progress()

    return np.array(data), np.array(labels)


def svm_multi_classify(oss, x):
    max_predict = 0
    max_class = -1

    for i, os in enumerate(oss):
        predict = svm_predict(os, x)
        if predict > max_predict:
            max_predict = predict
            max_class = i

    return max_class


def multi_get_error_rate(oss, test_xs, test_ys):
    test_count = test_xs.shape[0]
    test_error = 0
    for i in range(test_count):
        if test_ys[i] != float(svm_multi_classify(oss, test_xs[i, :])):
            test_error += 1

    return test_error / test_count


def train_svm(i, xs, ys, c, epsilon, max_iter, kernel, k_cache=None):
    ys_i = ys.copy()
    ys_i[ys == i] = 1
    ys_i[ys != i] = -1
    return i, smo(xs, ys_i, c, epsilon, max_iter, kernel, k_cache=k_cache)


def test_hand_written(c=200, epsilon=0.0001, max_iter=10000, kernel=linear_kernel,
                      parallel=False, skip_rate=0, vrate=0.90):
    train_dir = 'data/Ch02/digits/trainingDigits'
    test_dir = 'data/Ch02/digits/testDigits'
    begin_progress('Reading train data')
    train_xs, train_ys = load_digits(train_dir, skip_rate)
    end_progress()

    begin_progress('Reading test data')
    test_xs, test_ys = load_digits(test_dir, skip_rate)
    end_progress()

    pcs, means = pca.pca(train_xs, vrate=vrate)
    train_xs = pca.transform(train_xs, pcs, means)
    test_xs = pca.transform(test_xs, pcs, means)

    print("Dimension reduction from {} to {}".format(*pcs.shape))

    begin_progress('Train svms')
    num_classes = 10
    svms = [None] * num_classes

    if not parallel:
        k_cache = create_k_cache(train_xs, kernel)
        for i in range(num_classes):
            k, os = train_svm(i, train_xs, train_ys, c, epsilon, max_iter, kernel, k_cache)
            svms[k] = os
            progress()
    else:
        def done_hook(future):
            nonlocal svms
            i, svm = future.result()
            svms[i] = svm
            progress()

        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(train_svm,
                                       i,
                                       train_xs,
                                       train_ys,
                                       c,
                                       epsilon,
                                       max_iter,
                                       kernel)
                       for i in range(num_classes)]
            for future in futures:
                future.add_done_callback(done_hook)

    end_progress()

    print('Testing svms:')
    train_er = multi_get_error_rate(svms, train_xs, train_ys) * 100
    print('SVM handwritten error rate on train set: %{}'.format(train_er))
    test_er = multi_get_error_rate(svms, test_xs, test_ys) * 100
    print('SVM handwritten error rate on test set: %{}'.format(test_er))
    return train_er, test_er


def gaussian_x(x1, x2):
    return gaussian(x1, x2, 7.3)

if __name__ == '__main__':
    #data, labels = load_data_set('data/Ch06/testSet.txt')
    #os = smo(data, labels)
    #plot_data(data, labels, os)

    #test_rbf(1.3)
    #test_hand_written()
    repeat_count = 10
    train_total_er = 0
    test_total_er = 0
    for i in range(repeat_count):
        train_er, test_er = test_hand_written(c=200,
                                              epsilon=0.0001,
                                              max_iter=100000,
                                              kernel=gaussian_x,
                                              parallel=True,
                                              skip_rate=0.5,
                                              vrate=0.90)
        train_total_er += train_er
        test_total_er += test_er

    print('SVM avg train & test error rate on hand written digits: %{}, %{}'
          .format(train_total_er/repeat_count, test_total_er/repeat_count))

    #winsound.PlaySound('C:\Windows\Media\Ring08', winsound.SND_FILENAME)

    #cProfile.run('test_hand_written(kernel=lambda x1, x2: gaussian(x1, x2, 7.5))', sort='time')

    #train_dir = 'data/Ch02/digits/trainingDigits'
    #train_xs, train_ys = load_digits(train_dir, skip_rate=0)
    #print("Shape of train_xs: ", train_xs.shape)
    #vecs, means = pca.pca(train_xs)
    #low_d_xs = pca.transform(train_xs, vecs, means)
    #print("Shape of train_xs after pca: ", low_d_xs.shape)