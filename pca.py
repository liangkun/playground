#!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pp


def load_data_set(fname, delimt='\t'):
    with open(fname, 'r') as f:
        str_data = [line.strip().split(delimt) for line in f]
        data = [list(map(float, strs)) for strs in str_data]
        return np.array(data)


def replace_nan_with_mean(data):
    m, n = data.shape
    for i in range(n):
        nan_msk_i = np.isnan(data[:, i])
        mean_i = np.mean(data[np.bitwise_not(nan_msk_i), i])
        data[nan_msk_i, i] = mean_i


def pca(data, vrate=0.95, n=None):
    means = np.mean(data, axis=0)
    mn_data = data - means
    covarience = np.cov(mn_data, bias=1, rowvar=0)
    eig_vals, eig_vecs = la.eig(covarience)
    sorted_idxes = np.argsort(eig_vals)

    num_vecs = n
    if num_vecs is None:
        total_eig_vals = np.sum(eig_vals)
        partial_eig_vals = 0
        num_vecs = 0
        while partial_eig_vals / total_eig_vals < vrate:
            num_vecs += 1
            partial_eig_vals += eig_vals[sorted_idxes[-num_vecs]]

    pcs = eig_vecs[:, sorted_idxes[-num_vecs:]]

    return pcs, means


def transform(data, pcs, means):
    return (data - means).dot(pcs)


def reconstruct(data, pcs, means):
    return data.dot(pcs.T) + means


def plot_data(data, c='b', m='o'):
    pp.scatter(data[:, 0], data[:, 1], c=c, marker=m)


if __name__ == '__main__':
    data = load_data_set('data/Ch13/testSet.txt')
    pcs, means = pca(data, n=1)
    low_d_data = transform(data, pcs, means)
    recon_data = reconstruct(low_d_data, pcs, means)
    pp.scatter(data[:, 0], data[:, 1], c='b')
    pp.scatter(recon_data[:, 0], recon_data[:, 1], c='r', marker='x')
    pp.show()