#!/usr/bin/env python3

import numpy as np
import numpy.linalg as la


def load_data():
    return np.mat([[1, 1, 1, 0, 0],
                   [2, 2, 2, 0, 0],
                   [5, 5, 5, 0, 0],
                   [1, 1, 1, 0, 0],
                   [1, 1, 0, 2, 2],
                   [0, 0, 0, 3, 3],
                   [0, 0, 0, 1, 1]])


def eclud_sim(inA, inB):
    return 1 / (1 + la.norm(inA - inB))


def cosin_sim(inA, inB):
    return 0.5 + 0.5 * (inA.T * inB / (la.norm(inA) * la.norm(inB)))


if __name__ == '__main__':
    data = load_data()
    U, Sigma, Vt = la.svd(data)