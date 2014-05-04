#!/usr/bin/env python3

import unittest as ut
from trees import *

class Test(ut.TestCase):
    def test_shannon_entropy(self):
        data_set = [(i, i) for i in range(1024)]
        self.assertEqual(10, calc_shannon_entropy(data_set))

