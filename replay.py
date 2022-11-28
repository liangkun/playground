#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
import random

class Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def add(self, sample, **kwargs):
        self.memory.append(sample)

    def sample(self, n):
        return random.sample(self.memory, n)

    def update(self, sample, **kwargs):
        raise Exception("unsupported operation")

class PriorityReplay(Replay):
    pass