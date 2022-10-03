#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import time

def choose_action(state, Q, epsilon=0):
    n_states, n_actions = Q.shape
    if random.uniform(0.0, 1.0) < epsilon:
        return random.randint(0, n_actions-1)
    else:
        return np.argmax(Q[state])

def train(env: gym.Env, alpha, gamma, epsilon=0.1, epoches=1000):
    """trainning at env, with params alpha, gamma"""
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for epoch in range(epoches):
        si, _ = env.reset()
        total_reward = 0
        total_error = 0
        terminated = False
        if epoch > 0.90 * epoches:
            real_eps = 0
        else:
            real_eps = epsilon * (1 - epoch/epoches)
        while not terminated:
            ai = choose_action(si, Q, real_eps)
            sn, ri, terminated, truncated, info = env.step(ai)
            Q[si, ai] = Q[si, ai] + alpha * (ri + gamma * np.max(Q[sn]) - Q[si, ai])
            si = sn
            total_reward += ri
            if ri == -10:
                total_error += 1
        if epoch % 1000 == 0:
            print(f"epoch {epoch} finished, epsilon {real_eps:.2f}, total reward: {total_reward}, total error: {total_error}")
    
    return Q

def test(env: gym.Env, Q, epoches=100):
    rewards = []
    errors = 0.0
    for epoch in range(epoches):
        si, _ = env.reset()
        terminated = False
        total_reward = 0
        while not terminated:
            ai = choose_action(si, Q, epsilon=0)
            sn, ri, terminated, truncated, info = env.step(ai)
            total_reward += ri
            si = sn
            if ri == -10:
                errors += 1
        rewards.append(total_reward)
    avg_rewards = np.average(rewards)
    avg_errors = errors/epoches
    print(f"avg reward: {avg_rewards}, avg errors: {avg_errors}")

env = gym.make("Taxi-v3")
gamma = 0.5
alpha = 0.05
epsilon = 0.2

Q = train(env, alpha, gamma, epsilon, epoches=100000)
test(env, Q)

def test_visual(Q):
    env = gym.make("Taxi-v3", render_mode="human")
    si, _ = env.reset()
    env.render()
    terminated = False
    while not terminated:
        time.sleep(1)
        ai = choose_action(si, Q)
        sn, ri, terminated, _, _ = env.step(ai)
        env.render()
        si = sn
        print(f"action: {ai}, reward: {ri}")

test_visual(Q)