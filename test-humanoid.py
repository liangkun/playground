#!/usr/bin/env
# -*- coding: utf-8 -*-

import gym
import time

env = gym.make("Humanoid-v4", render_mode="human")
si, _ = env.reset()
done = False
while not done:
    env.render()
    sn, reward, done, _, _ = env.step(env.action_space.sample())
    print(f"sn {sn}, reward {reward}")
    si = sn
env.close()