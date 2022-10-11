#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import time

def test_env(options):
    env = gym.make(options.env, render_mode="human")
    si, _ = env.reset()
    done = False
    while not done:
        env.render()
        sn, reward, done, _, _ = env.step(env.action_space.sample())
        si = sn
    env.close()

REGISTRY = {
    "test_env": test_env
}

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser("RLL")
    parser.add_argument("--prog", type=str, default="test_env")
    parser.add_argument("--env", type=str, default="Humanoid-v4")
    options = parser.parse_args(sys.argv[1:])

    if options.prog in REGISTRY:
        REGISTRY[options.prog](options)
    else:
        print(f"unknown prog: {options.prog}", file=sys.stderr)
