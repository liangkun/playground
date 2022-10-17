#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import time
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, env: gym.Env, options=None):
        super().__init__()
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.model = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, self.n_actions),
            nn.Softmax()
        )
        self.options = options
    
    def forward(self, xs):
        return self.model(xs)

    def select_action(self, state):
        states = torch.FloatTensor(state).unsqueeze(0)
        probabilities = self(states)[0]
        c = Categorical(probabilities)
        action = c.sample()
        log_prob = c.log_prob(action)

        return action, log_prob

class Agent:
    def __init__(self, env: gym.Env, policy: nn.Module, options=None):
        super().__init__()
        self.env = env
        self.policy = policy
        self.options = options
        self.gamma = options.gamma
    
    def rewards(self, rs):
        """compute discounted q(s, a) from reward of each step"""
        rs.reverse()
        result = [0] * len(rs) # reversed result

        result[0] = rs[0]
        i = 1
        while i < len(rs):
            result[i] = rs[i] + self.gamma * result[i-1]
            i += 1

        result.reverse()
        return result
    
    def train(self):
        self.policy.train()
        optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.options.lr)

        total_reward = 0
        total_loss = 0
        total = 0
        for episode in range(self.options.train_episodes + 1):
            s, _ = self.env.reset()
            done = False
            rs = []
            log_probs = []
            while (not done) and (len(rs) < self.options.max_episode_length):
                action, log_prob = self.policy.select_action(s)
                sn, r, done, _, _ = self.env.step(action.item())
                rs.append(r)
                log_probs.append(log_prob)
                s = sn
            
            qs = torch.tensor(self.rewards(rs))
            normalized_qs = (qs - qs.mean()) / (qs.std() + 1e-6)
            log_probs = torch.tensor(log_probs, requires_grad=True)
            loss = - (normalized_qs @ log_probs)
            loss = loss / len(normalized_qs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += qs[0]
            total_loss += loss
            total += 1

            if episode % options.log_interval == 0:
                print(f"episode: {episode}, value(s): {total_reward/total:.2f}, loss: {total_loss/total:.4f}")
                total_reward = 0
                total_loss = 0
                total = 0

            

def reinforce(options):
    env = gym.make("CartPole-v1")
    policy = Policy(env)
    agent = Agent(env, policy, options=options)
    agent.train()

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
    "test_env": test_env,
    "reinforce": reinforce
}

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser("RLL")
    parser.add_argument("--prog", type=str, default="test_env")
    parser.add_argument("--env", type=str, default="Humanoid-v4")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_episodes", type=int, default=100)
    parser.add_argument("--max_episode_length", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    options = parser.parse_args(sys.argv[1:])

    if options.prog in REGISTRY:
        REGISTRY[options.prog](options)
    else:
        print(f"unknown prog: {options.prog}", file=sys.stderr)
