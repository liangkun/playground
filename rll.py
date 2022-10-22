#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import time
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    def __init__(self, env: gym.Env, options=None, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.model = nn.Sequential(
            nn.Linear(self.n_states, 32),
            nn.ReLU(),
            nn.Linear(32, 20),
            nn.ReLU(),
            nn.Linear(20, self.n_actions),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.options = options
    
    def forward(self, xs):
        xs = xs.to(self.device)
        return self.model(xs)

    def select_action(self, state):
        self.eval()
        state = torch.FloatTensor(state)
        probabilities = self(state).detach()
        probabilities = probabilities.cpu()
        c = Categorical(probabilities)
        action = c.sample()
        return action.item()
    
    def train_batch(self, states, rewards, actions, optimizer):
        self.train()
        optimizer.zero_grad()

        states = torch.FloatTensor(np.array(states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        # convert probabilities to log probabilities
        log_probs = torch.log(self(states))
        selected_log_probs = torch.gather(log_probs, dim=1, index=actions.unsqueeze(1)).squeeze(1)
        #print(selected_log_probs.cpu())
        #selected_log_probs = log_probs[np.arange(len(actions)), actions]
        # Loss is negative of expected policy function J = R * log_prob
        loss = -(rewards * selected_log_probs).mean()

        # Do the update gradient descent(with negative reward hence is gradient ascent) 
        loss.backward()
        optimizer.step()

        return loss.cpu().item()

class Agent:
    def __init__(self, env: gym.Env, policy: nn.Module, options=None):
        super().__init__()
        self.env = env
        self.policy = policy
        self.options = options
        self.gamma = options.gamma

    def discount_qs(self, rs):
        """compute discounted q(s, a) from reward of each step"""
        rs.reverse()
        result = [0] * len(rs) # reversed result

        result[0] = rs[0]
        i = 1
        while i < len(rs):
            result[i] = rs[i] + self.gamma * result[i-1]
            i += 1
        result.reverse()

        result = np.array(result)
        result = (result - result.mean()) / result.std()

        return result

    def train(self):
        opt = torch.optim.Adam(self.policy.parameters(), lr=self.options.lr)

        lr_steps = int(self.options.episodes / self.options.batchsize)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=int(lr_steps/3), gamma=0.2, verbose=True)
        total_rewards = []
        total_loss = []

        batch_states = []
        batch_actions = []
        batch_qs = []
        batch_count = 0
        batches = 0
        episode = 0
        start = time.time()
        while episode < self.options.episodes:
            s, _ = self.env.reset()
            done = False
            states = []
            actions = []
            rewards = []
            while (not done) and (len(states) < self.options.max_episode_length):
                action = self.policy.select_action(s)
                sn, r, done, _, _ = self.env.step(action)
                states.append(s)
                actions.append(action)
                rewards.append(r)
                s = sn

            if len(states) >= self.options.max_episode_length:
                continue  # ignore truncated trace
            
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_qs.extend(self.discount_qs(rewards))
            batch_count += 1
            total_rewards.append(np.sum(rewards))

            if episode % self.options.log_interval == 0:
                avg_reward = np.mean(total_rewards)
                total_rewards = []
                current = time.time()
                print(f"episode: {episode}, value(s): {avg_reward:.2f}, total time: {current-start:.0f}")

            if batch_count == self.options.batchsize:
                loss = self.policy.train_batch(batch_states, batch_qs, batch_actions, opt)
                batch_states = []
                batch_actions = []
                batch_qs = []
                batch_count = 0
                total_loss.append(loss)
                lr_scheduler.step()
                batches += 1

                if batches % 10 == 0:
                    avg_loss = np.mean(total_loss)
                    total_loss = []
                    print(f"last 10 batches avg loss: {avg_loss:.8f}")
            episode += 1

    def save(self, file):
        torch.save(self.policy, file)
    
    def load(self, file):
        self.policy = torch.load(file)
    
    def play(self):
        reward = 0
        s, _ = self.env.reset()
        done = False
        steps = 0
        while (not done) and (steps <= self.options.max_episode_length):
            self.env.render()
            action = self.policy.select_action(s)
            sn, r, done, _, _ = self.env.step(action)
            s = sn
            reward += r
            steps += 1
        return reward

def reinforce(options):
    render_mode = "human" if options.action == "play" else None
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    policy = Policy(env)
    agent = Agent(env, policy, options=options)

    if options.action == "train":
        if options.pretrain:
            agent.load(options.pretrain)

        agent.train()

        if options.policy:
            agent.save(options.policy)
    elif options.action == "play":
        if options.policy:
            agent.load(options.policy)
        else:
            print("WARN: play with random policy")
        reward = agent.play()
        print(f"total reward {reward}")
    else:
        raise Exception(f"unknown action {options.action} for reinforce")
    env.close()

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
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--max_episode_length", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--action", type=str, default="train", choices=["train", "play"])
    parser.add_argument("--policy", type=str, default=None, help="policy model file/path")
    parser.add_argument("--pretrain", type=str, default=None, help="policy model file/path")

    options = parser.parse_args(sys.argv[1:])

    if options.prog in REGISTRY:
        REGISTRY[options.prog](options)
    else:
        print(f"unknown prog: {options.prog}", file=sys.stderr)
