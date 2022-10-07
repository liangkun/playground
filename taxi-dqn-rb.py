#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, n_state, n_action, hidden=64):
        super(QNet, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden),
            nn.ReLU(),
        )
        self.v = nn.Linear(hidden, 1)
        self.a = nn.Linear(hidden, n_action)
    
    def forward(self, states):
        xs = F.one_hot(states, num_classes=self.n_state)
        hidden = self.net(xs.float())
        v = self.v(hidden)
        a = self.a(hidden)
        mean_a = torch.mean(a, dim=1, keepdim=True)
        out = v + (a - mean_a)
        return out

class DQN:
    def __init__(self, env:gym.Env, replay_capacity, gamma, target_update_freq=100, lr=1e-3, max_eposide_length=100):
        self.env = env
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n
        self.gamma = gamma
        self.qnet = QNet(self.n_state, self.n_action)
        self.target = QNet(self.n_state, self.n_action)
        self.target.load_state_dict(self.qnet.state_dict())
        self.replay = deque()
        self.replay_capacity = replay_capacity
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.lossfn = torch.nn.MSELoss()
        self.updates = 0
        self.target_update_freq = target_update_freq
        self.max_eposide_length = max_eposide_length
    
    def train(self, episodes, eps, batchsize):
        for episode in range(episodes):
            si, _ = self.env.reset()
            done = False
            total_loss = 0.0
            total_cnt = 0
            episode_length = 0
            episode_reward = 0.0
            trace = []
            if episode > episodes * 0.9:
                real_eps = 0.0
                real_max_length = 1000
            elif episode > episodes * 0.5:
                real_eps = eps * 0.1
                real_max_length = 500
            elif episode > episodes * 0.25:
                real_eps = eps * 0.2
                real_max_length = 200
            else:
                real_eps = eps
                real_max_length = self.max_eposide_length
            
            while not done:
                ai = self.choose_action(si, real_eps)
                sn, reward, done, _, _ = self.env.step(ai)
                trace.append((si, ai, reward, sn, done))
                episode_reward += reward
                si = sn
                episode_length += 1
                if episode_length >= self.max_eposide_length:
                    done = True
            
            if episode_length >= real_max_length:
                # useless samples, continue
                continue
            else:
                for elem in trace:
                    if len(self.replay) >= self.replay_capacity:
                        self.replay.popleft()
                    self.replay.append(elem)

                    if len(self.replay) >= batchsize:
                        batch_loss = self.train_batch(batchsize)
                        total_loss += batch_loss
                        total_cnt += 1
            
            if episode >= 0 and total_cnt > 0:
                loss = total_loss / total_cnt
                print(f"episode {episode}, length {episode_length}, avg loss: {loss}, reward: {episode_reward}")

    def train_batch(self, batchsize):
        self.qnet.train()
        batch = random.choices(self.replay, k=batchsize)
        sis, ais, ris, sns, dones = zip(*batch)
        
        targets = self.compute_target(ris, sns, dones)
        
        action_values = self.qnet(torch.tensor(sis))
        ais = torch.tensor(ais).unsqueeze(1)
        qvalues = torch.gather(action_values, dim=1, index=ais).squeeze(1)

        loss = self.lossfn(qvalues, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.updates += 1
        if self.updates >= self.target_update_freq:
            self.updates = 0
            self.target.load_state_dict(self.qnet.state_dict())

        return loss.item()

    @torch.no_grad()
    def test(self, episodes=100):
        total_reward = 0.0
        for _ in range(episodes):
            si, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            while not done:
                ai = self.choose_action(si, epsilon=0)
                sn, reward, done, _, _ = self.env.step(ai)
                #env.render()
                episode_reward += reward
                si = sn
                episode_length += 1
                if episode_length >= self.max_eposide_length:
                    done = True
            total_reward += episode_reward

        avg_reward = total_reward / episodes
        print(f"avg reward: {avg_reward}")

    def save(self, file):
        torch.save(self.qnet, file)
    
    def load(self, file):
        self.qnet = torch.load(file)

    @torch.no_grad()
    def choose_action(self, state, epsilon=0):
        if random.uniform(0.0, 1.0) < epsilon:
            return random.randint(0, self.n_action - 1)
        else:
            self.qnet.eval()
            states = torch.tensor([state], dtype=int)
            action_values = self.qnet(states)
            return torch.argmax(action_values[0]).item()

    @torch.no_grad()
    def compute_target(self, ris, sns, dones):
        sns = torch.tensor(sns)

        self.target.eval()
        action_values = self.target(sns)
        # ddqn, select s' action by qnet
        action_selections = torch.argmax(self.qnet(sns), dim=1).unsqueeze(1)
    
        # max_action_values, _ = torch.max(action_values, axis=1)
        max_action_values = torch.gather(action_values, dim=1, index=action_selections).squeeze(1)
        target = torch.tensor(ris) + max_action_values * (1.0 - torch.tensor(dones).float()) * self.gamma
        return target

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser("taxi dqn")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--model", default="taxi-qnet.pt", type=str)
    parser.add_argument("--visual", default=False, action="store_true")
    parser.add_argument("--episodes", default=100, type=int)
    options = parser.parse_args(sys.argv[1:])

    if options.visual:
        env = gym.make("Taxi-v3", render_mode="human")
    else:
        env = gym.make("Taxi-v3")
    model = DQN(env, replay_capacity=50000, gamma=0.95, target_update_freq=100, lr=1e-3)
    if options.test:
        model.load(options.model)
        model.test(episodes=options.episodes)
    else:
        model.train(episodes=20000, eps=0.8, batchsize=32)
        model.save(options.model)
        model.test(episodes=options.episodes)
    
    env.close()