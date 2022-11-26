#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy, time
import matplotlib.pyplot as plt
from torchsummary import summary

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY

OB_H = 168
OB_W = 168
#ACTIONS = RIGHT_ONLY
ACTIONS = [
    ['NOOP'],
    ['right'],
    ['right', 'A']
]

class SkipFrame(gym.Wrapper):
    """return only one frame in every interval frames"""
    def __init__(self, env, interval):
        super().__init__(env)
        self.interval = interval
    
    def step(self, action):
        total_reward = 0.0
        for i in range(self.interval):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info

class ObsTensor(gym.ObservationWrapper):
    """transform observation into torch tensor of specified shape"""
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = shape

        obs_shape = (1,) + self.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=float)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(self.shape),
            T.Grayscale()
        ])
    
    def observation(self, observation):
        return self.transforms(observation.copy()).squeeze(0)

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.c, _, _ = input_dim
        self.output_dim = output_dim

        self.policy = nn.Sequential(
            nn.Conv2d(in_channels=self.c, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(18496, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )
        self.policy.train()

        self.target = copy.deepcopy(self.policy)
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()
    
    def forward(self, xs, model="policy"):
        if model == "policy":
            return self.policy(xs)
        elif model == "target":
            return self.target(xs)
        else:
            raise Exception(f"unknown model: {model}")

class MarioStopper:
    def __init__(self, xstopper):
        self.last_x_pos = -1
        self.last_x_pos_cnt = 0
        self.xstopper = xstopper
    
    def stopped(self, info):
        x_pos = info["x_pos"]

        if x_pos == self.last_x_pos:
            self.last_x_pos_cnt += 1
        else:
            self.last_x_pos = x_pos
            self.last_x_pos_cnt = 1
        
        return self.last_x_pos_cnt > self.xstopper

    def reset(self, xstopper=None):
        self.last_x_pos = -1
        self.last_x_pos_cnt = 0
        if xstopper:
            self.xstopper = xstopper

class Mario:
    def __init__(self, state_dim, action_dim, save_dir, train=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qnet = MarioNet(self.state_dim, self.action_dim)
        self.qnet = self.qnet.to(self.device)

        # training settings
        self.train = train
        self.exploration_rate = 0.3
        self.exploration_rate_decay = 0.9999997
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.burnin = 10000
        self.learn_every = 1
        self.sync_every = 10000

        self.memory = deque(maxlen=30000)
        self.batch_size = 32
        self.gamma = 0.95
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100000, 500000], gamma=0.25)
        self.lossfn = torch.nn.MSELoss()

        self.save_every = 1e5
    
    def state(self, obs):
        state = torch.tensor(np.array(obs), device=self.device)
        # state = state.permute(1, 0, 2, 3)  # (d, c, h, w) => (c, d, h, w)
        return state

    @torch.no_grad()
    def choose_action(self, state):
        "given a state, choose an epislon greedy action"
        if np.random.rand() < self.exploration_rate and self.train:
            action_idx = np.random.randint(self.action_dim)
        else:
            self.qnet.policy.eval()
            state = state.unsqueeze(0)
            action_values = self.qnet(state, model="policy")
            self.qnet.policy.train()
            action_idx = torch.argmax(action_values[0]).item()
        
        if self.curr_step >= self.burnin:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

        self.curr_step += 1

        return action_idx

    def cache(self, experience):
        "remember the experience for learning"
        state, action, next_state, reward, done = experience
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device).float()
        done = torch.tensor([done], device=self.device).float()
        self.memory.append((state, action, next_state, reward, done))

    def recall(self):
        "sample experiences from memory"
        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, dones = map(torch.stack, zip(*batch))
        return states, actions.squeeze(0), next_states, rewards.squeeze(0), dones.squeeze(0)

    def learn(self):
        "Update policy action value"
        if self.curr_step % self.sync_every == 0:
            self.sync_target_net()
        
        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        states, actions, next_states, rewards, dones = self.recall()
        td_tgt = self.td_target(rewards, next_states, dones)
        td_est = self.td_estimate(states, actions)
        loss = self.update_policy_net(td_est, td_tgt)

        return td_est.mean().item(), loss

    def td_estimate(self, states, actions):
        current_qs = self.qnet(states, model="policy")
        current_q = torch.gather(current_qs, dim=1, index=actions).squeeze(1)
        return current_q
    
    @torch.no_grad()
    def td_target(self, rewards, next_states, dones):
        self.qnet.policy.eval()
        next_states_q = self.qnet(next_states, model="policy")  # using policy net choose best action
        self.qnet.policy.train()
        best_actions = torch.argmax(next_states_q, axis=1)
        next_qs = self.qnet(next_states, model="target")  # using target net estimate next state q
        next_q = torch.gather(next_qs, dim=1, index=best_actions.unsqueeze(1)).squeeze(1)
        target = rewards.squeeze(1) + self.gamma * (1 - dones.squeeze(1)) * next_q
        return target
    
    def update_policy_net(self, td_estimte, td_target):
        loss = self.lossfn(td_estimte, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()
    
    def sync_target_net(self):
        self.qnet.target.load_state_dict(self.qnet.policy.state_dict())
        self.qnet.target.eval()
    
    def save(self):
        save_path = os.path.join(self.save_dir, f"mario_net_{self.curr_step}.chkpt")
        torch.save(self.qnet, save_path)
    
    def load(self, path):
        self.qnet = torch.load(path)

def create_mario_game(train=True):
    if train:
        render_mode = "rgb"
    else:
        render_mode = "human"

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, ACTIONS)
    env = SkipFrame(env, interval=4)
    env = ObsTensor(env, shape=(OB_H, OB_W))
    env = FrameStack(env, num_stack=4)

    return env

def test(env, mario, options):
    #stopper = MarioStopper(50)
    for episode in range(options.episodes):
        obs, _ = env.reset()
        si = mario.state(obs)
        #stopper.reset(100)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action = mario.choose_action(si)
            next_obs, reward, done, trunc, info = env.step(action)
            sn = mario.state(next_obs)
            si = sn
            episode_reward += reward
            steps += 1
            time.sleep(0.05)
            #if stopper.stopped(info):
            #    break

        print(f"episode: {episode}, steps: {steps}, reward: {episode_reward}")

def train(env, mario, options):
    #stopper = MarioStopper(50)
    episode_rewards = []
    episode_steps = []
    for episode in range(options.episodes):
        obs, info = env.reset()
        si = mario.state(obs)
        #if episode > 5000:
        #    stopper.reset(1000)
        #else:
        #    stopper.reset(50)
        done = False

        qs = []
        losses = []
        episode_rewards.append(0.0)
        episode_steps.append(0.0)

        while not done:
            action = mario.choose_action(si)
            next_obs, reward, done, _, info = env.step(action)
            sn = mario.state(next_obs)
            mario.cache((si, action, sn, reward, done))
            q, loss = mario.learn()
            si = sn

            if q: qs.append(q)
            if loss: losses.append(loss)
            episode_rewards[-1] += reward
            episode_steps[-1] += 1
            #if stopper.stopped(info):
            #    break

        rmean = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else -1.0
        smean = np.mean(episode_steps[-10:]) if len(episode_steps) >= 10 else -1.0
        qmean = np.mean(qs[-10:]) if len(qs) >= 10 else -1.0
        lossmean = np.mean(losses[-10:]) if len(losses) >= 10 else -1.0
        lr = mario.lr_scheduler.get_last_lr()[0]
        explore = mario.exploration_rate
        print(f"episode: {episode}[{mario.curr_step}], steps: {smean:.0f}, reward: {rmean:.0f}, q: {qmean:.2f}, loss: {lossmean:.2f}, explore: {explore:.2f}, lr: {lr}")

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser("robot mario")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--max_episode_length", type=int, default=1000)
    parser.add_argument("--action", type=str, default="test")
    parser.add_argument("--modelfile", type=str, default="")

    options = parser.parse_args(sys.argv[1:])
    if options.action == "train":
        env = create_mario_game(train=True)
        mario = Mario(state_dim=(4,OB_H,OB_W), action_dim=len(ACTIONS), save_dir="mario", train=True)
        if options.modelfile:
            mario.load(options.modelfile)
        summary(mario.qnet, (4,OB_H,OB_W), device=mario.device)
        train(env, mario, options)
    elif options.action == "test":
        env = create_mario_game(train=False)
        mario = Mario(state_dim=(4,OB_H,OB_W), action_dim=len(ACTIONS), save_dir="mario", train=False)
        if options.modelfile:
            mario.load(options.modelfile)
        test(env, mario, options)
    else:
        print(f"unknown option action={options.action}")
        sys.exit(1)

