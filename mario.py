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
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT

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
    def __init__(self, env, shape, options):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = shape

        if options.fullcolor:
            channel = 3
        else:
            channel = 1
        obs_shape = (channel,) + self.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=float)

        transforms = [T.ToTensor(), T.Resize(self.shape)]
        if not options.fullcolor:
            transforms.append(T.Grayscale())
        self.transforms = T.Compose(transforms)
    
    def observation(self, observation):
        return self.transforms(observation.copy()).squeeze(0)

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim, options):
        super().__init__()
        self.c = input_dim[0]
        self.output_dim = output_dim
        self.backbone_dim = 18496

        if options.fullcolor:
            self.backbone = self.create_3d_net()
        else:
            self.backbone = self.create_grayscale_net()

        self.vnet = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.anet = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )
    
    def forward(self, xs):
        fc = self.backbone(xs)
        v = self.vnet(fc)
        a = self.anet(fc)
        amean = torch.mean(a, dim=1, keepdim=True)
        qvalues = v + (a - amean)
        return qvalues
    
    def get_target_net(self):
        target = copy.deepcopy(self)
        for p in target.parameters():
            p.requires_grad = False
        return target
    
    def create_grayscale_net(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=self.c, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
    
    def create_3d_net(self):
        return nn.Sequential(
            nn.Conv3d(in_channels=self.c, out_channels=32, kernel_size=(2, 8, 8), stride=(1, 4, 4)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(2, 4, 4), stride=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(2, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Flatten()
        )

class Mario:
    def __init__(self, state_dim, action_dim, options):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.options = options
        self.save_dir = options.savedir
        self.train = (options.action == "train")

        # prepare nets
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_net = MarioNet(self.state_dim, self.action_dim, options=options)
        self.policy_net = self.policy_net.to(self.device)
        self.exploration_rate = options.explore
        self.curr_step = 0
        self.gamma = 0.9

        if self.train:
            # training settings
            self.policy_net.train()
            self.target_net = self.policy_net.get_target_net()
            self.target_net.eval()
            self.exploration_rate_decay = 0.999999
            self.exploration_rate_min = 0.1
            self.burnin = 5000
            self.learn_every = 1
            self.sync_every = 10000

            self.memory = deque(maxlen=options.replaysize)
            self.batch_size = options.batchsize
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), options.lr)
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[250000, 500000, 1000000], gamma=0.25)
            self.lossfn = torch.nn.SmoothL1Loss()

            self.save_every = 1e5
        else:
            self.policy_net.eval()
    
    def state(self, obs):
        state = torch.tensor(np.array(obs), device=self.device)
        if self.options.fullcolor:
            state = state.permute(1, 0, 2, 3)  # (d, c, h, w) => (c, d, h, w) for 3D CNN
        return state

    @torch.no_grad()
    def choose_action(self, state):
        "given a state, choose an epislon greedy action"
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            self.policy_net.eval()
            state = state.unsqueeze(0)
            action_values = self.policy_net(state)
            self.policy_net.train()
            action_idx = torch.argmax(action_values[0]).item()
        
        if self.train and self.curr_step >= self.burnin:
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
        current_qs = self.policy_net(states)
        current_q = torch.gather(current_qs, dim=1, index=actions).squeeze(1)
        return current_q
    
    @torch.no_grad()
    def td_target(self, rewards, next_states, dones):
        self.policy_net.eval()
        next_states_q = self.policy_net(next_states)  # using policy net choose best action
        self.policy_net.train()
        best_actions = torch.argmax(next_states_q, axis=1)
        next_qs = self.target_net(next_states)  # using target net estimate next state q
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
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
    
    def save(self, best=False):
        if best:
            policy_path = os.path.join(self.save_dir, f"mario_best.chkpt")
        else:
            policy_path = os.path.join(self.save_dir, f"mario_{self.curr_step}.chkpt")
        torch.save(self.policy_net, policy_path)
    
    def load(self, path):
        self.policy_net = torch.load(path)
        if self.train:
            self.target_net = self.policy_net.get_target_net()

def create_mario_game(options):
    train = options.action == "train"

    if train:
        render_mode = "rgb"
    else:
        render_mode = "human"

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, ACTIONS)
    env = SkipFrame(env, interval=4)
    env = ObsTensor(env, shape=(OB_H, OB_W), options=options)
    env = FrameStack(env, num_stack=4)

    return env

def test(env, mario, options):
    for episode in range(options.episodes):
        obs, _ = env.reset()
        si = mario.state(obs)
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

        print(f"episode: {episode}, steps: {steps}, reward: {episode_reward}")

def train(env, mario, options):
    best_rewards = -1.0
    episode_rewards = []
    episode_steps = []
    for episode in range(options.episodes):
        obs, info = env.reset()
        si = mario.state(obs)
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

        rmean = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else -1.0
        smean = np.mean(episode_steps[-10:]) if len(episode_steps) >= 10 else -1.0
        qmean = np.mean(qs[-10:]) if len(qs) >= 10 else -1.0
        lossmean = np.mean(losses[-10:]) if len(losses) >= 10 else -1.0

        if rmean > best_rewards:
            best_rewards = rmean
            mario.save(best=True)

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
    parser.add_argument("--savedir", type=str, default="mario")
    parser.add_argument("--explore", type=float, default=0.05)
    parser.add_argument("--replaysize", type=int, default=10000)
    parser.add_argument("--fullcolor", action="store_true", default=False)
    options = parser.parse_args(sys.argv[1:])

    if options.fullcolor:
        state_dim=(3, 4, OB_H, OB_W)
    else:
        state_dim=(4, OB_H, OB_W)

    env = create_mario_game(options)
    mario = Mario(state_dim=state_dim, action_dim=len(ACTIONS), options=options)
    summary(mario.policy_net, state_dim, device=mario.device)

    if options.modelfile:
        mario.load(options.modelfile)

    if options.action == "train":
        train(env, mario, options)
    elif options.action == "test":
        test(env, mario, options)
    else:
        print(f"unknown option action={options.action}")
        sys.exit(1)
