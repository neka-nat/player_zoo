# -*- coding: utf-8 -*-
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import visdom
from libs import utils, wrapped_env
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

class Actor(nn.Module):
    def __init__(self, n_action):
        super(Actor, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.n_action)
        self.n_params = [torch.numel(w) for w in self.parameters()]
        self.n_param = sum(self.n_params)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

    def set_weight(self, mean, std):
        pos = 0
        shapes = [w.detach().cpu().numpy().shape for w in self.parameters()]
        for w, s in zip(self.parameters(), shapes):	
            mean_t = torch.from_numpy(mean[pos:pos + np.prod(s)].reshape(s))
            std_t = torch.from_numpy(std[pos:pos + np.prod(s)].reshape(s))
            w.data.copy_(torch.normal(mean_t, std_t))
            pos += np.prod(s)

    def get_weights_flat(self):
        weights_flat = np.zeros(self.n_param)
        pos = 0
        for w, s in zip(self.parameters(), self.n_params):
            weights_flat[pos:pos + s] = w.detach().cpu().numpy().flatten()
            pos = pos + s
        return weights_flat


BATCH_SIZE = 64

env = gym.make('MultiFrameBreakout-v0')
actor = Actor(env.action_space.n).to(device)
win1 = vis.image(utils.preprocess(env.env._get_image()))
win2 = vis.line(X=np.array([0]), Y=np.array([0.0]),
                opts=dict(title='Score'))


def get_sample(env, actor, max_itr=1000):
    total_rwd = 0.0
    state = env.reset()
    for i in range(max_itr):
        a = actor(torch.from_numpy(state).unsqueeze(0).to(device)) 
        state, rwd, done, _ = env.step(a.max(1)[1].cpu().item())
        if i % 50 == 0:
            vis.image(utils.preprocess(env.env._get_image()), win=win1)
        total_rwd += rwd
        if done:
            break
    return total_rwd, actor.get_weights_flat()

steps_done = 0
n_episodes = 20000
def optimize_step(env, actor, mean, std, batch_size, n_best):
    total_rwds = []
    populations = []
    for _ in range(batch_size):
        actor.set_weight(mean = mean, std = std)
        total_rwd, population = get_sample(env, actor)
        total_rwds.append(total_rwd)
        populations.append(population)
    best_idx = np.argsort(total_rwds)[-n_best:]
    best = np.vstack([populations[i] for i in best_idx])
    mean_nxt = np.mean(best, axis=0)
    std_nxt = np.std(best, axis=0)
    return mean_nxt, std_nxt, total_rwds[best_idx[-1]]

mean, std = np.zeros(actor.n_param), np.ones(actor.n_param)
n_best = int(0.05 * BATCH_SIZE)
for n in range(n_episodes):
    mean, std, best_rwd = optimize_step(env, actor, mean, std, BATCH_SIZE, n_best)
    print("Episode: %d, Best Reward: %f" % (n, best_rwd))
    vis.line(X=np.array([n]), Y=np.array([best_rwd]), win=win2, update='append')

print('Complete')
env.close()
