# -*- coding: utf-8 -*-
import gym
import roboschool
import numpy as np
from itertools import count
from OpenGL import GLU

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import visdom
from libs import replay_memory, utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

def soft_update(dst, src, tau=0.001):
    for dp, sp in zip(dst.parameters(), src.parameters()):
        dp.data.copy_(dp.data * (1.0 - tau) + sp.data * tau)

class OUNoise:
    def __init__(self, n_action, mu=0, theta=0.15, sigma=0.2):
        self.n_action = n_action
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.n_action, dtype=np.float32) * self.mu

    def __call__(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state)).astype(np.float32)
        return self.state

class Actor(nn.Module):
    def __init__(self, n_action, n_state, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_state, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_action)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, n_action, n_state, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_state + n_action, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        s, a = x
        x = F.relu(self.fc1(torch.cat([s, a], 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000

env = gym.make("RoboschoolInvertedDoublePendulum-v1")
noise = OUNoise(env.action_space.shape[0])
actor = Actor(env.action_space.shape[0], env.observation_space.shape[0]).to(device)
target_actor = Actor(env.action_space.shape[0], env.observation_space.shape[0]).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
critic = Critic(env.action_space.shape[0], env.observation_space.shape[0]).to(device)
target_critic = Critic(env.action_space.shape[0], env.observation_space.shape[0]).to(device)
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

actor_optimizer = optim.Adam(actor.parameters(), lr=0.0001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

def optimize_model(memory, batch_size, criterion=nn.MSELoss(), gamma=0.999):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = utils.Transition(*zip(*transitions))

    next_state_batch = torch.stack(batch.next_state).to(device)
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.stack(batch.action).to(device)
    reward_batch = torch.stack(batch.reward).to(device)
    done_batch = torch.stack(batch.done).to(device)

    state_action_values = critic([state_batch, action_batch])
    next_state_action_values = target_critic([next_state_batch, target_actor(next_state_batch)]).detach()
    expected_state_action_values = (next_state_action_values * gamma * (1.0 - done_batch)) + reward_batch
    critic_loss = criterion(state_action_values, expected_state_action_values)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = -critic([state_batch, actor(state_batch)]).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(target_actor, actor)
    soft_update(target_critic, critic)

steps_done = 0
n_episodes = 20000
warmup = 1000
memory = replay_memory.ReplayMemory(50000)
win = vis.line(X=np.array([0]), Y=np.array([0.0]),
               opts=dict(title='Score'))
for n in range(n_episodes):
    # Initialize the environment and state
    state = env.reset().astype(np.float32)
    noise.reset()
    sum_rwd = 0
    for t in count():
        # Perform an action
        if steps_done < warmup:
            action = np.random.uniform(-1.0, 1.0, env.action_space.shape[0]).astype(np.float32)
        else:
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            action = actor(torch.tensor(state).to(device)).detach().cpu().numpy() + eps * noise()
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.astype(np.float32)
        env.render()
        reward = torch.tensor([reward])
        done = torch.tensor([float(done)])
        memory.push(torch.from_numpy(state), torch.from_numpy(action),
                    torch.from_numpy(next_state), reward, done)
        state = next_state.copy()

        # Perform one step of the optimization (on the target network)
        optimize_model(memory, BATCH_SIZE)
        sum_rwd += reward.numpy()
        steps_done += 1
        if done:
            break
    print("Episode: %d, Total Reward: %f" % (n, sum_rwd))
    vis.line(X=np.array([n]), Y=np.array([sum_rwd]), win=win, update='append')

print('Complete')
env.close()
