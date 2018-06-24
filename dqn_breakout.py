# -*- coding: utf-8 -*-
import gym
import random
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import replay_memory
import utils
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.head = nn.Linear(22528, self.n_action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))

# This is based on the code from gym.
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

env = gym.make('Breakout-v0')
policy_net = DQN(env.action_space.n).to(device)
target_net = DQN(env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = replay_memory.ReplayMemory(10000)

def optimize_model(memory, batch_size):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = utils.Transition(*zip(*transitions))

    next_state_batch = torch.stack(batch.next_state)
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    done_batch = torch.stack(batch.done)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA * (1.0 - done_batch)) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

steps_done = 0
n_episodes = 50
preprocess = lambda s: np.ascontiguousarray(s.transpose((2, 0, 1)), dtype=np.float32) / 255
for n in range(n_episodes):
    # Initialize the environment and state
    state = preprocess(env.reset())
    for t in count():
        # Select and perform an action
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        action = utils.epsilon_greedy(torch.from_numpy(state).unsqueeze(0).to(device),
                                      policy_net, eps, device)
        next_state, reward, done, _ = env.step(action.item())
        next_state = preprocess(next_state)
        reward = torch.tensor([reward], device=device)
        done = torch.tensor([float(done)], device=device)
        memory.push(torch.from_numpy(state).to(device), action,
                    torch.from_numpy(next_state).to(device), reward, done)
        state = next_state.copy()
        env.render()

        # Perform one step of the optimization (on the target network)
        optimize_model(memory, BATCH_SIZE)
        steps_done += 1
        if done:
            break
    # Update the target network
    if n % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
