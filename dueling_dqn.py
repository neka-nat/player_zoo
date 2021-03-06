# -*- coding: utf-8 -*-
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import visdom
from libs import replay_memory, utils, wrapped_env
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

class DuelingDQN(nn.Module):
    def __init__(self, n_action, input_shape=(4, 84, 84)):
        super(DuelingDQN, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        r, c = utils.outsize(input_shape[1:], 8, 0, 4)
        r, c = utils.outsize((r, c), 4, 0, 2)
        r, c = utils.outsize((r, c), 3)
        self.adv1 = nn.Linear(r * c * 64, 512)
        self.adv2 = nn.Linear(512, self.n_action)
        self.val1 = nn.Linear(r * c * 64, 512)
        self.val2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)
        val = F.relu(self.val1(x))
        val = self.val2(val)
        return val + adv - adv.mean(1, keepdim=True)

BATCH_SIZE = 32
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE = 50

env = gym.make('MultiFrameBreakout-v0')
policy_net = DuelingDQN(env.action_space.n).to(device)
target_net = DuelingDQN(env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
memory = replay_memory.ReplayMemory(50000)

def optimize_model(memory, batch_size, gamma=0.999):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = utils.Transition(*zip(*transitions))

    next_state_batch = torch.stack(batch.next_state).to(device)
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.stack(batch.action).to(device)
    reward_batch = torch.stack(batch.reward).to(device)
    done_batch = torch.stack(batch.done).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma * (1.0 - done_batch)) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

steps_done = 0
n_episodes = 20000
win1 = vis.image(utils.preprocess(env.env._get_image()))
win2 = vis.image(env.reset())
win3 = vis.line(X=np.array([0]), Y=np.array([0.0]),
                opts=dict(title='Score'))
for n in range(n_episodes):
    # Initialize the environment and state
    state = env.reset()
    sum_rwd = 0
    for t in count():
        # Select and perform an action
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        action = utils.epsilon_greedy(torch.from_numpy(state).unsqueeze(0).to(device),
                                      policy_net, eps)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([min(max(reward, -1.0), 1.0)])
        done = torch.tensor([float(done)])
        memory.push(torch.from_numpy(state), action, reward,
                    torch.from_numpy(next_state), done)
        vis.image(utils.preprocess(env.env._get_image()), win=win1)
        vis.image(next_state, win=win2)
        state = next_state.copy()

        # Perform one step of the optimization (on the target network)
        optimize_model(memory, BATCH_SIZE)
        sum_rwd += reward.numpy()
        steps_done += 1
        if done:
            break
    print("Episode: %d, Total Reward: %f" % (n, sum_rwd))
    vis.line(X=np.array([n]), Y=np.array([sum_rwd]), win=win3, update='append')
    # Update the target network
    if n % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
