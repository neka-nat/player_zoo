# -*- coding: utf-8 -*-
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import visdom
from libs import replay_memory, utils, optimizers, wrapped_env
vis = visdom.Visdom()

class ActorCritic(nn.Module):
    def __init__(self, n_action, input_shape=(84, 84, 3)):
        super(ActorCritic, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        r = int((int(input_shape[0] / 4) - 1) / 2) - 3
        c = int((int(input_shape[1] / 4) - 1) / 2) - 3
        self.lstm = nn.LSTMCell(r * c * 64, 512)
        self.pi = nn.Linear(512, self.n_action)
        self.v = nn.Linear(512, 1)
        self.reset(True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        self.hx, self.cx = self.lstm(x.view(x.size(0), -1), (self.hx, self.cx))
        logit = self.pi(self.hx)
        value = self.v(self.hx)
        return logit, value

    def act(self, x):
        logit, value = self.forward(x)
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(dim=1, keepdim=True)
        action = prob.multinomial(num_samples=1).detach()
        return action, value, log_prob.gather(1, action), entropy

    def reset(self, done=False):
        if done:
            self.cx = torch.zeros(1, 512)
            self.hx = torch.zeros(1, 512)
        else:
            self.cx = self.cx.detach()
            self.hx = self.hx.detach()

    def sync_grads(self, model):
        for p, op in zip(self.parameters(), model.parameters()):
            if op.grad is not None:
                return
            op._grad = p.grad

    def sync(self, model):
        self.load_state_dict(model.state_dict())

def train(global_model, optimizer, n_steps=20, gamma=0.99, tau=1.0,
          max_grad_norm=50.0, value_loss_coef=0.5, entropy_coef=0.01):
    env = gym.make('SingleFrameBreakout-v0')
    model = ActorCritic(env.action_space.n)
    model.train()
    state = env.reset()
    sum_rwd = 0.0
    n_episode = 0
    win1 = vis.image(utils.preprocess(env.env._get_image()))
    win2 = vis.line(X=np.array([0]), Y=np.array([0.0]),
                    opts=dict(title='Score'))

    for t in count():
        model.sync(global_model)
        buffer = []
        for _ in range(n_steps):
            action, value, log_prob, entropy = model.act(torch.from_numpy(state[0]).unsqueeze(0))
            state, reward, done, _ = env.step(action.item())
            buffer.append(utils.ActorCriticData(value, log_prob, reward, entropy))
            sum_rwd += reward
            vis.image(utils.preprocess(env.env._get_image()), win=win1)
            if done:
                vis.line(X=np.array([n_episode]), Y=np.array([sum_rwd]), win=win2, update='append')
                print("Episode: %d, Total Reward: %f" % (n_episode, sum_rwd))
                state = env.reset()
                sum_rwd = 0.0
                n_episode += 1
                break

        R = torch.zeros(1, 1)
        if not done:
            _, R, _, _ = model.act(torch.from_numpy(state[0]).unsqueeze(0))
            R = R.detach()

        buffer.append(utils.ActorCriticData(R, None, None, None))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(buffer) - 1)):
            R = gamma * R + buffer[i].reward
            advantage = R - buffer[i].value
            value_loss += 0.5 * advantage.pow(2)
            delta_t = buffer[i].reward + gamma * buffer[i + 1].value - buffer[i].value
            gae = gae * gamma * tau + delta_t
            policy_loss -= (buffer[i].log_prob * gae + entropy_coef * buffer[i].entropy)

        optimizer.zero_grad()
        (policy_loss + value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        model.sync_grads(global_model)
        optimizer.step()
        model.reset(done)

import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
    env = gym.make('SingleFrameBreakout-v0')
    global_model = ActorCritic(env.action_space.n)
    global_model.share_memory()
    optimizer = optimizers.SharedAdam(global_model.parameters(), lr=0.0001)
    processes = []
    for _ in range(4):
        p = mp.Process(target=train, args=(global_model, optimizer,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
