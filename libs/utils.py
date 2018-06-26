import random
from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

preprocess = lambda s: np.ascontiguousarray(s.transpose((2, 0, 1)), dtype=np.float32) / 255

def epsilon_greedy(state, policy_net, eps=0.1):
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1).cpu()
    else:
        return torch.tensor([random.randrange(policy_net.n_action)], dtype=torch.long)
