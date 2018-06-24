import random
from collections import namedtuple
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

def epsilon_greedy(state, policy_net, eps=0.1, device='cpu'):
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1)
    else:
        return torch.tensor([random.randrange(policy_net.n_action)],
                            device=device, dtype=torch.long)
