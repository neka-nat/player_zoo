import random
from collections import namedtuple
import numpy as np
from PIL import Image
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

ActorCriticData = namedtuple('ActorCriticData',
                             ('value', 'log_prob', 'reward', 'entropy'))

_outsize = lambda x, f, p, s: int((x - f + 2 * p) / s) + 1

def outsize(x, f, p=0, s=1):
    return (_outsize(x[0], f, p, s), _outsize(x[1], f, p, s))

def preprocess(img, shape=None, gray=False):
    pil_img = Image.fromarray(img)
    if not shape is None:
        pil_img = pil_img.resize(shape)
    if gray:
        img_ary = np.asarray(pil_img.convert("L"))
    else:
        img_ary = np.asarray(pil_img).transpose((2, 0, 1))
    return np.ascontiguousarray(img_ary, dtype=np.float32) / 255

def epsilon_greedy(state, policy_net, eps=0.1):
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1).cpu()
    else:
        return torch.tensor([random.randrange(policy_net.n_action)], dtype=torch.long)
