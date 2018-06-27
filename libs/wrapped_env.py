from collections import deque
import numpy as np
from gym.envs.atari.atari_env import AtariEnv
from libs import utils

class MultiFrameAtariEnv(AtariEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, game='pong', obs_type='ram', frameskip=(2, 5), repeat_action_probability=0.):
        super(MultiFrameAtariEnv, self).__init__(game, obs_type, frameskip, repeat_action_probability)
        self._img_buf = deque(maxlen=4)
        self._shape = (84, 84)
        self._init_buf()

    def _init_buf(self):
        for _ in range(self._img_buf.maxlen):
            st = super(MultiFrameAtariEnv, self).reset()
            self._img_buf.append(utils.preprocess(st, self._shape, True))

    def step(self, a):
        nx_st, rwd, done, info = super(MultiFrameAtariEnv, self).step(a)
        self._img_buf.append(utils.preprocess(nx_st, self._shape, True))
        return np.array(list(self._img_buf)), rwd, done, info

    def reset(self):
        st = super(MultiFrameAtariEnv, self).reset()
        self._img_buf.clear()
        self._init_buf()
        return np.array(list(self._img_buf))

from gym.envs.registration import register

register(
    id='MultiFrameBreakout-v0',
    entry_point='libs.wrapped_env:MultiFrameAtariEnv',
    kwargs={'game': 'breakout', 'obs_type': 'image', 'repeat_action_probability': 0.25},
    max_episode_steps=10000,
    nondeterministic=False,
)
