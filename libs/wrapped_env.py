from collections import deque
import numpy as np
from gym.envs.atari.atari_env import AtariEnv
from libs import utils

class MultiFrameAtariEnv(AtariEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}
    no_op_steps = 30
    def __init__(self, game='pong', obs_type='image', frameskip=4, repeat_action_probability=0.):
        super(MultiFrameAtariEnv, self).__init__(game, obs_type, 1, repeat_action_probability)
        self._cur_st = None
        self._nx_st = None
        self._img_buf = deque(maxlen=frameskip)
        self._shape = (84, 84)
        self._initialize()

    def _initialize(self):
        self._nx_st = super(MultiFrameAtariEnv, self).reset()
        self._cur_st = self._nx_st.copy()
        for _ in range(self._img_buf.maxlen):
            self._img_buf.append(utils.preprocess(self._cur_st, self._shape, True))
        for _ in range(np.random.randint(1, self.no_op_steps) // self.frameskip):
            self.step(0)

    def step(self, a):
        reward = 0.0
        infos = {}
        for _ in range(self.frameskip):
            self._cur_st = self._nx_st.copy()
            self._nx_st, rwd, done, info = super(MultiFrameAtariEnv, self).step(a)
            nx_st = np.maximum(self._nx_st, self._cur_st)
            self._img_buf.append(utils.preprocess(nx_st, self._shape, True))
            reward += rwd
            infos.update(info)
        return np.array(list(self._img_buf)), reward, done, info

    def reset(self):
        self._img_buf.clear()
        self._initialize()
        return np.array(list(self._img_buf))

from gym.envs.registration import register

register(
    id='MultiFrameBreakout-v0',
    entry_point='libs.wrapped_env:MultiFrameAtariEnv',
    kwargs={'game': 'breakout', 'obs_type': 'image'},
    max_episode_steps=10000,
    nondeterministic=False,
)
