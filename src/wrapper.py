import numpy as np

import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper, spaces
from gymnasium.spaces import Box, Discrete
from collections import deque
from .utils import flatten, PredictorDataset, fine_tune_predictor, getDistance
from .safety_controller import safetyCheck
# from state_predictor import predictor_infer, load_predictor
import torch
class DelayWrapper(ObservationWrapper):
    def __init__(self, env, max_delay=10, mode=['only_delayed_state' 'delayed_state_and_action', 'delayed_state_and_delay', 'all'], drl_ormoc=False, delay_mode='uniform'):
        super().__init__(env)
        self.orig_shape = self.env.observation_space.shape
        orig_shape = self.env.observation_space.shape
        self.mode = mode
        self.delay_mode = delay_mode
        self.congested = False  # for bursty mode

        self.max_delay = max_delay
 
        # self.observation_space = spaces.Dict({
        #     "entities": self.env.observation_space,
        #     "global": spaces.Box(low=-5.0, high=5.0, shape=(self.global_dim,))
        # })
        if drl_ormoc:
            mode = 'only_delayed_state'
        else:
            mode = mode
        if self.mode == 'only_delayed_state':
            self.observation_space = self.env.observation_space
        elif self.mode == 'delayed_state_and_action':
            self.observation_space = Box(
                shape=(self.orig_shape[0] * self.orig_shape[1] + self.max_delay * self.env.action_space.shape[0],),
                low=-1000,
                high=1000,
                dtype=np.float32
            )
        elif self.mode == 'delayed_state_and_delay':
            self.observation_space = Box(
                shape=(self.orig_shape[0] * self.orig_shape[1] + 1,),
                low=-1000,
                high=1000,
                dtype=np.float32
            )
        elif self.mode == 'all':
            self.observation_space = Box(
                shape=(self.orig_shape[0] * self.orig_shape[1] + self.max_delay * self.env.action_space.shape[0] + 1,),
                low=-1000,
                high=1000,
                dtype=np.float32
            )
        self.max_delay = max_delay
        # Keep full-shaped entity observations in history: (num_entities, feature_dim)
        self.observation_history = deque([
            {
                'observation': np.zeros((self.orig_shape[0], self.orig_shape[1]), dtype=np.float32),
                'delay': 0
            }
            for _ in range(self.max_delay + 1)
        ], maxlen=self.max_delay + 1)
        # Keep action history as float32 vectors; avoid shared inner lists
        self.action_history = deque([
            np.zeros(self.env.action_space.shape[0], dtype=np.float32)
            for _ in range(self.max_delay)
        ], maxlen=self.max_delay)
        self.last_observation = None
        self.delay_of_received_observation = 0
        self.delay_of_last_observation = 0
        self.obs = None

    def reset(self, **kwargs):
        self.obs, info = self.env.reset(**kwargs)
        self.current_observation_delay = 0
        self.last_observation = self.obs
        self.delay_of_received_observation = 0
        self.delay_of_last_observation = 0
        self.congested = False  # reset bursty state
        self.observation_history = deque([
            {
                'observation': np.zeros((self.orig_shape[0], self.orig_shape[1]), dtype=np.float32),
                'delay': 0
            }
            for _ in range(self.max_delay + 1)
        ], maxlen=self.max_delay + 1)
        self.action_history = deque([
            np.zeros(self.env.action_space.shape[0], dtype=np.float32)
            for _ in range(self.max_delay)
        ], maxlen=self.max_delay)
        # self.newX = {
        #     "entities": self.obs,
        #     "global": flatten([[a.tolist() for a in self.action_history], [0]])
        # }
        self.newX = flatten([self.obs, list(self.action_history), [0]])
        if self.mode == 'only_delayed_state':
            return self.obs, info
        elif self.mode == 'delayed_state_and_action':
            return flatten([self.obs, list(self.action_history)]), info
        elif self.mode == 'delayed_state_and_delay':
            return flatten([self.obs, [0]]), info
        elif self.mode == 'all':
            return self.newX, info
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        # return flatten([self.obs, list(self.action_history), [0]]), info

    def _sample_delay(self):
        """Sample observation delay based on the configured delay_mode."""
        if self.delay_mode == "uniform":
            return np.random.randint(0, self.max_delay + 1)

        elif self.delay_mode == "exponential":
            # Favors low delays, long tail clipped to max_delay
            raw = np.random.exponential(scale=self.max_delay / 3.0)
            return int(np.clip(round(raw), 0, self.max_delay))

        elif self.delay_mode == "triangular":
            # Symmetric triangular distribution peaking at max_delay / 2
            mid = self.max_delay / 2.0
            return int(round(np.random.triangular(0, mid, self.max_delay)))

        elif self.delay_mode == "bursty":
            # Markov regime-switching: normal vs congested
            if self.congested:
                self.congested = np.random.random() < 0.9  # 90% stay congested
                return np.random.randint(self.max_delay // 2, self.max_delay + 1)
            else:
                self.congested = np.random.random() < 0.05  # 5% chance of burst
                return np.random.randint(0, self.max_delay // 4 + 1)

        elif self.delay_mode == "bimodal":
            # Mixture of good connection (low delay) and bad connection (high delay)
            if np.random.random() < 0.6:
                return np.random.randint(0, max(self.max_delay // 5, 1) + 1)
            else:
                return np.random.randint(self.max_delay * 3 // 5, self.max_delay + 1)

        else:
            raise ValueError(f"Unknown delay_mode: {self.delay_mode}")

    def observation(self, obs):
        self.obs = obs
        self.current_observation_delay = self._sample_delay()
        # self.current_observation_delay = 2
        # self.d = self.max_delay
        self.action_history.pop()
        self.action_history.appendleft(self.env.action)
        self.observation_history[-self.current_observation_delay-1] = {'observation': self.obs, 'delay': self.current_observation_delay}   
        received_obsWithDelay = self.observation_history.pop()
        received_obs = received_obsWithDelay['observation']
        self.delay_of_received_observation = received_obsWithDelay['delay']
        self.observation_history.appendleft({'observation': np.zeros(self.orig_shape[0]), 'delay': 0})
        # print('Received observation:', received_obs)
        # print('Received observation:', received_obs)
        first_if_statement = not received_obs.any()
        second_if_statement = self.delay_of_last_observation + 1 < self.delay_of_received_observation
        # print('First if statement:', first_if_statement)
        # print('Second if statement:', second_if_statement)
        if_statement = first_if_statement or second_if_statement
        if if_statement:
            received_obs = self.last_observation
            self.delay_of_received_observation = self.delay_of_last_observation + 1
        self.last_observation = received_obs
        self.delay_of_last_observation = self.delay_of_received_observation
        received_obs[0][:] = self.env.observation[0][:] #ego state is not affected by delay
        # Round values in newState to three decimal points 
        # self.newX = flatten([received_obs, list(self.action_history), [self.delay_of_received_observation]])
        # self.newX = {
        #     "entities": received_obs.astype(np.float32),
        #     "global": flatten([[a for a in self.action_history], [self.delay_of_received_observation]])
        # }
        masked_action_history = np.array(list(self.action_history)).copy()
        for i in range(self.delay_of_received_observation, len(masked_action_history)):
            masked_action_history[i] = np.zeros_like(masked_action_history[i])
        self.newX = flatten([received_obs, masked_action_history, [self.delay_of_received_observation]])
        if self.mode == 'only_delayed_state':
            return received_obs
        elif self.mode == 'delayed_state_and_action':
            return flatten([received_obs, masked_action_history])
        elif self.mode == 'delayed_state_and_delay':
            return flatten([received_obs, [self.delay_of_received_observation]])
        elif self.mode == 'all':
            return self.newX
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

class SafetyWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        action, safety_info = safetyCheck(action, self.env.observation, True, 'all')
        return action