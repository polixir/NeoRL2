"""
System with two weights and three springs.

Author: Ian Char
Date: January 8, 2022
"""
from collections import deque
from typing import Any, Dict, Tuple, Optional
# import random

import gymnasium as gym
# import gym
import numpy as np
import matplotlib.pyplot as plt
from .reward.dmsd_reward import get_reward
from .base import Env

# from gym import register


FULL_OBS = ('X', 'P', 'I', 'D', 'T', 'V', 'C', 'K', 'M', 'F')
OBS_MAP = {
    'X': (0, 2),
    'P': (2, 4),
    'I': (4, 6),
    'D': (6, 8),
    'T': (8, 10),
    'V': (10, 12),
    'C': (12, 15),
    'K': (15, 18),
    'M': (18, 20),
    'F': (20, 22),
}


class DoubleMassSpringDamperEnv(Env):
    """Environment of two masses back to back in series with a wall on either side.

    Observation Space: Maximum: (P, I, D) 3D
    Action Space: Force 1D
    Reward: -abs(P)
    """

    def __init__(
        self,
        observations: Tuple[str] = FULL_OBS,
        damping_constant_bounds: Tuple[float, float] = (4.0, 4.0),
        spring_stiffness_bounds: Tuple[float, float] = (2.0, 2.0),
        mass_bounds: Tuple[float, float] = (20.0, 20.0),
        target_bounds: Tuple[float, float] = (-1.5, 1.5),
        dt: float = 0.2,
        force_bounds: Tuple[float] = (-30, 30),
        reset_task_on_reset: bool = True,
        max_targets_per_ep: int = 2,
        action_is_change: bool = False,
        first_d_is_p: bool = False,
        i_horizon: Optional[int] = None,
        max_episode_steps: int = 100,
        seed: float = None,
        start_vel_bounds: Tuple[float] = (-0.1, 0.1),
        mode: str = 'train'
    ):
        """Constructor.

        Args:
            observations: The possible observations. The includes:
                X: The position.
                P: The Error.
                I: Integral of the error.
                D: The difference in the error.
                T: The target.
                V: The forward velocity.
                C: The damping constant.
                K: The spring constant.
                M: The mass constant.
                F: The current force setting.
            damping_constant_bounds: Bounds for drawing damping constant for the
            system.
            spring_stiffness_bounds: Bounds for drawing spring stiffnesses.
            mass_bounds: Bounds for drawing the mass.
            target_bounds: Bounds for the possible targets.
            dt: The discrete time step to take.
            force_bounds: Bounds on the force that can be applied.
            max_targets_per_ep: The maximum number of targets that can be in any
                one task.
            action_is_change: Whether the action should be the change in force.
                The force is in term of the standardized, clipped [-1, 1] amount.
            first_d_is_p: Whether the first difference should just be the error.
            i_horizon: How far the lookback should be for computing the I term.
            max_episode_steps: Length of an episode.
        """
        super(DoubleMassSpringDamperEnv, self).__init__()
        assert len(observations) > 0 and len(observations) < 11

        self.mode = mode
        self.obs_dim = np.sum([OBS_MAP[o.upper()][1] - OBS_MAP[o.upper()][0]
                               for o in observations])

        default_obs_dim = 6
        self.observation_space = gym.spaces.Box(-np.ones(default_obs_dim)*np.inf,
                                                np.ones(default_obs_dim)*np.inf)
        self.action_space = gym.spaces.Box(-1 * np.ones(2), np.ones(2))

        self.observations = tuple([o.lower() for o in observations])

        self.task_dim = 2 + 2 * 3
        self.max_targets_per_ep = max_targets_per_ep
        self._damping_bounds = damping_constant_bounds
        self._stiffness_bounds = spring_stiffness_bounds
        if self.mode == 'train':
            self._mass_bounds = mass_bounds
        else:
            self._mass_bounds = (30.0, 30.0)
        self._target_bounds = target_bounds
        self._dt = dt
        self._force_bounds = force_bounds
        self._start_posn_bounds = (-0.25, 0.25)
        self._start_vel_bounds = start_vel_bounds
        self._action_is_change = action_is_change
        self._first_d_is_p = first_d_is_p
        if i_horizon is not None:
            if i_horizon < 1:
                raise ValueError(f'Horizon must be at least 1, received {i_horizon}')
        self._i_horizon = i_horizon
        self._err_hist = deque(maxlen=self._i_horizon)
        self.state = None
        self.reset_task()
        self._dynamics_mat = None
        self._max_episode_steps = max_episode_steps
        self._reset_task_on_reset = reset_task_on_reset
        self.t = 0
        self.seed(seed)

    def seed(self,seed):
        if seed is None:
            self._seed = np.random.randint(1,100)
            self.np_random = np.random.RandomState()
            return
        self._seed = seed
        # random.seed(self._seed)
        # self.np_random, _ = gym.utils.seeding.np_random(self._seed)
        self.np_random = np.random.RandomState(seed=self._seed)

    @property
    def dynamics_mat(self):
        return self._dynamics_mat

    def reset(
        self,
        task: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
        seed=None,
        **kwargs
    ) -> np.ndarray:
        """Reset the system. If parameters are provided set them.

        Args:
            task: The task as [damping, stiffness, mass] with shape (3).
            targets: The target position to hit with shape (max_horizon)

        Returns:
            The observation.
        """
        self.seed(seed)
        self.t = 0
        self._last_act = np.zeros(2)
        self._err_hist = deque(maxlen=self._i_horizon)
        if self._reset_task_on_reset:
            self.reset_task(task)
        self._targets = targets
        if self._targets is None:
            self._targets = self.sample_targets(1)[0]
        self._dynamics_mat = np.array([
            [1, 0, self._dt, 0],  # First position update.
            [0, 1, 0, self._dt],  # Second position update.
            [-self._dt * (self._k1 + self._k2) / self._m1,  # First vel update.
             self._dt * self._k2 / self._m1,
             1 - self._dt * (self._d1 + self._d2) / self._m1,
             self._dt * self._d2 / self._m1],
            #  -self._dt * self._d2 / self._m1],
            [self._dt * self._k2 / self._m2,  # Seoncd vel update.
             -self._dt * (self._k2 + self._k3) / self._m2,
             self._dt * self._d2 / self._m2,
             1 - self._dt * (self._d2 + self._d3) / self._m2]])
        if start is None:
            self.state = self.sample_starts(1)[0]
        else:
            self.state = start
        dterm = self.target - self.state[:2] if self._first_d_is_p else np.zeros(2)
        full_obs = self._form_observation(np.concatenate([
            self.state[:2],
            self.target - self.state[:2],
            self.iterm,
            dterm,
            self.target,
            self.state[2:],
            self.get_task(),
            self._last_act,
        ]))
        self.obs = full_obs[[0,1,10,11,8,9]]
        return self.obs, {}
    
    def sample_starts(self, n_starts):
        starts = np.concatenate([
            np.array([self.np_random.uniform(*self._start_posn_bounds)
                      for _ in range(2 * n_starts)]).reshape(-1, 2),
            np.array([self.np_random.uniform(*self._start_vel_bounds)
                      for _ in range(2 * n_starts)]).reshape(-1, 2),
        ], axis=1)
        return starts

    def sample_tasks(self, n_tasks):
        tasks = np.array([np.array([
                self.np_random.uniform(*self._damping_bounds),
                self.np_random.uniform(*self._damping_bounds),
                self.np_random.uniform(*self._damping_bounds),
                self.np_random.uniform(*self._stiffness_bounds),
                self.np_random.uniform(*self._stiffness_bounds),
                self.np_random.uniform(*self._stiffness_bounds),
                self.np_random.uniform(*self._mass_bounds),
                self.np_random.uniform(*self._mass_bounds)])
            for _ in range(n_tasks)])
        return tasks

    def set_task(self, task):
        self._d1 = task[0]
        self._d2 = task[1]
        self._d3 = task[2]
        self._k1 = task[3]
        self._k2 = task[4]
        self._k3 = task[5]
        self._m1 = task[6]
        self._m2 = task[7]

    def get_task(self):
        return np.array([
            self._d1,
            self._d2,
            self._d3,
            self._k1,
            self._k2,
            self._k3,
            self._m1,
            self._m2,
        ])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    @property
    def target(self):
        return self._targets[np.min([self.t, self._max_episode_steps - 1])]
    
    def set_target(self, target):
        self._targets[np.min([self.t, self._max_episode_steps - 1])] = target

    @property
    def iterm(self) -> float:
        if len(self._err_hist):
            return -np.sum(np.array(self._err_hist), axis=0) * self._dt
        return np.zeros(2)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def set_obs(self,state=None, target=None):
        self.state = state
        self.set_target(target)


    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """"Take a step in the environment.

        Args:
            action: The force to applied.

        Returns:
            The next state, the reward, whether it is terminal and info dict.
        """
        self.t += 1
        if self._action_is_change:
            clipped_act = np.clip(self._last_act + action, -1, 1)
        else:
            clipped_act = np.clip(action, -1, 1)
        self._last_act = clipped_act
        action = (clipped_act + 1) / 2
        action = action * (self._force_bounds[1] - self._force_bounds[0])
        action += self._force_bounds[0]
        prev_change = self.state[2:]
        self.state = (self._dynamics_mat @ self.state.reshape(-1, 1)).flatten()
        self.state[2:] += self._dt * np.array([1 / self._m1, 1 / self._m2]) * action
        err = self.state[:2] - self.target
        self._err_hist.append(err)
        obs = self._form_observation(np.concatenate([
            self.state[:2],
            self.target - self.state[:2],
            self.iterm,
            -prev_change,
            self.target,
            self.state[2:],
            self.get_task(),
            self._last_act,
        ]))

        if self.t >= self.max_episode_steps:
            terminate = True
        else:
            terminate = False

        _next_obs = obs[[0,1,10,11,8,9]]
        _data = {
            'obs': self.obs.astype(np.float32),
            'action': action.astype(np.float32),
            "next_obs" : _next_obs.astype(np.float32),
                }
        reward = get_reward(_data)

        self.obs = _next_obs
        return self.obs, reward, False, False, {'target': self.target}  #False, terminate


    def sample_targets(self, num_target_trajs: int) -> np.ndarray:
        """Draw targets.

        Args:
            num_target_trajs: The number of target trajectories.

        Returns: The target ndarray w shape (num_target_trajs, 2, max horizon).
        """
        targets_to_return = []
        for _ in range(num_target_trajs):
            curr_target = []
            for _ in range(2):
                num_targets = self.np_random.randint(1, self.max_targets_per_ep+1)
                targets = np.array([])
                for tnum in range(num_targets):
                    if tnum == num_targets - 1:
                        targets = np.append(
                            targets,
                            np.ones(self._max_episode_steps - len(targets))
                            * self.np_random.uniform(*self._target_bounds),
                        )
                    else:
                        targets = np.append(
                            targets,
                            np.ones(self._max_episode_steps // num_targets)
                            * self.np_random.uniform(*self._target_bounds),
                        )
                curr_target.append(targets)
            targets_to_return.append(np.array(curr_target).T)
        return np.array(targets_to_return)

    def _form_observation(self, full_obs):
        """Form observation based on what self.observations"""
        idx_list = []
        for ob_name in self.observations:
            idx_list += [i for i in range(*OBS_MAP[ob_name.upper()])]
        if len(full_obs.shape) == 1:
            return full_obs[idx_list]
        return full_obs[:, idx_list]




# register(id='DMSD-v0', 
#          entry_point=DoubleMassSpringDamperEnv, 
#          kwargs = dict(damping_constant_bounds = (4.0, 4.0),
#                        spring_stiffness_bounds = (2.0, 2.0),
#                        mass_bounds = (20.0, 20.0))
#         )