"""
Mass-spring-damper system environment.

Code based on: https://www.halvorsen.blog/documents/programming/python/resources
/powerpoints/Mass-Spring-Damper%20System%20with%20Python.pdf

Author: Ian Char
Date: September 5, 2022
"""
from collections import deque
from typing import Any, Dict, Tuple, Optional
import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# from gym import register

FULL_OBS = ('X', 'P', 'I', 'D', 'T', 'V', 'C', 'K', 'M', 'F')


class MassSpringDamperEnv(gym.Env):
    """Environment of the classic mass spring damper system.

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
        force_bounds: Tuple[float] = (-10, 10),
        reset_task_on_reset: bool = True,
        max_targets_per_ep: int = 1,
        action_is_change: bool = False,
        first_d_is_p: bool = False,
        i_horizon: Optional[int] = None,
        max_episode_steps: int = 100,
        exclude_current_positions_from_observation: bool = False,
        seed: float = 42,
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
        """
        super(MassSpringDamperEnv, self).__init__()
        assert len(observations) > 0 and len(observations) < 11

        default_obs_dim = 3
        self.observation_space = gym.spaces.Box(-np.ones(default_obs_dim),
                                                np.ones(default_obs_dim))
        self.action_space = gym.spaces.Box(-1 * np.ones(1), np.ones(1))

        self.observations = tuple([o.lower() for o in observations])
        

        self.max_targets_per_ep = max_targets_per_ep
        self._damping_bounds = damping_constant_bounds
        self._stiffness_bounds = spring_stiffness_bounds
        self._mass_bounds = mass_bounds
        self._target_bounds = target_bounds
        self._dt = dt
        self._force_bounds = force_bounds
        self._start_posn_bounds = (-0.25, 0.25)
        self._start_vel_bounds = (-0.1, 0.1)
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
        random.seed(seed)


    def reset(
        self,
        task: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
        seed=None
    ) -> np.ndarray:
        """Reset the system. If parameters are provided set them.

        Args:
            task: The task as [damping, stiffness, mass] with shape (3).
            targets: The target position to hit with shape (max_horizon)

        Returns:
            The observation.
        """
        self.t = 0
        self._last_act = 0.0
        self._err_hist = deque(maxlen=self._i_horizon)
        if self._reset_task_on_reset:
            self.reset_task(task)
        self._targets = targets
        if self._targets is None:
            self._targets = self.sample_targets(1)[0]
        self._dynamics_mat = np.array([
            [1, self._dt],
            [-self._dt * self._stiffness / self._mass,
             (1 - self._dt * self._damping / self._mass)]])
        if start is None:
            self.state = self.sample_starts(1)[0]
        else:
            self.state = start
        dterm = self.target - self.state[0] if self._first_d_is_p else 0
        full_obs = self._form_observation(np.array([self.state[0],
                                                    self.target - self.state[0],
                                                    -np.sum(self._err_hist) * self._dt,
                                                    dterm,
                                                    self.target,
                                                    self.state[1],
                                                    self._damping,
                                                    self._stiffness,
                                                    self._mass,
                                                    self._last_act,
                                                    ]))
        return full_obs[[0,5,4]],{}

    def sample_starts(self, n_starts):
        return np.concatenate([
            np.array([random.uniform(*self._start_posn_bounds)
                      for _ in range(n_starts)]).reshape(-1, 1),
            np.array([random.uniform(*self._start_vel_bounds)
                      for _ in range(n_starts)]).reshape(-1, 1),
        ], axis=1)

    def sample_tasks(self, n_tasks):
        tasks = np.array([np.array([random.uniform(*self._damping_bounds),
                                    random.uniform(*self._stiffness_bounds),
                                    random.uniform(*self._mass_bounds)])
                          for _ in range(n_tasks)])
        return tasks

    def set_task(self, task):
        self._damping = task[0]
        self._stiffness = task[1]
        self._mass = task[2]

    def get_task(self):
        return np.array([self._damping, self._stiffness, self._mass])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    @property
    def target(self):
        return self._targets[np.min([self.t, self._max_episode_steps - 1])]

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

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
            clipped_act = np.clip(float(action), -1, 1)
        self._last_act = clipped_act
        action = (clipped_act + 1) / 2
        action = action * (self._force_bounds[1] - self._force_bounds[0])
        action += self._force_bounds[0]
        prev_change = self.state[1]
        self.state = (self._dynamics_mat @ self.state.reshape(-1, 1)).flatten()
        self.state[1] += self._dt / self._mass * float(action)
        err = self.state[0] - self.target
        self._err_hist.append(err)
        obs = self._form_observation(np.array([
            self.state[0],
            -err,
            -np.sum(self._err_hist) * self._dt,
            -prev_change,
            self.target,
            self.state[1],
            self._damping,
            self._stiffness,
            self._mass,
            float(self._last_act),
        ]))

        if self.t > self.max_episode_steps:
            terminate = True
        else:
            terminate = False
        # return obs, -np.abs(err), False, {'target': self.target}

        reward = -np.log(np.abs(err)+0.01)

        return obs[[0,5,4]], reward, False, terminate, {'target': self.target}

    def sample_targets(self, num_target_trajs: int) -> np.ndarray:
        """Draw targets.

        Args:
            num_target_trajs: The number of target trajectories.

        Returns: The target ndarray w shape (num_target_trajs, max horizon).
        """
        targets_to_return = []
        for _ in range(num_target_trajs):
            num_targets = random.randint(1, self.max_targets_per_ep)
            targets = np.array([])
            for tnum in range(num_targets):
                if tnum == num_targets - 1:
                    targets = np.append(
                        targets,
                        np.ones(self._max_episode_steps - len(targets))
                        * random.uniform(*self._target_bounds),
                    )
                else:
                    targets = np.append(
                        targets,
                        np.ones(self._max_episode_steps // num_targets)
                        * random.uniform(*self._target_bounds),
                    )
            targets_to_return.append(targets)
        return np.array(targets_to_return)

    def _form_observation(self, full_obs):
        """Form observation based on what self.observations"""
        idx_list = [idx for idx, o in enumerate(FULL_OBS)
                    if o.lower() in self.observations]
        if len(full_obs.shape) == 1:
            return full_obs[idx_list]
        return full_obs[:, idx_list]



# register(id='MSD-v0', 
#          entry_point=MassSpringDamperEnv, 
#          kwargs = dict(damping_constant_bounds = (4.0, 4.0),
#                        spring_stiffness_bounds = (2.0, 2.0),
#                        mass_bounds = (20.0, 20.0))
#         )