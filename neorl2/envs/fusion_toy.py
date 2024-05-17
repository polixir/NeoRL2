"""
Toy environment of tokamak to do BetaN tracking.

Author: Ian Char
Date: March 21, 2023
"""
from typing import Any, Dict, Tuple, Optional
import random

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# from gym import register


BETAN_MU = 1.7797029
BETAN_SIG = 1.2159256
PINJ_MU = 5817915.52734375
PINJ_SIG = 4223408.935546875
PINJ_ACT_SCALE = 0.25 * PINJ_SIG
TRANSITION_COEF = 2e2
BETA_OBS_COEF = 5.0
DT = 0.025


class ToyFusionEnv(gym.Env):
    """Toy fusion environment.

    Obvervation Space: (BetaN, Curr Power, Target) (and possibly the PID components).
    Action Space: Change in the beam powers.
    Reward: -abs(P)
    """

    def __init__(
        self,
        include_pid_in_obs: bool = True,
        w_start_dist: Tuple[float, float] = (5e4, 2.5e3),
        dw_start_dist: Tuple[float, float] = (0, 2.5e3),
        pinj_start_dist: Tuple[float, float] = (1.0e6, 1e5),
        aminor_dist: Tuple[float, float] = (0.589, 0), #(0.589, 0.02),
        bt_dist: Tuple[float, float] = (2.75, 0), #(2.75, 0.1),
        ip_dist: Tuple[float, float] = (1e6, 0), #(1e6, 1e5),
        betan_target_bounds: Tuple[float, float] = (1.5, 2.25),
        pinj_bounds: Tuple[float, float] = (5e5, 1.2e7),
        w_bounds: Tuple[float, float] = (0.0, 5e5),
        dw_bounds: Tuple[float, float] = (-8.5e5, 8.5e5),
        momentum: float = 0.5,
        obs_is_pid_only: bool = False,
        action_is_change: bool = True,
        seed: float = 42,
    ):
        """Constructor.

        Args:
            include_pid_in_obs: Whether the observation space should have PID
                components included.
            w_start_dist: Mean and std deviation for the W start distribution.
            pinj_start_dist: Mean and std deviation for the Pinj start distribution.
            aminor_dist: Mean and std deviation for aminor.
            bt_dist: Mean and std deviation for bt.
            ip_dist: Mean and std deviation for ip.
            betan_target_bounds: Bounds on the betan target.
            pinj_bounds: Bounds on how high power can be.
            w_bounds: Bounds on the stored energy.
            dw_bounds: Bounds on how fast the stored energy can change.
            momentum: Added momentum term in the dynamics.
            obs_is_pid_only: Whether the observations should only be pid.
            action_is_change: Whether the action is change in current power or is it
                the delta based on the midpoint of the power bounds.
        """
        super().__init__()
        self._include_pid_in_obs = include_pid_in_obs
        self._obs_is_pid_only = obs_is_pid_only
        self.observation_dim = 3 * (not self._obs_is_pid_only) + include_pid_in_obs * 3
       
        default_obs_dim = 3
        self.observation_space = gym.spaces.Box(
            -np.ones(default_obs_dim),
            np.ones(default_obs_dim))
        self._action_is_change = action_is_change
        self.action_space = gym.spaces.Box(-1 * np.ones(1), np.ones(1))

        self.task_dim = 3  # i.e. aminor, bt, and ip.
        self._w_start_dist = w_start_dist
        self._dw_start_dist = dw_start_dist
        self._pinj_start_dist = pinj_start_dist
        self._aminor_dist = aminor_dist
        self._bt_dist = bt_dist
        self._ip_dist = ip_dist
        self._betan_target_bounds = betan_target_bounds
        self._pinj_bounds = pinj_bounds
        self._w_bounds = w_bounds
        self._dw_bounds = dw_bounds
        self._momentum = momentum
        self.state = None
        self._dt = 0.025
        self.reset_task()
        self._max_episode_steps = 100
        self._error_accum = None
        self.t = 0
        self._seed = seed
        random.seed(self._seed)
    
    def seed(self, seed):
        self._seed = seed
        # random.seed(self._seed)

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
        self.reset_task(task)
        if targets is None:
            targets = self.sample_targets(1)[0]
        self.target = targets
        if start is None:
            start = self.sample_starts(1)[0]
        self.state = start
        self._error_accum = None
        full_obs = self._form_observation(self.state, self.target)

        return full_obs[[5,0,4]] #,{}

    def sample_starts(self, n_starts):
        start_ws = np.array([
            random.gauss(*self._w_start_dist)
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_dws = np.array([
            random.gauss(*self._dw_start_dist)
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_ps = np.array([
            np.clip(
                random.gauss(*self._pinj_start_dist),
                self._pinj_bounds[0],
                self._pinj_bounds[1]
            )
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        return np.concatenate([
            start_ws,
            start_dws,
            start_ps,
        ], axis=1)

    def sample_tasks(self, n_tasks):
        tasks = np.array([np.array([random.gauss(*self._aminor_dist),
                                    random.gauss(*self._bt_dist),
                                    random.gauss(*self._ip_dist)])
                          for _ in range(n_tasks)])
        return tasks

    def set_task(self, task):
        self._aminor = task[0]
        self._bt = task[1]
        self._ip = task[2]

    def get_task(self):
        return np.array([self._aminor, self._bt, self._ip])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

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
            pinj_change = np.clip(action, -1, 1) * PINJ_ACT_SCALE
            next_pinj = np.clip(
                self.state[2] + pinj_change,
                self._pinj_bounds[0],
                self._pinj_bounds[1]
            )
        else:
            midpoint = (self._pinj_bounds[1] - self._pinj_bounds[0]) / 2
            # next_pinj = np.clip(action, -1, 1) * midpoint + midpoint
            next_pinj = np.clip(
                np.clip(action, -1, 1) * midpoint + midpoint,
                self.state[2] - PINJ_ACT_SCALE,
                self.state[2] + PINJ_ACT_SCALE,
            )
        next_dw = np.clip((self._momentum * self.state[1] + (1 - self._momentum) * (next_pinj - TRANSITION_COEF * self.state[0] * self._ip ** -0.93 * self._bt ** -0.15 * self.state[2] ** 0.69)),
                          self._dw_bounds[0],
                          self._dw_bounds[1])
        next_w = np.clip(
            self.state[0] + self._dt * self.state[1],
            self._w_bounds[0],
            self._w_bounds[1],
        )
        next_state = np.array([next_w, next_dw.item(), next_pinj.item()])
        obs, rew = self._form_observation_and_rew(
            next_state, self.target, None, self.state)
        self.state = next_state

        if self.t >= self.max_episode_steps:
            terminate = True
        else:
            terminate = False

        # rew = -np.log(np.abs(obs[0] - self.target))
        return obs[[5,0,4]], rew,  terminate, {'target': self.target} #, False


    def sample_targets(self, num_target_trajs: int) -> np.ndarray:
        """Draw targets.

        Args:
            num_target_trajs: The number of target trajectories.

        Returns: The target ndarray w shape (num_target_trajs, max horizon).
        """
        return np.array([random.uniform(*self._betan_target_bounds)
                         for _ in range(num_target_trajs)])

    def _form_observation(self, state, target, task=None, last_state=None):
        """Form the observation."""
        return self._form_observation_and_rew(state, target, task, last_state)[0]

    def _form_observation_and_rew(self, state, target, task=None, last_state=None):
        """Form the observation and the reward."""
        if task is None:
            task = self.get_task().reshape(1, -1)
        if not isinstance(target, np.ndarray):
            target = np.array([target])
        # Calculate the betan.
        if len(state.shape) > 1:
            state = state
            original = True
        else:
            state = state.reshape(1, -1)
            original = False

        to_convert = [state]
        if last_state is not None:
            to_convert.append(last_state if len(last_state.shape) > 1
                              else last_state.reshape(1, -1))
        betans = [
            tc[:, 0] * task[:, 0] * task[:, 1] / task[:, 2] * BETA_OBS_COEF
            for tc in to_convert
        ]
        # Calculate the difference in the target.
        errs = betans[0] - target
        if self._error_accum is None:
            self._error_accum = errs.reshape(1, -1)
        else:
            self._error_accum = np.concatenate([
                self._error_accum,
                errs.reshape(1, -1)
            ], axis=0)
        # Calculate the I and D term.
        dterm = np.zeros(len(task)) if last_state is None else betans[0] - betans[1]
        dterm /= self._dt
        iterm = np.sum(self._error_accum, axis=0) * self._dt
        # Normalize terms in the observation and return.
        # if self._obs_is_pid_only:
        #     obs = np.concatenate([
        #         -errs.reshape(-1, 1) / BETAN_SIG,
        #         -iterm.reshape(-1, 1) / BETAN_SIG,
        #         -dterm.reshape(-1, 1) / BETAN_SIG,
        #     ], axis=1)
        # elif self._include_pid_in_obs:
        obs = np.concatenate([
            (betans[0].reshape(-1, 1) - BETAN_MU) / BETAN_SIG,
            -errs.reshape(-1, 1) / BETAN_SIG,
            -iterm.reshape(-1, 1) / BETAN_SIG,
            -dterm.reshape(-1, 1) / BETAN_SIG,
            (target.reshape(-1, 1) - BETAN_MU) / BETAN_SIG,
            (state[:, 2].reshape(-1, 1) - PINJ_MU) / PINJ_SIG,
        ], axis=1)
        # else:
        #     obs = np.concatenate([
        #         (betans[0].reshape(-1, 1) - BETAN_MU) / BETAN_SIG,
        #         (target.reshape(-1, 1) - BETAN_MU) / BETAN_SIG,
        #         (state[:, 2].reshape(-1, 1) - PINJ_MU) / PINJ_SIG,
        #     ], axis=1)
        if not original:
            obs = obs.reshape(-1)

        return obs, -np.log(np.abs(errs) / BETAN_SIG)

    def plot_paths(self, rollouts, colors=('blue', 'red', 'green', 'orange', 'purple'),
                   is_pid=False):
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 1)
        for cidx, obs in enumerate(rollouts['observations']):
            if is_pid:
                axs[0].plot(-obs[:, 0] * BETAN_SIG + rollouts['targets'][cidx],
                            color=colors[cidx],
                            alpha=0.8, label=np.sum(rollouts['rewards'][cidx]))
                axs[0].axhline(rollouts['targets'][cidx],
                               color=colors[cidx],
                               ls='--')
                axs[1].plot(rollouts['actuators'][cidx],
                            alpha=0.8, color=colors[cidx])
                axs[1].axhline((self._pinj_bounds[1] - self._pinj_bounds[0]) / 2
                               + self._pinj_bounds[0], ls=':', color='black')
            else:
                # Plot the Betan.
                axs[0].axhline(obs[0, -2] * BETAN_SIG + BETAN_MU,
                               color=colors[cidx], ls='--', alpha=0.6)
                axs[0].plot(obs[:, 0] * BETAN_SIG + BETAN_MU,
                            color=colors[cidx], alpha=0.8,
                            label=np.sum(rollouts['rewards'][cidx]))
                # Plot the powers.
                axs[1].plot(obs[:, -1] * PINJ_SIG + PINJ_MU,
                            color=colors[cidx], alpha=0.6)
        axs[0].legend()
        plt.tight_layout()
        plt.show()


# register(id='Fusion-v0', 
#          entry_point=ToyFusionEnv,
#          kwargs = dict(aminor_dist = (0.589, 0),
#                        bt_dist = (2.75, 0),
#                        ip_dist = (1e6, 0))
#         )