import os
import numpy as np
import gymnasium
import gymnasium as gym

from datetime import datetime
from collections import deque, namedtuple
from gymnasium import spaces
from gymnasium.utils import seeding
from .data.simglucose.simulation.envNeoRL import T1DSimEnv as _T1DSimEnv
from .data.simglucose.patient.t1dpatient import T1DPatient
from .data.simglucose.sensor.cgm import CGMSensor
from .data.simglucose.actuator.pump import InsulinPump
from .data.simglucose.simulation.scenario_gen import RandomScenario
from .data.simglucose.controller.base import ctrller_action
from .base import Env

from .reward.simglucose_reward import get_reward
from .terminated.simglucose_terminated import get_terminated

PATIENT_PARA_FILE = os.path.join(os.path.dirname(__file__), 'data/simglucose/patient/vpatient_params.csv')
patient_property_names = ['BW', 'Fsnc', 'Ib', 'Km0', 'Vi', 'Vm0', 'Vmx', 'b', 'd', 'f', 'k1', 'k2', 
                         'ka1', 'ka2', 'kabs', 'kd', 'ke1', 'ke2', 'ki', 'kmax', 'kp1', 'kp2', 'kp3', 
                         'ksc', 'm1', 'm2', 'm30', 'm4', 'p2u', 'u2ss']
patient_property = namedtuple("patient_property", patient_property_names)

def hash_seed(seed: int) -> int:
    seed = int(seed)
    if seed >= 2**31:
        seed %= 2**31
    elif seed < 0:
        seed = (seed + 2**63) % 2**31
    return seed


class T1DSimEnvNeoRL(gym.Env):
    """
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    """

    metadata = {"render.modes": ["human"]}
    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(self, 
                 patient_name=None, 
                 custom_scenario=None, 
                 reward_fun=None, 
                 seed=None, 
                 stack_obs=1, 
                 mode='train',
    ):
        super(T1DSimEnvNeoRL, self).__init__()
        if mode == 'train':
            self.patient_names = ["adolescent#001", 
                                  "adolescent#002", 
                                  "adolescent#003",
                                  "adolescent#004", 
                                  "adolescent#005",
                                  "adult#001", 
                                  "adult#002", 
                                  "adult#003", 
                                  "adult#004", 
                                  "adult#005", 
                                  "child#001", 
                                  "child#002", 
                                  "child#003", 
                                  "child#004", 
                                  "child#005",]
        else:
            self.patient_names = ["adolescent#006",
                                  "adolescent#007", 
                                  "adolescent#008", 
                                  "adolescent#009",
                                  "adolescent#010", 
                                  "adult#006",
                                  "adult#007", 
                                  "adult#008", 
                                  "adult#009", 
                                  "adult#010",
                                  "child#006", 
                                  "child#007", 
                                  "child#008", 
                                  "child#009", 
                                  "child#010"]

        self.stack_obs = stack_obs
        self.obs = deque(maxlen=self.stack_obs)
        self.random_patient = False
        if patient_name is None:
            self.random_patient = True
            patient_name = np.random.choice(self.patient_names)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
        act = ctrller_action(basal=action, bolus=0)
        if self.reward_fun is None:
            obs, reward, done, info = self.env.step(act)
        else:
            obs, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
            
        self.obs.append(obs)
        obs = np.array(self.obs).reshape(-1, )
        
        truncated = True if self.env.time_count > 480 else False
        
        return np.concatenate((obs, self.patient_property)), reward, done, truncated, info

    def step(self, action:float):
        return self._step(action)

    # Not reload the patient, pump, sensor, scenario
    def _raw_reset(self):
        obs, reward, done, info = self.env.reset()
        self.obs.append(obs)
        if len(self.obs) < self.stack_obs:
            for _ in range(self.stack_obs - len(self.obs)):
                self.obs.append(obs)
        init_obs = np.array(self.obs).reshape(-1, )
        return np.concatenate((init_obs, self.patient_property)), reward, done, info

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        # print(self.env.patient._params['Name'])
        obs, reward, done, info = self.env.reset()
        self.obs.append(obs)
        if len(self.obs) < self.stack_obs:
            for _ in range(self.stack_obs - len(self.obs)):
                self.obs.append(obs)
        init_obs = np.array(self.obs).reshape(-1, )
        return np.concatenate((init_obs, self.patient_property)), reward, done, info

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = hash_seed(np.random.randint(0, 1000)) % 2**31
        seed3 = hash_seed(seed2 + 1) % 2**31
        seed4 = hash_seed(seed3 + 1) % 2**31

        # hour = 8
        hour = np.random.randint(low=0, high=24)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        if self.random_patient:
            # patient_name = self.np_random.choice(self.patient_name)
            self.patient_name = np.random.choice(self.patient_names)
            patient = T1DPatient.withName(self.patient_name, random_init_bg=False, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )
        # print(f"patient_name: {self.patient_name}")

        _patient_property = patient_property(** {k:patient._params.get(k) for k in patient_property_names})
        self.patient_property = np.array(_patient_property)
        
        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        self.env.render(close=close)

    def _close(self):
        super()._close()
        self.env._close_viewer()

    @property
    def action_space(self):
        ub = self.env.pump._params["max_basal"]
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(self.stack_obs + len(self.patient_property),))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]


class SimglucoseEnv(Env):
    metadata = {"render_modes": ["human"]}
    MAX_BG = 1000

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
        stack_obs=1,
        inclue_action=False,
        mode='train',
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnvNeoRL(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            stack_obs=stack_obs,
            mode=mode,
        )
        self.stack_obs = stack_obs
        self.patient_property = self.env.patient_property
        self.inclue_action = inclue_action
        patient_property_low =  [23, 1, 82, 185,0, 2, 0, 0, 0, 0, 0, 0,  0, 0, 0, 
                                 0, 0, 339, 0,0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        patient_property_high =  [112, 1, 139, 279,1, 15, 1, 1, 1, 1, 1, 1,1, 1, 
                                  2, 1, 1, 339, 1,1, 12, 1, 1, 1, 1, 1,1, 1, 1, 3]
        
        if not self.inclue_action:
            self.observation_space = gymnasium.spaces.Box(
                low=np.array([0]*self.stack_obs + patient_property_low, dtype=np.float32),
                high=np.array([self.MAX_BG]*self.stack_obs + patient_property_high, dtype=np.float32),
                shape=(self.stack_obs + len(self.patient_property),), 
                dtype=np.float32
            )
        
        else:
            self.observation_space = gymnasium.spaces.Box(
                low=np.array([0]*self.stack_obs + [0,] + patient_property_low, dtype=np.float32),
                high=np.array([self.MAX_BG]*self.stack_obs + [self.env.max_basal,] + patient_property_high, dtype=np.float32),
                shape=(self.stack_obs + 1 + len(self.patient_property),), 
                dtype=np.float32
            )
        # # self.env.max_basal = 30
        # self.action_space = gymnasium.spaces.Box(
        #     low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
        # )
        self.action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

    def step(self, action):
        action = np.clip(action, -1, 1)
        action = (action + 1) / 2 * 5
        obs, reward, done, truncated, info = self.env.step(action)

        if self.inclue_action:
            obs = np.insert(obs, 1, action)
            
        data = {
            "obs" :self.pre_observation,
            "action" : np.array([action,]).astype(np.float32),
            "next_obs" : obs,
        }
        reward = get_reward(data)
        self.pre_observation = obs
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, info = self.env._reset()
        if self.inclue_action:
            obs = np.insert(obs, 1, 0)
            
        self.pre_observation = obs
        return obs, info

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()