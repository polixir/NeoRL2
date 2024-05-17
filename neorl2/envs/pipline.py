import math
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from .reward.pipeline_reward import get_reward
from .base import Env

def get_target_flow_rate(target_flow_rate=None, self=None):
    if self.np_random.random() < 0.003 or target_flow_rate is None:
        return self.np_random.choice([50,80,110,140])
    else:
        return target_flow_rate

class PipelineEnv(Env):
    def __init__(self, 
                 target_flow_rate=get_target_flow_rate, 
                 history_obs_fps=25, 
                 history_action_fps=25, 
                 seed=None):
        super(PipelineEnv, self).__init__() 
        self.channel_length = 100 
        self.max_flow_rate = 200 
        self.pipe_velocity = 5   
        
        self._target_flow_rate = target_flow_rate 
        self.target_flow_rate = None
        self.target_flow_rate_list = []
 
        self.history_obs_fps = history_obs_fps
        self.history_action_fps = history_action_fps
        
        self.max_history_fps = max(self.history_obs_fps, self.history_action_fps )
        
        obs_shape = 2
        obs_spaces_low = [0,0]
        obs_spaces_high = [self.max_flow_rate,self.max_flow_rate]
        
        if self.history_obs_fps > 0:
            obs_shape += self.history_obs_fps
            obs_spaces_low += [0 for _ in range(self.history_obs_fps)]
            obs_spaces_high += [self.max_flow_rate for _ in range(self.history_obs_fps)]
            
        if self.history_action_fps > 0:
            obs_shape += self.history_action_fps
            obs_spaces_low += [-1. for _ in range(self.history_action_fps)]
            obs_spaces_high += [1. for _ in range(self.history_action_fps)]
        
        self.observation_space = gym.spaces.Box(low=np.array(obs_spaces_low,dtype=np.float32), high=np.array(obs_spaces_high,dtype=np.float32)) 
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) 
        
        self.reset(seed)
        self.render_mode = None
    
    def get_target_flow_rate(self):
        if callable(self._target_flow_rate):
            self.target_flow_rate = self._target_flow_rate(self.target_flow_rate, self)
        else:
            self.target_flow_rate = self._target_flow_rate
            
        self.target_flow_rate_list.insert(0, self.target_flow_rate)
        
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._step += 1
        self.get_target_flow_rate()
    
        self.watergate_flow += action

        self.watergate_flow = np.clip(self.watergate_flow, 0, self.max_flow_rate)[0]
        self.water_flow_list = [self.watergate_flow,] + self.water_flow_list[:-1]
        
        self.full_water_flow_list.insert(0, self.watergate_flow)
        self.downstream_flow = self.water_flow_list[-1]
        
        self.action_hisoty_list.insert(0, action[0])
        self.obs_history_list.insert(0, self.downstream_flow )

        obs = [self.downstream_flow, self.target_flow_rate]
        
        history_obs = list(self.obs_history_list[1:self.history_obs_fps+1])
        history_action = list(self.action_hisoty_list[:self.history_action_fps])
        
        if self.history_obs_fps > 0:
            obs += history_obs
        if self.history_action_fps > 0:
            obs += history_action
        
        obs = np.array(obs).astype(np.float32)
        
        data = {
            "obs" :self.pre_observation,
            "action" : np.array([action,]).astype(np.float32),
            "next_obs" : obs,
            
        }
        reward = get_reward(data)
        
        self.pre_observation = obs
        
        return (obs, reward, False, False, {})
    
    def seed(self, seed=None):
        #if seed is None:
        #    return
        self.np_random, _ = gym.utils.seeding.np_random(seed)
    
    def reset(self, seed=None, options=None):
        self.seed(seed)
        self.reset_water_flow_list()
        self.target_flow_rate_list = []
        self.get_target_flow_rate()
        self._step = 0
        
        self.watergate_flow = self.water_flow_list[0]
        self.downstream_flow = self.water_flow_list[-1] 
        obs = [self.downstream_flow, self.target_flow_rate]
        
        if self.history_obs_fps > 0:
            obs += self.obs_history_list[1:self.history_obs_fps+1]
        if self.history_action_fps > 0:
            obs += self.action_hisoty_list[:self.history_action_fps]

        obs = np.array(obs).astype(np.float32)
        self.pre_observation = obs
            
        return obs, {}
        
    def reset_water_flow_list(self):
        init_history_fps = max(self.max_history_fps, math.ceil(self.channel_length//self.pipe_velocity)) + 1
        init_obs = self.np_random.uniform(80, 120)
        
        self.full_water_flow_list = list(np.ones([init_history_fps], dtype=np.float32) * init_obs)
        self.water_flow_list = list(np.ones([math.ceil(self.channel_length//self.pipe_velocity)], dtype=np.float32) * init_obs)
        
        self.obs_history_list = list(np.ones([init_history_fps], dtype=np.float32) * init_obs)
        self.action_hisoty_list = list(np.zeros([init_history_fps], dtype=np.float32))
    
    def render(self, mode='human'):
        fig, ax1 = plt.subplots(figsize=(10, 3))
        ax1.plot(range(len(self.target_flow_rate_list)), self.full_water_flow_list[:len(self.target_flow_rate_list)][::-1], label='Input Water Flow Rate')
        ax1.plot(range(len(self.target_flow_rate_list)), self.target_flow_rate_list[::-1], label='Target Flow Rate')
        ax1.plot(range(len(self.target_flow_rate_list)), self.obs_history_list[:len(self.target_flow_rate_list)][::-1], label='Output Flow Rate')
        ax2 = ax1.twinx()
        ax2.plot(range(len(self.target_flow_rate_list)), self.action_hisoty_list[:len(self.target_flow_rate_list)][::-1], label='Action', color='red')
        ax2.axhline(y=0, color='gray', linestyle='--', label="Zero line")
        ax2.set_ylim(-5, 5)
        ax2.set_ylabel('Action')
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper left')
        
        plt.show()

    def close(self):
        pass
    
    
if __name__ == "__main__":
    import neorl2
    import gymnasium as gym
    import numpy as np
    def expert_policy(obs, env):
        action =  obs[-1:] - obs[0:1]
        clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
        return clipped_action

    def random_policy(obs, env):
        return env.action_space.sample(obs)
    data = {
        "obs": [],
        "action": [],
        "next_obs": [],
        "reward": [],
        "done": [],
        "truncated" : [],
    }

    env_name = "Pipeline"
    env = gym.make(env_name)

    rewards = 0
    lengths = 0
    for i in range(100):
        obs,_ = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            action = expert_policy(obs,env)
            next_obs, reward, done, truncated, _ = env.step(action)
            data['obs'].append(obs)
            data['action'].append(action)
            data['next_obs'].append(next_obs)
            data['reward'].append(reward)
            data['done'].append(done)
            data['truncated'].append(truncated)
            rewards += reward
            lengths += 1
            obs = next_obs
            
    print(rewards)