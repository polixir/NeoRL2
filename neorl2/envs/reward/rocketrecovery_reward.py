import torch
import numpy as np


def get_reward(data):    
    obs = data["obs"]
    action = data["action"]
    next_obs = data["next_obs"]
    singel_sample = False
    if len(obs.shape) == 1:
        obs = obs.reshape(1,-1)
        singel_sample = True
    if len(action.shape) == 1:
        action = action.reshape(1,-1)
    if len(next_obs.shape) == 1:
        next_obs = next_obs.reshape(1,-1)
    
    
    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        array_type = torch
        
    
    def get_shaping(state):
        shaping = (
            -100 * array_type.sqrt(state[...,0:1] * state[...,0:1] + state[...,1:2] * state[...,1:2])
            - 100 * array_type.sqrt(state[...,2:3] * state[...,2:3] + state[...,3:4] * state[...,3:4])
            - 100 * abs(state[...,4:5])
        )
        return shaping
    
    prev_shaping = get_shaping(obs)
    shaping = get_shaping(next_obs)
    reward = shaping - prev_shaping
    
    
    m_power = (array_type.clip(action[...,0:1], 0.0, 1.0) + 1.0) * 0.5 
    m_power = array_type.where(action[...,0:1]>0.0, m_power, 0)
    
    s_power = array_type.clip(array_type.abs(action[...,1:2]), 0.5, 1.0)
    s_power = array_type.where(array_type.abs(action[...,1:2]) > 0.5,s_power, 0)
    reward -= m_power * 0.3
    reward -= s_power * 0.03
        
    # Status: landing and flight
    condition_1 = array_type.abs(next_obs[..., 1:2]) <= 0.05
    # Landing: Is it a safe landing
    condition_2 = (array_type.abs(next_obs[..., 4:5]) < 0.3) & (array_type.abs(next_obs[..., 2:3]) < 0.5) & (array_type.abs(next_obs[..., 3:4]) < 2)
    
    landing_reward = array_type.where(condition_2, 100, -100)
    reward = array_type.where(condition_1, landing_reward, reward)
    
    # done = array_type.where(condition_1, True, False)
    
    if singel_sample:
        reward = reward[0]
        if array_type == np:
            reward = reward.item()

    return reward