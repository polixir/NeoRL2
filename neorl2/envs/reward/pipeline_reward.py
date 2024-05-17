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
        
    downstream_flow = next_obs[...,:1]
    target_flow_rate = next_obs[...,1:2]
    
    reward = array_type.square((200 - array_type.abs(downstream_flow - target_flow_rate)) * 0.01) - 3
    #reward = reward - (action*action) * 0.01
    
    if singel_sample:
        reward = reward[0]
        if array_type == np:
            reward = reward.item()

    return reward