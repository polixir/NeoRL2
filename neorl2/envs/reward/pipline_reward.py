import torch
import numpy as np


def get_reward(data):    
    obs = data["obs"]
    action = data["action"]
    next_obs = data["next_obs"]
    singel_reward = False
    if len(obs.shape) == 1:
        obs = obs.reshape(1,-1)
        singel_reward = True
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
    
    reward = array_type.square((200 - array_type.abs(downstream_flow - target_flow_rate)) * 0.01)
    reward = reward - array_type.sum(array_type.square(action)) * 0.01
    
    if singel_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1,1)
    return reward