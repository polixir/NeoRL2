import numpy as np
import torch


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
    
    
    forward_reward_weight = 1.0 
    ctrl_cost_weight = 0.1
    dt = 0.05
    velocity_threshold = 3.2096
    
    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        array_type = torch
    
    
    ctrl_cost = ctrl_cost_weight * array_type.sum(array_type.square(action),axis=-1).reshape(-1,1)
    x_position_before = obs[...,0:1]
    x_position_after = next_obs[...,0:1]
    x_velocity = ((x_position_after - x_position_before) / dt)

    forward_reward = forward_reward_weight * x_velocity
    reward = forward_reward - ctrl_cost
    
    reward = array_type.where(x_velocity>velocity_threshold, -100., reward)
    if singel_sample:
        reward = reward[0]
        if array_type == np:
            reward = reward.item()

    return reward