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
        
    position = next_obs[...,:2]
    target = next_obs[...,-2:]

    err = position - target
    reward = array_type.sum(-array_type.log(array_type.abs(err)+0.01),axis=-1)
    
    if singel_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1,1)
    return reward