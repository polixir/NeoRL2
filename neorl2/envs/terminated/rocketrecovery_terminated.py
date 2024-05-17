import torch
import numpy as np


def get_terminated(data):    
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
        
    # Status: landing and flight
    condition_1 = array_type.abs(next_obs[..., 1:2]) <= 0.05
    
    done = array_type.where(condition_1, 1., 0.)
        
    if singel_sample:
        done = done[0]
        if array_type == np:
            done = done.item()

    return done