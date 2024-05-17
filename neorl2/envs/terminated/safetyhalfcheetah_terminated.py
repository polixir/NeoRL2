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
    
    dt = 0.05
    velocity_threshold = 3.2096
    
    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        array_type = torch
    
    x_position_before = obs[...,0:1]
    x_position_after = next_obs[...,0:1]
    x_velocity = ((x_position_after - x_position_before) / dt)
    done = array_type.where(x_velocity>velocity_threshold, 1., 0.)

    if singel_sample:
        done = done[0]
        if array_type == np:
            done = done.item()

    return done