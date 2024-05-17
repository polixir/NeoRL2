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

    z = next_obs[..., 1:2]
    angle = next_obs[..., 2:3]
    state = next_obs[..., 3:]

    min_state, max_state = (-100.0, 100.0)
    min_z, max_z = (0.7, float('inf'))
    min_angle, max_angle = (-0.2, 0.2)

    if array_type == np:
        healthy_state = array_type.all(array_type.logical_and(min_state < state, state < max_state), axis=-1,keepdims=True)
    else:
        healthy_state = array_type.all(array_type.logical_and(min_state < state, state < max_state), axis=-1,keepdim=True)
    healthy_z = array_type.logical_and(min_z < z, z < max_z)
    healthy_angle = array_type.logical_and(min_angle < angle, angle < max_angle)

    is_healthy = array_type.logical_and(array_type.logical_and(healthy_state, healthy_z), healthy_angle)

    done = array_type.logical_not(is_healthy) * 1.

    if singel_sample:
        done = done[0]
        if array_type == np:
            done = done.item()

    return done