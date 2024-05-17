import torch
import numpy as np


def get_reward(data):    

    obs = data["obs"]
    next_obs = data["next_obs"]
    action = data["action"]
    singel_reward = False

    if len(obs.shape) == 1:
        obs = obs.reshape(1,-1)
        singel_reward = True
    if len(next_obs.shape) == 1:
        next_obs = next_obs.reshape(1,-1)
    if len(action.shape) == 1:
        action = action.reshape(1,-1)
    
    if isinstance(obs, np.ndarray):
        array_type = np
        tofloat = lambda x : x.astype(np.float32)
        scale = np.array([[0.2, 0.5, 0.05]], dtype=np.float32)
    else:
        array_type = torch
        tofloat = lambda x : x.to(torch.float32)
        scale = torch.tensor([[0.2, 0.5, 0.05]]).to(torch.float32)

        

    error_scale = (next_obs[...,6:9] - obs[...,9:12])/scale

    rms = array_type.sqrt(array_type.mean(error_scale**2, axis=-1))
    reward = -array_type.log(rms)
    # reward = tofloat(reward)

    if singel_reward:
        reward = reward[0].item()
    
    return reward