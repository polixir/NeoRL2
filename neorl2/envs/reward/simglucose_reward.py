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

    
    if isinstance(obs, np.ndarray):
        array_type = np
    else:
        array_type = torch
        
    def get_risk(bg):
        fBG = 1.509 * (array_type.log(bg)**1.084 - 5.381)
        risk = fBG**2 * 10
        return risk
   
    bg = obs[...,:1]
    next_bg = next_obs[...,:1]
    risk_prev = get_risk(bg)
    risk_current = get_risk(next_bg)

    ori_rew = risk_prev - risk_current

    MAX_GLUCOSE = 600
    x_max, x_min = 0, -100 
    reward = ((ori_rew - x_min) / (x_max - x_min))  
    reward = array_type.where(next_bg <= 40, -15, reward)
    reward = array_type.where(next_bg >= MAX_GLUCOSE, 0, reward)

    if singel_sample:
        reward = reward[0]
        if array_type == np:
            reward = reward.item()

    return reward