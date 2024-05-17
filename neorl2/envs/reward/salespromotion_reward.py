import torch
import numpy as np

def get_reward(data):    
    obs = data["obs"]
    next_obs = data["next_obs"]
    action = data["action"]
    singel_reward = False
    bonus_weight = 10 

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
    
    # if (action[..., 0].min() >=-1 and action[..., 0].max() < 1+1e-6) and ((1e-6 < action[...,0]) & (action[...,0] < 1.0)).sum()>0:  
    #     action[..., 0] *= action_coupon_num_max
    # action[...,0] = action[...,0].round()
    # action[...,0] = array_type.clip(action[...,0], 0, action_coupon_num_max)
    # action_cost = action[...,0] * coupon_cost
    # gived_coupon_num = obs[...,-1]
    # continue_give = (gived_coupon_num + action[:,0]) <= max_give_coupon_num
    #强行对发券量进行限制不超过总发券最大值
    # action[...,0] = action[...,0] * continue_give + (~continue_give) * (action[...,0] - (gived_coupon_num + action[...,0]-max_give_coupon_num ))

    d_total_cost, d_total_gmv = next_obs[..., 4:5], next_obs[..., 5:6]
    bonus = next_obs[..., 7:8] * bonus_weight
    reward = (d_total_gmv - d_total_cost) + bonus 

    if singel_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1,1)

    return reward, bonus


def get_reward_val(data):    
    obs = data["obs"]
    next_obs = data["next_obs"]
    action = data["action"]
    singel_reward = False
    bonus_weight = 10 #0 

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
    
    # if (action[..., 0].min() >=-1 and action[..., 0].max() < 1+1e-6) and ((1e-6 < action[...,0]) & (action[...,0] < 1.0)).sum()>0:  
    #     action[..., 0] *= action_coupon_num_max
    # action[...,0] = action[...,0].round()
    # action[...,0] = array_type.clip(action[...,0], 0, action_coupon_num_max)
    # action_cost = action[...,0] * coupon_cost
    # gived_coupon_num = obs[...,-1]
    # continue_give = (gived_coupon_num + action[:,0]) <= max_give_coupon_num
    #强行对发券量进行限制不超过总发券最大值
    # action[...,0] = action[...,0] * continue_give + (~continue_give) * (action[...,0] - (gived_coupon_num + action[...,0]-max_give_coupon_num ))

    d_total_cost, d_total_gmv = next_obs[..., 4:5], next_obs[..., 5:6]
    bonus = next_obs[..., 7:8] * bonus_weight
    reward = (d_total_gmv - d_total_cost) + bonus 

    if singel_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1,1)

    return reward, bonus