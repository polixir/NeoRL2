import time
import neorl2
import numpy as np
import gymnasium as gym


def test_gym_environment(env_name, nums=1000):
    t1 = time.time()
    env = gym.make(env_name)
    env = neorl2.make(env_name)

    total_reward = 0
    total_length = 0

    for i in range(nums):
        state, _ = env.reset(seed=i)
        done = False
        truncated = False
        while not done and not truncated:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_length += 1
            state = next_state
    print(f"Env: {env_name},Total reward: {total_reward}, Mean reward: {total_reward/nums}, Total length: {total_length}, Score: {env.get_normalized_score(total_reward/nums)}")
    train_data,val_data = env.get_dataset()
    
    train_nums = np.logical_or( np.bool_(train_data["done"]),np.bool_(train_data["truncated"])).sum()
    val_nums = np.logical_or( np.bool_(val_data["done"]),np.bool_(val_data["truncated"])).sum()
    
    train_data_reward = np.sum(train_data["reward"])
    train_data_score = env.get_normalized_score(train_data_reward/train_nums)
    val_data_reward = np.sum(val_data["reward"])
    val_data_score = env.get_normalized_score(val_data_reward/val_nums)

    print(f"Env: {env_name},Train data score: {train_data_score}, Val data score: {val_data_score}")    


if __name__ == "__main__":
    env_name_list = ["Pipeline", "Simglucose", "RocketRecovery", "RandomFrictionHopper", 
                            "DMSD", "Fusion", "Salespromotion", "SafetyHalfCheetah"]
    for env_name in env_name_list:
        if env_name=="Fusion":
            test_gym_environment(env_name,1)
        else:
            test_gym_environment(env_name)