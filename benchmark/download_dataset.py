import time
import neorl2
import gymnasium as gym


def test_gym_environment(env_name, nums=1):
    t1 = time.time()
    env = gym.make(env_name)
    train_data,val_data = env.get_dataset()

if __name__ == "__main__":
    env_name_list = ["Pipeline", "Simglucose", "RocketRecovery", "RandomFrictionHopper", 
                            "DMSD", "Fusion", "SafetyHalfCheetah"]
    for env_name in env_name_list:
        if env_name=="Fusion":
            test_gym_environment(env_name,1)
        else:
            test_gym_environment(env_name)