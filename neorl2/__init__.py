import os
import sys
import importlib
import gymnasium
from gymnasium.envs.registration import register

register(
    id="Pipeline",
    entry_point="neorl2.envs.pipline:PipelineEnv",
    max_episode_steps=1000,
)

register(
    id="Simglucose",
    entry_point="neorl2.envs.simglucose:SimglucoseEnv",
    max_episode_steps=480,
    kwargs={'mode': 'train'},
)

register(
    id="RocketRecovery",
    entry_point="neorl2.envs.rocketrecovery:RocketRecoveryEnv",
    max_episode_steps=500,
    kwargs={'mode': 'train'},
)

register(
    id="RandomFrictionHopper",
    entry_point="neorl2.envs.randomfrictionhopper:RandomFrictionHopperEnv",
    kwargs = dict(random_friction = True),
    max_episode_steps=1000,
)

register(id='DMSD', 
        entry_point="neorl2.envs.dmsd:DoubleMassSpringDamperEnv", 
        kwargs = dict(mode = 'train'),
        max_episode_steps = 100                    
        )

register(id='Fusion', 
    entry_point='neorl2.envs.fusion:FusionEnv',
    kwargs = dict(random_target = True,
                mode = 'train'),
    max_episode_steps = 100
    )

register(
    id='Salespromotion', 
    entry_point='neorl2.envs.salespromotion:SalesPromotion_v0',
    kwargs = dict(num_users = 1,
                max_give_coupon_num = 100,
                mode = 'train'),
    max_episode_steps = 50
)

register(
    id="SafetyHalfCheetah",
    entry_point="neorl2.envs.safetyhalfcheetah:SafetyHalfCheetahEnv",
    max_episode_steps=1000,
)

def make(task: str, reward_func=None, done_func=None, *args, **kwargs):
    try:
        if task.lower() in ["pipeline", "simglucose", "rocketrecovery", "randomfrictionhopper", 
                            "dmsd", "fusion", "salespromotion", "safetyhalfcheetah"]:
            if task.lower() == "pipeline":
                env = gymnasium.make("Pipeline")
            elif task.lower() == "simglucose":
                env = gymnasium.make("Simglucose")
            elif task.lower() == "rocketrecovery":
                env = gymnasium.make("RocketRecovery")
            elif task.lower() == "randomfrictionhopper":
                env = gymnasium.make("RandomFrictionHopper")
            elif task.lower() == "dmsd":
                env = gymnasium.make("DMSD")
            elif task.lower() == "fusion":
                env = gymnasium.make("Fusion")
            elif task.lower() == "salespromotion":
                inkwargs = dict(num_users = 1,
                                max_give_coupon_num = 100,
                                mode = 'train')
                inkwargs.update(kwargs)
                register(
                    id='Salespromotion', 
                    entry_point='neorl2.envs.salespromotion:SalesPromotion_v0',
                    kwargs = inkwargs,
                    max_episode_steps = 50
                )
                env = gymnasium.make("Salespromotion")
            else:
                env = gymnasium.make("SafetyHalfCheetah")
            def load_module_from_file(file_path, module_name):
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)

                return module
            
            try:
                reward_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"envs/reward/{task.lower()}_reward.py")
                default_reward_func = load_module_from_file(reward_file_path, module_name = f"{task.lower()}_reward").get_reward
                if task.lower()  == "salespromotion" and env.mode!='train':
                    default_reward_func = load_module_from_file(reward_file_path, module_name = f"{task.lower()}_reward").get_reward_val
            except ModuleNotFoundError:
                default_reward_func = None
            env.set_reward_func(default_reward_func if reward_func is None else reward_func)

            try:
                done_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"envs/terminated/{task.lower()}_terminated.py")
                default_done_func = load_module_from_file(done_file_path,module_name = f"{task.lower()}_done").get_terminated
            except FileNotFoundError:
                default_done_func = None
            
            env.set_done_func(default_done_func if done_func is None else done_func)
        else:
            try:
                import neorl
                env = neorl.make(task)
            except ModuleNotFoundError:
                pass
    except Exception as e:
        print(f'Warning: Env {task} can not be create. Pleace Check!')
        raise e
    
    return env