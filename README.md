# NeoRL2

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License](https://licensebuttons.net/l/by/3.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

The NEORL2 repository is an extension of the offline reinforcement learning benchmark [NeoRL](https://github.com/polixir/NeoRL). The NEORL2 repository contains datasets for training and corresponding environments for testing the trained policies. The current datasets are collected from seven open-source environments: Pipeline, Simglucose, RocketRecovery, RandomFrictionHopper, DMSD, Fusion and SafetyHalfCheetah tasks. We perform online training using reinforcement learning algorithms or PID policies on these tasks and then select suboptimal policies with returns ranging from 50% to 80% of the expert's return to generate offline datasets for each task. These suboptimal policy-sampled datasets better align with real-world task scenarios compared to random or expert policy datasets.

## Install NeoRL2 interface

NeoRL2 interface can be installed as follows:

```
git clone https://agit.ai/Polixir/neorl2.git
cd neorl
pip install -e .
```

After installation, Pipeline、Simglucose、RocketRecover、DMSD and Fusion environments will be available. However, the "RandomFrictionHopper" and "SafetyHalfCheetah" tasks rely on MuJoCo. If you need to use these two environments, it is necessary to obtain a [license](https://www.roboti.us/license.html) and follow the setup instructions, and then run:

```
pip install -e .[mujoco]
```

## Using NeoRL2

NeoRL2 uses the [OpenAI Gym](https://github.com/openai/gym) API. Tasks can be created as follows:

```
import neorl2
import gymnasium as gym

# Create an environment
env = gym.make("Pipeline")
env.reset()
env.step(env.action_space.sample())
```

After creating the environment, you can use the `get_dataset()` function to obtain training data and validation data:

```
train_data, val_data = env.get_dataset()
```

Each environment supports setting and getting the reward function and done function of the environment, which is very useful for adjusting the environment settings when needed.

```
# Set reward function
env.set_reward_func(reward_func)

# Get reward function
env.get_reward_func(reward_func)

# Set done function
env.get_done_func(done_func)

# Get done function
env.set_done_func(done_func)
```

## Data in NeoRL2

In NeoRL2, training data and validation data returned by `get_dataset()` function are `dict` with  the same format:

- `obs`: An <i> N by observation dimensional array </i> of current step's observation.

- `next_obs`: An <i> N by observation dimensional array </i> of next step's observation.

- `action`: An <i> N by action dimensional array </i> of actions.

- `reward`: An <i> N dimensional array of rewards</i>.

- `done`: An <i> N dimensional array of episode termination flags</i>.

- `index`: An <i> trajectory number-dimensional array</i>. The numbers in index indicate the beginning of trajectories.

## Reference

**Simglucose**: Jinyu Xie. Simglucose v0.2.1 (2018) [Online]. Available: https://github.com/jxx123/simglucose. Accessed on: 5-17-2024. [code](https://github.com/jxx123/simglucose)

**DMSD**: Char, Ian, et al. "Correlated Trajectory Uncertainty for Adaptive Sequential Decision Making." *NeurIPS 2023 Workshop on Adaptive Experimental Design and Active Learning in the Real World*. 2023. [paper](https://arxiv.org/abs/2307.05891) [code](https://github.com/IanChar/GPIDE/tree/main)

**MuJoCo**: Todorov E, Erez T, Tassa Y. "Mujoco: A Physics Engine for Model-based Control." Proceedings of the 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026-5033, 2012. [paper](https://ieeexplore.ieee.org/abstract/document/6386109) [website](https://gym.openai.com/envs/#mujoco)

**Gym**: Brockman, Greg, et al. "Openai gym." *arXiv preprint arXiv:1606.01540* (2016). [paper](https://arxiv.org/abs/1606.01540) [code](https://github.com/openai/gym)

## Licenses

All datasets are licensed under the [Creative Commons Attribution 4.0 License (CC BY)](https://creativecommons.org/licenses/by/4.0/), and code is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html).