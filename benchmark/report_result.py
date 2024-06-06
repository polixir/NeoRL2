import os
import glob
import numpy as np
import pandas as pd

import neorl2
import gymnasium as gym


ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
logs_files = glob.glob(os.path.join(ResultDir, '**/*.csv'), recursive=True)

domains = ['Pipeline', 'Simglucose', 'RocketRecovery', 'RandomFrictionHopper', 'DMSD', 'Fusion', 'SafetyHalfCheetah']
algos = ["bc", "cql", "edac", "mcq", "td3bc", "mopo", "combo", "rambo", "mobile"]
df = pd.DataFrame(index=domains, columns=algos)


def process_column(column):
    column = column.str.split('_', n=1, expand=True)[1] 
    column = column.str.split(',seed', n=1, expand=True)[0]
    
    return column


task_reward = []
task_score = []
for task in domains:
    env = gym.make(task)
    train_data,val_data = env.get_dataset()
    train_nums = np.logical_or( np.bool_(train_data["done"]),np.bool_(train_data["truncated"])).sum()
    train_data_reward = np.sum(train_data["reward"])/train_nums
    train_data_score = env.get_normalized_score(train_data_reward)
    
    task_reward.append(train_data_reward)
    task_score.append(train_data_score)

for file_path in logs_files:
    if "bc_model" in file_path:
        continue
    if "old" in file_path:
        continue
    task = os.path.split(os.path.split(file_path)[0])[1]
    domain, algo = task.split("-")
    log = pd.read_csv(file_path)
    log = log[["Reward_Mean_Env","experiment_tag"]]
    log['experiment_tag'] = process_column(log['experiment_tag'])
    log = log.groupby(['experiment_tag']).agg({'Reward_Mean_Env': ['mean', 'std']})
    df.loc[df.index== domain, algo] = str(log.loc[log['Reward_Mean_Env','mean'].idxmax()]['Reward_Mean_Env',"mean"].round(2))+"±"+str(log.loc[log['Reward_Mean_Env','mean'].idxmax()]['Reward_Mean_Env',"std"].round(2))
    
df.insert(0, 'data', np.array(task_reward))

df.to_csv("task_reward.csv")
print("Reward result:\n",df)

df = pd.DataFrame(index=domains, columns=algos)

for file_path in logs_files:
    if "bc_model" in file_path:
        continue
    if "old" in file_path:
        continue
    task = os.path.split(os.path.split(file_path)[0])[1]
    domain, algo = task.split("-")
    
    log = pd.read_csv(file_path)
    log = log[["Reward_Mean_Env","experiment_tag"]]
    log['experiment_tag'] = process_column(log['experiment_tag'])
    env = gym.make(domain)
    log["Reward_Mean_Env"] = log["Reward_Mean_Env"].apply(env.get_normalized_score)
    log = log.groupby(['experiment_tag']).agg({'Reward_Mean_Env': ['mean', 'std']})
    df.loc[df.index== domain, algo] = str(log.loc[log['Reward_Mean_Env','mean'].idxmax()]['Reward_Mean_Env',"mean"].round(2))+"±"+str(log.loc[log['Reward_Mean_Env','mean'].idxmax()]['Reward_Mean_Env',"std"].round(2))

df.insert(0, 'data', np.array(task_score))
df.to_csv("task_score.csv")
print("Score result:\n",df)


for file_path in logs_files:
    if "bc_model" in file_path:
        continue
    if "old" in file_path:
        continue
    task = os.path.split(os.path.split(file_path)[0])[1]
    domain, algo = task.split("-")
    
    log = pd.read_csv(file_path)
    log = log[["Reward_Mean_Env","experiment_tag"]]
    log['experiment_tag'] = process_column(log['experiment_tag'])
    env = gym.make(domain)
    log["Reward_Mean_Env"] = log["Reward_Mean_Env"].apply(env.get_normalized_score)
    log = log.groupby(['experiment_tag']).agg({'Reward_Mean_Env': ['mean', 'std']})
    df.loc[df.index== domain, algo] = log.loc[log['Reward_Mean_Env','mean'].idxmax()]['Reward_Mean_Env',"mean"].round(2)

df_filled = df.fillna(0)

df = pd.DataFrame(index=algos, columns=[f"+{i}" for i in [0, 3, 5, 10]])

for i in [0, 3, 5, 10]:
    df[f"+{i}"] = (df_filled.sub(df_filled["data"]+i, axis=0)>0).sum(axis=0)[1:]

df.to_csv("successful_tasks.csv")
print(df)