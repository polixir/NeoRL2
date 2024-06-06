import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import neorl2
import gymnasium as gym
from offlinerl.algo import algo_select

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
    
best_params = {algo:{} for algo in algos}
all_params_result = {algo:{} for algo in algos}
    
for file_path in logs_files:
    if "bc_model" in file_path:
        continue
    task = os.path.split(os.path.split(file_path)[0])[1]
    domain, algo = task.split("-")
    _, _, algo_config = algo_select({"algo_name": algo})
    log = pd.read_csv(file_path)
    
    log = log[["Reward_Mean_Env","experiment_tag"] + ["hparams/"+i for i in list(algo_config["grid_tune"].keys())]]

    def generate_experiment_tag(row):
        hparams = {col.split('/')[1]: value for col, value in row.items() if col.startswith('hparams/')}
        experiment_tag = ';'.join(f'{key}={value}' for key, value in hparams.items())
        return experiment_tag
    log['experiment_tag'] = log.apply(lambda row: generate_experiment_tag(row), axis=1)
    log['experiment_tag'] = log['experiment_tag'].str.replace('hparams/', '')

    
    log = log.groupby(['experiment_tag']).agg({'Reward_Mean_Env': ['mean', 'std']})
    
    env = gym.make(domain)
    scores = [env.get_normalized_score(r) for r in list(log['Reward_Mean_Env','mean'].values)]
    all_params_result[algo][domain] = [max(-10,s) for s in scores]
    params = log.loc[log['Reward_Mean_Env','mean'].idxmax()].name
    
    best_params[algo][domain] = dict(item.split('=') for item in params.split(';'))
    
concatenated_df = pd.DataFrame()

for k,v in best_params.items():
    v = pd.DataFrame(v)
    #v["Hyper-parameters"] = v.index
    v.index.name = "Hyper-parameters"
    v = v.reset_index()
    v.insert(0, 'Algorithm', k.upper())
    
    v = v.set_index(['Algorithm', 'Hyper-parameters'])
    # 将当前数据框与拼接结果进行拼接
    concatenated_df = pd.concat([concatenated_df, v])

# 打印拼接结果
print(concatenated_df)
#print(concatenated_df.to_latex(index=True))
concatenated_df.to_csv("./exp/best_hyperparameters.csv")


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, axes = plt.subplots(9, 1, figsize=(10*2, 15*2))
scatter_list = []
for i,algo in enumerate(algos):
    for j,domain in enumerate(domains):
        num = len(all_params_result[algo][domain])
        scatter = axes[i].scatter([domain,]*num, all_params_result[algo][domain], c=[colors[j],]*num)
        scatter_list.append(scatter)
    axes[i].set_ylabel(f"{algo}", fontsize=25)
    axes[i].set_xticks(range(len(domains)))
    axes[i].set_xticklabels(domains, fontsize=18)
    axes[i].set_ylim(-10, 100)
plt.subplots_adjust(hspace=0.2)
plt.tight_layout()
plt.savefig("hyperparameter_sensitivity.svg", format="svg")
plt.savefig("hyperparameter_sensitivity.png", format="png")



for file_path in logs_files:
    if "bc_model" in file_path:
        continue
    task = os.path.split(os.path.split(file_path)[0])[1]
    domain, algo = task.split("-")
    log = pd.read_csv(file_path)
    log = log[["Reward_Mean_Env","experiment_tag"]]
    log['experiment_tag'] = process_column(log['experiment_tag'])
    log = log.groupby(['experiment_tag']).agg({'Reward_Mean_Env': ['mean', 'std']})
    df.loc[df.index== domain, algo] = str(log.loc[log['Reward_Mean_Env','mean'].idxmax()]['Reward_Mean_Env',"mean"].round(2))+"±"+str(log.loc[log['Reward_Mean_Env','mean'].idxmax()]['Reward_Mean_Env',"std"].round(2))
    
df.insert(0, 'data', np.array(task_reward))

df.to_csv("./exp/task_reward.csv")
print("Reward result:\n",df)

df = pd.DataFrame(index=domains, columns=algos)

for file_path in logs_files:
    if "bc_model" in file_path:
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
df.to_csv("./exp/task_score.csv")
print("Score result:\n",df)


for file_path in logs_files:
    if "bc_model" in file_path:
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

df.to_csv("./exp/successful_tasks.csv")
print(df) 