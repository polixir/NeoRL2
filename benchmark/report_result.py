import os
import glob
import pandas as pd


ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
logs_files = glob.glob(os.path.join(ResultDir, '**/*.csv'), recursive=True)

domains = ['Pipeline', 'Simglucose', 'RocketRecovery', 'RandomFrictionHopper', 'DMSD', 'Fusion', 'SafetyHalfCheetah']
algos = ["bc", "cql", "edac", "mcq", "td3bc", "mopo", "combo", "rambo", "mobile"]
df = pd.DataFrame(index=domains, columns=algos)


def process_column(column):
    column = column.str.split('_', n=1, expand=True)[1] 
    column = column.str.split(',seed', n=1, expand=True)[0]
    
    return column


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
    
# df.to_csv("./result.csv")
print(df)