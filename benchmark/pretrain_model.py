import os
import ray
import json
import time
import argparse
import numpy as np
from ray import tune

from offlinerl.algo import algo_select
from offlinerl.data import load_data_from_neorl
from offlinerl.evaluation import ModelCallBackFunction


ResultDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))


def training_function(config):
    ''' run on a seed '''
    config["kwargs"]['seed'] = config['seed']
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(config["kwargs"])
    train_buffer, val_buffer = load_data_from_neorl(algo_config["task"])
    algo_config.update(config)
    algo_config["device"] = "cuda"
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = ModelCallBackFunction()
    callback.initialize(train_buffer=train_buffer, val_buffer=val_buffer, task=algo_config["task"])
    
    res = algo_trainer.train(train_buffer, val_buffer,callback_fn=callback)
    algo_trainer.exp_run.close()
    time.sleep(10)  # sleep ensure the log is flushed even if the disks or cpus are busy


    return res


def find_result(exp_dir: str):
    ''' return the online performance of last epoch and the hyperparameter '''
    data_file = os.path.join(exp_dir, 'objects', 'map', 'dictionary.log')
    with open(data_file, 'r') as f:
        data = json.load(f)
    result = data['__METRICS__']['Reward_Mean_Env'][0]['values']['last']
    grid_search_keys = list(data['hparams']['grid_tune'].keys())
    parameter = {k: data['hparams'][k] for k in grid_search_keys}
    return result, parameter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str)
    parser.add_argument('--algo', type=str, help='select from `bc_model`')
    parser.add_argument('--address', type=str, default=None, help='address of the ray cluster')
    args = parser.parse_args()

    ray.init(args.address)

    domain = args.domain
    algo = args.algo

    ''' run and upload result '''
    config = {}
    config["kwargs"] = {
        "exp_name": f'{domain}-{algo}',
        "algo_name": algo,
        "task": domain,
    }
    _, _, algo_config = algo_select({"algo_name": algo})

    parameter_names = []
    grid_tune = algo_config["grid_tune"]
    for k, v in grid_tune.items():
        parameter_names.append(k)
        config[k] = tune.grid_search(v)

    config['seed'] = 42
    
    analysis = tune.run(
        training_function,
        name=f'{domain}-{algo}',
        config=config,
        metric='loss',
        mode='min',
        resources_per_trial={
            "cpu": 2,
            "gpu": 0.33,  # if no gpu or the memory of gpu is not enough, change this parameter
        }
    )

    df = analysis.results_df
    
    log_folder = os.path.join(ResultDir, config["kwargs"]["exp_name"])
    os.makedirs(log_folder, exist_ok=True)
    df.to_csv(os.path.join(log_folder,"log.csv"))
