import os
import urllib
import hashlib
import numpy as np
import gymnasium as gym
from .info import MIN_SCORES,MAX_SCORES,BASE_DATASET_URL,MD5,DATASET_SAVE_PATH

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_file_md5(filename):
    if not os.path.isfile(filename):
        return
    my_hash = hashlib.md5()
    f = open(filename, 'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        my_hash.update(b)
    f.close()
    return my_hash.hexdigest()

def download_dataset_from_url(dataset_url, dataset_md5, name, to_path, verbose=1):
    """
    Download dataset from url to `to_path + name`.
    """
    to_path = os.path.expanduser(to_path)
    try:
        if not os.path.exists(to_path):
            os.makedirs(to_path)
    except Exception:
        pass

    dataset_filepath = os.path.join(to_path, name)

    if os.path.exists(dataset_filepath):
        local_file_md5 = get_file_md5(dataset_filepath)
        if local_file_md5 == dataset_md5:
            return dataset_filepath
        else:
            print("local_file_md5 :", local_file_md5, "dataset_md5: " ,dataset_md5)
            if verbose != 0:
                print(f"Local dataset {name} is broken, ready to re-download.")
    if verbose != 0:
        print(f'Downloading dataset: {dataset_url} to {dataset_filepath}')
    urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError(f"Failed to download dataset from {dataset_url}")
    return dataset_filepath


def get_dataset_traj_num(dataset,traj_num):
    _index = np.where(np.logical_or(dataset["done"], dataset["truncated"]))[0]+1
    _index = _index[:traj_num]
    for k,v in dataset.items():
        dataset[k] = v[:_index[-1]]
        
    return dataset, dataset['obs'].shape[0]

class Env(gym.Env):
    @property
    def _env_name(self):
        return self.spec.id
    
    def get_normalized_score(self, returns):
        min_score = MIN_SCORES[self._env_name]
        max_score = MAX_SCORES[self._env_name]
        
        normalized_score = (returns - min_score) / (max_score - min_score) * 100
        return normalized_score
    
    def get_dataset(self, traj_num=None):
        task_base_url = f"{BASE_DATASET_URL}/{self._env_name}/"
        name = f"{self._env_name}-train.npz"
        train_dataset_url = task_base_url + name
        train_dataset_md5 = MD5[name]
        train_data_path =download_dataset_from_url(train_dataset_url, train_dataset_md5, name, DATASET_SAVE_PATH)
        
        name = f"{self._env_name}-val.npz"
        val_dataset_url = task_base_url + name
        val_dataset_md5 = MD5[name]
        val_data_path = download_dataset_from_url(val_dataset_url, val_dataset_md5, name, DATASET_SAVE_PATH)
        train_dataset = dict(np.load(train_data_path))
        val_dataset = dict(np.load(val_data_path))
        
        if self.spec.id == "RocketRecovery" and traj_num is None:
            traj_num = 100
        if traj_num != None:
            train_dataset, train_samples = get_dataset_traj_num(train_dataset, traj_num) 
            val_traj_num = int(traj_num/4)
            val_dataset,   val_samples   = get_dataset_traj_num(val_dataset, val_traj_num) 
        
        if self.spec.id == "Fusion" and traj_num is None:
            traj_num = 20
        if traj_num != None:
            train_dataset, train_samples = get_dataset_traj_num(train_dataset, traj_num) 
            val_traj_num = int(traj_num/4)
            val_dataset,   val_samples   = get_dataset_traj_num(val_dataset, val_traj_num) 
        
        return train_dataset, val_dataset
    
    def set_reward_func(self, reward_func):
        """
        Users can call this func to set customized reward func.
        """
        self._reward_func = reward_func

    def get_reward_func(self):
        """
        Users can call this func to get customized reward func.
        """
        return self._reward_func

    def get_name(self):
        """
        Get name of envs.
        """
        return self._name

    def set_name(self, name):
        """
        Set name for envs.
        """
        self._name = name
        
    def set_done_func(self, done_func):
        """
        Users can call this func to set done func.
        """
        self._done_func = done_func

    def get_done_func(self):
        """
        Users can call this func to get done func.
        """
        return self._done_func