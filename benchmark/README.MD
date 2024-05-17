## Instruction for reproducing the benchmark

Before getting started, please make sure you have installed NeoRL2 according to the instructions provided  [here](../README.md) . All the results of the benchmarking will be stored in the `results` folder.

### Step 1: Install dependencies

Benchmark uses algorithms from the OfflineRL library for training. Before training, you need to install the OfflineRL library. You can install it directly using pip:

```
pip install offlinerl
```

Alternatively, you can clone and install it from [GitHub](https://github.com/polixir/OfflineRL):

```bash
git clone https://github.com/polixir/OfflineRL
cd OfflineRL/
pip install -e .
```

### Step 2: Download the datasets

Download all the datasets before training the algorithms by `python download_datasets.py`.

### Step 3: Train policies by Offline RL Algorithms

The following command starts all tasks sequentially for training. Each task will undergo hyperparameter search, and each set of hyperparameters will be trained using 3 random seeds.

```
python launch_domain.py
```

You can also start a specific task for training individually. Invoke all algorithms for hyperparameter search training on a specific dataset.

```
python launch_domain.py --domain Pipeline
```

Launch a single task for hyperparameter search training.

```
python launch_task.py --domain Pipeline --algo bc
```

Note: The Model Base method requires pretraining the model first:

```
python pretrain_model.py --domain Pipeline --algo bc_model

python launch_task.py --domain Pipeline --algo mopo
```

### Step 4: Summarizing Experimental Results

After training is completed, you can use the "report_result.py" script to summarize all training results and display them in a single table.

```
python report_result.py
```
