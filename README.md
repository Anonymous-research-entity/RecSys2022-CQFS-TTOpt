# Are Quantum Computers Practical Yet? A Case for Feature Selection in Recommender Systems using Tensor Networks

This repository contains the source code for the RecSys 2022 Reproducibility track submission.

**It is a fork from the repository of the original paper that we base our work on.**
The original repository can be found here https://github.com/qcpolimi/CQFS. We 
wanted to make it as less intrusive as possible to perform fair replication of
the results. We added a TTOpt optimizer as a solver instead of the D-Wave 
quantum annealer and created necessary experimental scripts by mimicking the 
original source code.

Here we explain how to install dependencies, and how to run
experiments included in this repository.

## Installation

> NOTE: This repository requires Python 3.7

It is suggested to install all the required packages into a new Python environment. So, after repository checkout, enter
the repository folder and run the following commands to create a new environment:

If you're using `virtualenv`:

```bash
virtualenv -p python3 cqfs
source cqfs/bin/activate
```

If you're using `conda`:

```bash
conda create -n cqfs python=3.7 anaconda
conda activate cqfs
```

>Remember to add this project in the PYTHONPATH environmental variable if you plan to run the experiments 
on the terminal:
>```bash
>export PYTHONPATH=$PYTHONPATH:/path/to/project/folder
>```

Then, make sure you correctly activated the environment and install all the required packages through `pip`:
```bash
pip install -r requirements.txt
```

Moreover, install TTOpt library by following the instructions from https://github.com/SkoltechAI/ttopt.

After installing the dependencies, it is suggested to compile Cython code in the repository.

In order to compile you must first have installed: `gcc` and `python3 dev`. Under Linux those can be installed with the
following commands:

```bash
sudo apt install gcc 
sudo apt-get install python3-dev
```

If you are using Windows as operating system, the installation procedure is a bit more complex. You may refer
to [THIS](https://github.com/cython/cython/wiki/InstallingOnWindows) guide.

Now you can compile all Cython algorithms by running the following command. The script will compile within the current
active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some
warnings. Navigate to `recsys/` directory and run:

```bash
python run_compile_all_cython.py
```

## Running CQFS Experiments

First of all, you need to prepare the original files for the datasets.

For The Movies Dataset you need to download
[The Movies Dataset from Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset) and place the compressed files
in the directory `recsys/Data_manager_offline_datasets/TheMoviesDataset/`, making sure the file is called
`the-movies-dataset.zip`.

For CiteULike_a you need to download
[the following .zip file](https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EcjHpkI8TQdHnFVwVMkNGN4BmNkurMWw79sU8kpt4wk8eA?e=QYhdbz)
and place it in the directory `recsys/Data_manager_offline_datasets/CiteULike/`, making sure the file is called
`CiteULike_a_t.zip`.

We cannot provide data for Xing Challenge 2017, but if you have the dataset available, place the compressed file
containing the dataset's original files in the directory `recsys/Data_manager_offline_datasets/XingChallenge2017/`,
making sure the file is called `xing_challenge_data_2017.zip`.

After preparing the datasets, you should run the following command under the `data` directory:


```bash
python split_<NameOfTheDataset>.py
```

This python script will generate the data splits used in the experiments. Moreover, it will preprocess the dataset and
check for any error in the preprocessing phase. The resulting splits are saved in the
`recsys/Data_manager_split_datasets` directory.

After splitting the dataset, you can actually run the experiments. All the experiment scripts are in the `experiments`
directory, so navigate to the folder `experiments/<NameOfTheDataset>` first.
Each dataset has separated experiment scripts that you can find in the corresponding directories.
From now on, we will assume that you are running the following commands in the dataset-specific folders, thus running
the scripts contained there.

### Collaborative models

First of all, we need to optimize the chosen collaborative models to use with CQFS. To do so, run the following command:

```bash
python CollaborativeFiltering.py
```

The resulting models will be saved into the `results` directory.

### CQFS

Then, you can run the CQFS procedure. We divided the procedure into a _selection_ phase and a _recommendation_ phase. To
perform the selection through CQFS run the following command:

```bash
python CQFSTT.py
```

This script will solve the CQFS problem on the corresponding dataset and save all the selected features in appropriate
subdirectories under the `results` directory.

After solving the feature selection problem, you should run the following command:

```bash
python CQFSTTTrainer.py
```

This script will optimize an ItemKNN content-based recommender system for each selection corresponding to the given
hyperparameters (and previously obtained through CQFS), using only the selected features. Again, all the results are
saved in the corresponding subdirectories under the `results` directory.


### Baselines

In order to obtain the baseline evaluations you can run the corresponding scripts with the following commands:

```bash
# ItemKNN content-based with all the features
python baseline_CBF.py

# ItemKNN content-based with features selected through TF-IDF
python baseline_TFIDF.py

# CFeCBF feature weighting baseline
python baseline_CFW.py
```
