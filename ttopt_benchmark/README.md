# optrecsys


## Description

This python package, provides a set of methods for selecting the best subset of features by QUBO approach and a specialized discrete multidimensional optimization method TTOpt.

## Installation

1. Install [python](https://www.python.org) (version >= 3.7; you may use [anaconda](https://www.anaconda.com) package manager);
2. Install basic dependencies:
    ```bash
    pip install numpy scipy
    ```
3. Install additional dependency [maxvolpy](https://bitbucket.org/muxas/maxvolpy/src/master/) with effective realization of `maxvol` algorithm:
    ```bash
    pip install maxvolpy
    ```
4. Install additional dependency [ttopt](https://github.com/SkoltechAI/ttopt) with gradient-free optimization method:
    ```bash
    python setup.py install
    ```


## Usage

1. Unzip Feature Penalty Matrices (the data files should have structure like `DATAFOLDER/CiteULike_a_FPM/FPM_E.npz`, `DATAFOLDER/CiteULike_a_FPM/FPM_K.npz`)
2. Set correct options (including path to data, i.e., `FPATH_PREFIX`) in `opts.py`
3. Run the script `python demo.py` with one of the following arguments:
    - `citeulike` - for `CiteULike` dataset
    - `movies` - for `TheMovies` dataset
    - `xing` - for `XingChallenge2017` dataset
4. Run the script `python calc.py` to build and save the accuracy and time dependency vs number of function requests for all datasets (data will be saved into `results` folder)
5. Run the script `python plot.py` to plot saved accuracy and time dependency vs number of function requests (plots will be saved into `results` folder)
