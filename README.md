

# Informed Search

Implementation of the Informed Search approach to parameter space exploration. For more details see:

[Full paper](https://link.springer.com/article/10.1007%2Fs10514-019-09842-7)

[Project website](https://sites.google.com/view/informedsearch)


## Motivation and Method Description

- The goal is to efficiently explore the movement parameter space in as few trials as possible. 
- By evaluating the potentially most informative regions of the parameter space, a robust forward model is obtained.
- The informative regions are obtained via Informed Search, by using 3 functions defined over the parameter space:
    - Penalisation function 
    - Model uncertainty
    - Selection function
- The forward model can be used at test time to find the appropriate parameters to perform a desired action.

![Method components](img/method_components.png)


This approach has been implemented in simulation on a puck striking task as well as on the real dual-arm robot.


<img src="img/simulation_experiment.png" width="425"/>   <img src="img/deniro_hockey.jpg" width="380"/> 

---

# Using the code


## Prerequisites

MuJoCo: MuJoCo licence


##  Installation

Navigate to the base installation directory where to download the repository:
```
cd <INSTALL_PATH>
```


Download repository:

```
git clone https://github.com/nemanja-rakicevic/informed_search.git
cd informed_search

export $INFOSEARCH_PATH=$(pwd)

```


Sync to latest repo (*This will overwrite any local untracked changes!*):

`git fetch --all; git reset --hard origin/master`



Set up Conda environment with all dependencies. 
Choose whether to use GPU or not:

```
conda env create -f infosearch_env.yml

conda activate infosearch_env

```

Install repository:

`pip install -e . ; python setup.py clean`

This automatically adds `$INFOSEARCH_PATH` to `$PYTHON_PATH`.


##  Training

Basic usage example:

```
cd $INFOSEARCH_PATH

python main_training.py --config_file config_experiments/{striker, walker, quadruped}/nn_policy__ls_mape_ae.json
```


##  Preview saved behaviours

```
cd $INFOSEARCH_PATH
python test_run_env.py --load <path to experiment directory>
```


---

# Reference
```
@article{rakicevic2019active,
  title={Active learning via informed search in movement parameter space for efficient robot task learning and transfer},
  author={Rakicevic, Nemanja and Kormushev, Petar},
  journal={Autonomous Robots},
  volume={43},
  number={8},
  pages={1917--1935},
  year={2019},
  publisher={Springer}
}
```
