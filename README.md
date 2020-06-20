

# Informed Search

Implementation of the Informed Search approach to parameter space exploration. 
For more details see:

[Full paper](https://link.springer.com/article/10.1007%2Fs10514-019-09842-7)

[Project website](https://sites.google.com/view/informedsearch)


## Motivation and Method Description

- The goal is to efficiently explore the movement parameter space in as few 
trials as possible. 
- By evaluating the potentially most informative regions of the parameter space, 
a robust forward model is obtained.
- The forward model can be used at test time to find the appropriate parameters 
to perform a desired action.
- The informative regions are obtained via Informed Search, by using 3 
functions defined over the parameter space:
    - Penalisation function 
    - Model uncertainty
    - Selection function

<!-- ![Method components](img/method_components.png) -->
<p align="center">
  <img src="img/method_components.png" width="700" /> 
</p>


- This approach has been implemented in simulation on a puck striking task, 
as well as on the real dual-arm robot.


&emsp;Simulation Experiment
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Robot Experiment
<p align="center">
  <img src="img/simulation_experiment.png" width="400"  align="left"  
       title="Simulation Experiment"/>   
  <img src="img/deniro_hockey.jpg" width="350"  align="top" 
       title="Robot Experiment"/> 
</p>


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


##  Load saved models

Evaluate the learned model from a specified path, on the whole test set:
```
python load_full_test.py --load <path to experiment directory>
```

Check performace of the loaded model for specific test target positions:
```
python load_target_test.py --load <path to experiment directory>
```


---

# Reference
```
@Article{Rakicevic2019informed,
         author="Rakicevic, Nemanja and Kormushev, Petar",
         title="Active learning via informed search in movement parameter space for efficient robot task learning and transfer",
         journal="Autonomous Robots",
         year="2019",
         month="Feb",
         day="21",
         issn="1573-7527",
         doi="10.1007/s10514-019-09842-7",
         url="https://doi.org/10.1007/s10514-019-09842-7"
}
```
