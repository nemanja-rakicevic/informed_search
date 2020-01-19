
# Informed Search

Implementation of the Informed Search approach to parameter space exploration. For more details see:

[Full paper](https://link.springer.com/article/10.1007%2Fs10514-019-09842-7)

[Project website](https://sites.google.com/view/informedsearch)


## Motivation and Method Description

- The goal is to efficiently explore the movement parameter space in as few trials as possible. 
- By evaluating the potentially most informative regions of the parameter space,  a robust forward model is obtained.
- The forward model can be used at test time to find the appropriate parameters to perform an action.

![Method components](img/method_components.png)

## Code usage

### Installation

Download repository:

&nbsp;&nbsp;&nbsp;`git clone https://github.com/nemanja-rakicevic/informed_search.git`

&nbsp;&nbsp;&nbsp;`cd informed_search`


Set up conda environment<br/>
(assuming __Conda 4.6.14__, __CUDA v7.5.17__ and __NVIDIA Driver Version 396.44__ already installed) :

&nbsp;&nbsp;&nbsp;`conda env create -f behrep_env_[cpu/gpu].yml`

&nbsp;&nbsp;&nbsp;`conda activate behrep_env_[cpu/gpu]`


Install repository:
(might have to run this twice)

<!-- &nbsp;&nbsp;&nbsp;`python setup.py install clean` -->
&nbsp;&nbsp;&nbsp;`pip install -e . ; python setup.py clean`



### Training

Basic usage example:

&nbsp;&nbsp;&nbsp;`cd behaviour_representations`

&nbsp;&nbsp;&nbsp;`python main_training.py --config_file config_experiments/config_disp_param.json`
