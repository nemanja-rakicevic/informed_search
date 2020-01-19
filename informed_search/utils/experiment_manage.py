
"""
Author:         Nemanja Rakicevic
Date:           January 2020
Description:
                Main file for managing the experiment
"""

import numpy as np


import utils.modelling  as umodel
import utils.environemnts as uenv
import utils.testing as utest


class ExperimentManager(object):

    def __init__(self):
            
        experiment = uexp.SimulationExperiment(**task_kwargs)
        model = umodel.InformedModel(experiment.parameter_list, 
                                     experiment.type, 
                                     show_plots=args.plots&1, 
                                     other=[args.other], 
                                     folder_name=task_kwargs['dirname'])
        
        testing = utest.FullTest(experiment, model, 
                                 show_plots=args.plots&2, 
                                 verbose=args.verbose&2)


    def run_iteration(self):
        pass


    def evaluate(self):
        pass