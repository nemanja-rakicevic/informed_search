
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

    # def __new__(self, **training_dict):

    #     ae_type_param = training_dict['ae_param']['type']
    #     ae_type_traj = training_dict['ae_traj']['type']

    #     # Extract parameter AE model         
    #     param_ae_fns = {
    #         'param_encoder_fn': 
    #             arch.__dict__['{}_param_encoder'.format(ae_type_param)],
    #         'param_decoder_fn': 
    #             arch.__dict__['{}_param_decoder'.format(ae_type_param)],
    #         'param_branch_fn' : 
    #             arch.__dict__['{}_param_branch_2out'.format('fc')]}

    #     # Select and pass the model object
    #     if ae_type_traj == 'None':
    #         return TrainAE(**param_ae_fns, **training_dict)
    #     else:
    #         # Extract trajectory AE model  
    #         traj_ae_fns = {
    #           'traj_encoder_fn': 
    #               arch.__dict__['{}_traj_encoder'.format(ae_type_traj)],
    #           'traj_decoder_fn': 
    #               arch.__dict__['{}_traj_decoder'.format(ae_type_traj)]}
    #         return TrainAEwTRAJ(**param_ae_fns, **traj_ae_fns, **training_dict)



    def run_trial(self):
        pass


    def evaluate(self):
        pass