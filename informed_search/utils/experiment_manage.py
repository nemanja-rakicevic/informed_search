
"""
Author:         Nemanja Rakicevic
Date:           January 2020
Description:
                Managing and interfacing within the experiment 
"""

import logging
import numpy as np

import utils.modelling  as umodel
import utils.environments as uenv
from utils.misc import _TAB


logger = logging.getLogger(__name__) 


class ExperimentManager(object):
    """
        Manages interfacing the search algorithm, model and the environment.
    """

    def __init__(self, task_kwargs, load_model_path=None):
        # Initialise the experiment type
        self.environment = uenv.SimulationExperiment(**task_kwargs)
        # Initialise the search algorithm
        stype = task_kwargs['search_type']
        stype = stype if stype.isupper() else stype.capitalize()
        self.model = umodel.__dict__[stype+'Search'](
            parameter_list=self.environment.parameter_list, **task_kwargs)
        self.load_model_path = load_model_path
        if self.load_model_path is not None:
            self.model.load_model(load_model_path)
    

    def execute_trial(self, num_trial):
        # Generate trial parameters
        trial_coords, trial_params = self.model.generate_sample()
        # Execute trial
        trial_info = self.environment.execute_trial(trial_coords, 
                                                    trial_params, 
                                                    num_trial=num_trial)
        self.environment.info_list.append(trial_info)
        self.environment.save_trial_data()
        # Update model
        self.model.update_model(self.environment.info_list)
        # Log trial info
        logger.info("TRIAL {}:"
                    "\n{} - Trial_coords: {}"
                    "\n{} - Fail_status: {}; Distance to target: {:4.2f}"
                    "\n{} - Total (failed: {}; successful: {})"
                    "\n{} - Updated model uncertainty: {:4.2}".format(num_trial, 
                        _TAB, trial_coords, _TAB, 
                        trial_info['fail_status'], trial_info['target_dist'],
                        _TAB, self.environment.n_fail, 
                        self.environment.n_success, _TAB, 
                        self.model.uncertainty))


    def evaluate_test_cases(self, num_trial, verbose=False, save_model=True, 
                                  save_test_progress=True, **kwargs):
        # Save current model that is evaluated
        if save_model: self.model.save_model(num_trial=num_trial, **kwargs)
        # Evaluate on environments test cases, save results
        self.environment.verbose = verbose
        return self.environment.full_tests_sequential(
                                    num_trial=num_trial, 
                                    model_object=self.model,
                                    save_test_progress=save_test_progress, 
                                    **kwargs)


    def evaluate_single_test(self, test_target, display):
        """ Wrapper to evaluate a single test case """
        self.environment.display = display
        return self.environment.run_test_case(model_object=self.model, 
                                              test_target=test_target)
