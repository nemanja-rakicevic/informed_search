
"""
Author:         Nemanja Rakicevic
Date:           January 2020
Description:
                Managing and interfacing within the experiment.
"""

import logging

import informed_search.models.modelling as umodel
import informed_search.tasks.environments as uenv

from informed_search.utils.misc import _TAB


logger = logging.getLogger(__name__)


class ExperimentManager(object):
    """Manages interfacing the search algorithm, model and the environment."""

    def __init__(self, task_kwargs, load_model_path=None):
        # Initialise the experiment type
        self.environment = uenv.SimulationExperiment(**task_kwargs)
        # Initialise the search algorithm
        search_algo = task_kwargs['search_algo']
        search_algo = search_algo if search_algo.isupper() \
            else search_algo.capitalize()
        self.model = getattr(umodel, search_algo + 'Search')(
            parameter_list=self.environment.parameter_list,
            **task_kwargs)
        # Load previous experiment
        self.load_model_path = load_model_path
        if self.load_model_path is not None:
            self.model.load_model(load_model_path)

    def run_trial(self, num_trial, **kwargs):
        """
        Interface to generate and execute a parameter vector,
        and update the model based on the outcome.
        """
        # Generate trial parameters
        trial_coords, trial_params = self.model.generate_sample()
        if trial_coords is None:
            return 0
        # Execute trial
        trial_info = self.environment.execute_trial(
            trial_coords, trial_params, num_trial=num_trial)
        self.environment.info_list.append(trial_info)
        self.environment.save_trial_data()
        # Update model
        self.model.update_model(self.environment.info_list, **kwargs)
        # Log trial info
        logger.info("TRIAL {}:"
                    "\n{} - Trial coords: {} params: {}"
                    "\n{} - Fail_status: {}; Distance to target: {:4.2f}"
                    "\n{} - Total (failed: {}; successful: {})"
                    "\n{} - Updated model uncertainty: {:4.2}".format(
                        num_trial,
                        _TAB, trial_coords, trial_params, _TAB,
                        trial_info['fail_status'], trial_info['target_dist'],
                        _TAB, self.environment.n_fail,
                        self.environment.n_success, _TAB,
                        self.model.uncertainty))

    def evaluate_test_cases(self,
                            num_trial,
                            save_model=True,
                            save_test_progress=True,
                            **kwargs):
        """Interface to evaluate all test cases."""
        if save_model:
            self.model.save_model(num_trial=num_trial, **kwargs)
        return self.environment.full_tests_sequential(
            num_trial=num_trial,
            model_object=self.model,
            save_test_progress=save_test_progress,
            **kwargs)

    def evaluate_single_test(self, test_target):
        """Interface to evaluate a single test case."""
        return self.environment.run_test_case(
            model_object=self.model,
            test_target=test_target)
