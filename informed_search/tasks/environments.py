
"""
Author:         Nemanja Rakicevic
Date:           November 2017
Description:
                Classes for creating and running experiments.
                Supports:
                - MuJoCo simulation
                - (TODO) Baxter robot
"""

import os
import gym
# import envs
import pickle
import logging

import numpy as np
# import multiprocessing as mp

from itertools import product

import informed_search.envs
import informed_search.utils.plotting as uplot
from informed_search.utils.misc import _TAB


logger = logging.getLogger(__name__)


class SimulationExperiment(object):
    """
    Wrapper around a simulated environment task for trial execution and
    testing.
    """

    def __init__(self,
                 dirname,
                 environment,
                 resolution,
                 verbose=False,
                 display=False,
                 **kwargs):
        self.dirname = dirname
        self.verbose = verbose
        self.display = display
        # Initialise environment info
        ename = environment.split('_')[0]
        elim = 'NL' if len(environment.split('_')) > 1 else ''
        env_name = 'Striker{}Link{}Env-v0'.format(ename.strip('simlink'), elim)
        self.env = gym.make(env_name, resolution=resolution)
        self.parameter_list = self.env.unwrapped.parameter_list
        self._num_links = len(self.parameter_list)
        self._action_steps = 100
        self._episode_steps = self.env.unwrapped.spec.max_episode_steps
        # Initialise trial and test info
        self.info_list = []
        self.results_test_list = []
        self.test_angles = np.arange(-65, 31, 5)
        self.test_dist = np.arange(5, 36, 5)
        self.test_cases = np.vstack(
            list(product(self.test_angles, self.test_dist)))

    def _log_trial(self, fail_status, ball_polar, target_dist, test_target,
                   **kwargs):
        if self.verbose:
            error_string = ''
            if fail_status:
                outcome_string = "FAIL ({})".format(fail_status)
            else:
                outcome_string = "SUCCESS\t> ball_polar: {}".format(ball_polar)
                if test_target is not None:
                    error_string = "; euclid error: {}".format(target_dist)
            logger.info("--- trial executed: {}{}".format(
                outcome_string, error_string))

    @property
    def n_total(self):
        return len(self.info_list)

    @property
    def n_fail(self):
        if len(self.info_list):
            return np.count_nonzero([t['fail_status'] for t in self.info_list])
        else:
            return 0

    @property
    def n_success(self):
        if len(self.info_list):
            return self.n_total - self.n_fail
        else:
            return 0

    def execute_trial(self, param_coords, param_vals,
                      num_trial=None, test_target=None):
        """Execute a trial defined by parameters."""
        # Set up sequence of intermediate positions
        param_seq = np.array([np.linspace(0, p, self._action_steps)
                             for p in param_vals]).T
        # Place target for testing or just hide it
        if test_target is not None:
            self.env.unwrapped.init_qpos[-2] = \
                -test_target[1] * np.sin(np.deg2rad(test_target[0])) / 100.
            self.env.unwrapped.init_qpos[-1] = \
                test_target[1] * np.cos(np.deg2rad(test_target[0])) / 100.
        else:
            self.env.unwrapped.init_qpos[-2] = 0.0
            self.env.unwrapped.init_qpos[-1] = 1.3
        # Execute trial
        init_pos = self.env.reset()
        init_pos = init_pos[:self._num_links]
        obs_list = []
        for i in range(self._episode_steps):
            if self.display:
                self.env.render()
            if i < self._action_steps:
                control = init_pos + param_seq[i]
            observation, _, done, info_dict = self.env.step(control)
            obs_list.append(observation)
            # check collision
            if done:
                fail_status = 1
                break
        if self.display:
            self.env.close()
        # Check ball movement and calculate polar coords
        ball_xy = info_dict['ball_xy']
        if np.linalg.norm(ball_xy) > 1e-04:
            fail_status = 0
            ball_polar = np.array([
                np.rad2deg(np.arctan2(-ball_xy[0], ball_xy[1])),
                np.linalg.norm(ball_xy) * 100])
        else:
            fail_status = 2
            ball_polar = np.array([0, 0])
        # Compile trial info
        trial_info = {
            'trial_num': num_trial,
            'test_target': test_target,
            'parameters': param_vals,
            'coordinates': param_coords,
            'fail_status': fail_status,
            'trial_outcome': 'SUCCESS' if fail_status == 0 else 'FAIL',
            'ball_polar': ball_polar,
            'target_dist': observation[-1],
            'observations': np.vstack(obs_list)}
        self._log_trial(**trial_info)
        return trial_info

    def run_test_case(self, model_object, test_target, **kwargs):
        """Evaluate the learned model on a single test target."""
        # Generate movement parameter vector
        tc_coords, tc_params, model_polar_error, model_pidf = \
            model_object.query_target(*test_target, **kwargs)
        # Execute given parameter vector
        trial_info = self.execute_trial(tc_coords, tc_params,
                                        test_target=test_target)
        # Get test performance
        euclid_error = trial_info['target_dist']
        polar_error = np.linalg.norm(trial_info['ball_polar'] - test_target)
        # Trial stats dict
        test_stats = trial_info.copy()
        test_stats.update({'euclid_error': euclid_error,
                           'polar_error': polar_error,
                           'model_polar_error': model_polar_error,
                           'model_pidf': model_pidf})
        # Return based on outcome
        if test_stats['trial_outcome'] == 'FAIL':
            return -1, -1, test_stats
        else:
            return polar_error, euclid_error, test_stats

    def full_tests_sequential(self, num_trial, model_object,
                              save_test_progress=True, **kwargs):
        """Evaluate the learned model on the full test set."""
        ldist, langle = len(self.test_dist), len(self.test_angles)
        euclid_plot = []
        polar_plot = []
        statistics = []
        for t, tcase in enumerate(self.test_cases):
            if self.verbose:
                print("\nTEST # {} > angle, distance: ({},{})".format(
                    t, *tcase))
            # Get parameter for test case and execute
            polar_error, euclid_error, test_stats = self.run_test_case(
                model_object=model_object, test_target=tcase)
            euclid_plot.append(euclid_error)
            polar_plot.append(polar_error)
            statistics.append(test_stats)
        # Generate plots
        euclid_plot = np.array(euclid_plot[::-1]).reshape((langle, ldist)).T
        polar_plot = np.array(polar_plot[::-1]).reshape((langle, ldist)).T
        # Save statistics and plots
        if save_test_progress:
            self.save_test_results(
                num_trial, statistics, euclid_plot, polar_plot, **kwargs)
        return statistics

    def save_trial_data(self):
        """Save training data."""
        with open(self.dirname + "/statistics_trials.dat", "wb") as f:
                pickle.dump(self.info_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Avoid crashing during writing
        os.system("mv {} {}".format(
            self.dirname + "/statistics_trials.dat",
            self.dirname + "/statistics_trials.pkl"))

    def save_test_results(self, num_trial, statistics, euclid_plot, polar_plot,
                          save_plots=True, save_data=True, **kwargs):
        """Save evaluation data."""
        test_dict = {'angles': self.test_angles,
                     'dist': self.test_dist}
        errors_all = np.array([[x['euclid_error'], x['polar_error']]
                               for x in statistics])
        errors_mean = errors_all.mean(axis=0)
        errors_std = errors_all.std(axis=0)
        num_total = len(statistics)
        num_fails = sum([x['fail_status'] > 0 for x in statistics])
        num_success = num_total - num_fails
        model_error_mean = np.mean([x['model_polar_error'] for x in statistics])
        # Log test results
        logger.info("{} TESTING {} cases {}"
                    "\n{} - Sucessful/Failed:       {} / {} ({})"
                    "\n{} - Model polar error mean: {:4.2f}"
                    "\n{} - Euclidian error mean:   {:4.2f}"
                    "\n{} - Polar error norm mean:  {:4.2f}\n{}{}".format(
                        '-' * 15, num_total, '-' * 15, _TAB,
                        num_success, num_fails, num_total, _TAB,
                        model_error_mean, _TAB,
                        errors_mean[0], _TAB, errors_mean[1], _TAB, '-' * 50))
        # Plot and save data
        if save_plots:
            uplot.plot_evals(
                euclid_plot, polar_plot, errors_mean, test_dict,
                savepath=self.dirname, num_trial=num_trial)
        if save_data:
            self.results_test_list.append(
                [num_trial, statistics, euclid_plot, polar_plot,
                 errors_mean, errors_std, num_fails])
            with open(self.dirname + "/statistics_evaluation.dat", "wb") as f:
                pickle.dump(
                    self.results_test_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Avoid crashing during writing
            os.system("mv {} {}".format(
                self.dirname + "/statistics_evaluation.dat",
                self.dirname + "/statistics_evaluation.pkl"))


class RobotExperiment(object):
    """
    (TODO) Wrapper around the Baxter robot,
    for physical robot experiments trial execution and testing.
    """

    pass
