
import pickle
import logging
import numpy as np
import multiprocessing as mp
from itertools import product

import utils.plotting as uplot


import pdb


logger = logging.getLogger(__name__)


class FullTest(object):
    def __init__(self, experiment, model, 
                       show_plots=False, verbose=False, **kwargs):
        self.show_plots = show_plots
        self.verbose = verbose
        self.results_list = []
        self.test_angles = np.arange(-65, 31, 5)
        self.test_dist   = np.arange(5, 36, 5)
        self.test_cases = np.vstack(product(self.test_angles, 
                                                      self.test_dist)) #.astype(float)
        self.num_cpu = 4  # mp.cpu_count() - 1
        self.model = model
        self.experiment = experiment


    def run_test_case(self, test_target):
        # Generate movement parameter vector
        tc_coords, tc_params = self.model.query_target(*test_target)
        # Execute given parameter vector
        trial_info = self.experiment.execute_trial(tc_coords, tc_params, 
                                                   test_params=test_target)
        # Get test performance
        polar_error = np.linalg.norm(trial_info['ball_polar'] - test_target)
        euclid_error = trial_info['target_dist']
        # Trial stats dict
        trial_stats = {'test_target_polar': test_target,
                       'ball_polar': trial_info['ball_polar'],
                       'fail_status': trial_info['fail_status'],
                       'polar_error': polar_error,
                       'euclid_error': euclid_error}
        # Return base on outcome
        if trial_info['fail_status']>0:
            return -1, -1, trial_stats
        else:
            return polar_error, euclid_error, trial_stats


    def full_tests_sequential(self, trial_num, save_progress=True):
        test_dict = {'angles': self.test_angles, 'dist': self.test_dist}
        ldist, langle = len(self.test_dist), len(self.test_angles)

        euclid_plot = []
        polar_plot = []
        statistics = []
        for t in range(len(self.test_cases)):
            if self.verbose:
                print("\nTRIAL {}\nTEST # {} > angle, distance: ({},{})".format(
                        tr_num, t+1, *self.test_cases[t]))
            # Get parameter for test case and execute
            polar_error, euclid_error, trial_stats = \
                self.run_test_case(test_target=self.test_cases[t])
            euclid_plot.append(euclid_error)
            polar_plot.append(polar_error)
            statistics.append(trial_stats)
        # Generate plots
        euclid_plot = np.array(euclid_plot[::-1]).reshape((langle, ldist)).T
        polar_plot = np.array(polar_plot[::-1]).reshape((langle, ldist)).T
        # Save statistics and plots
        if save_progress:
            self.save_test_results(trial_num, statistics, test_dict,
                                   euclid_plot, polar_plot, 
                                   savepath=self.model.dirname)



    # def full_tests_parallel(self, trial_num, save_progress=True, heatmap=True):
    #     test_dict = {'angles': self.test_angles, 'dist': self.test_dist}
    #     ldist, langle = len(self.test_dist), len(self.test_angles)

    #     tc = [self.model.query_target(*test_target)+tuple(test_target) \
    #                 for test_target in self.test_cases]

    #     with mp.Pool(processes=self.num_cpu) as pool:
    #         polar_error, euclid_error, trial_stats = \
    #             pool.starmap(self.run_test_case, tc)

    #     euclid_plot = np.array(euclid_plot[::-1]).reshape((langle, ldist)).T
    #     polar_plot = np.array(polar_plot[::-1]).reshape((langle, ldist)).T

    #     # Save statistics and plots
    #     if save_progress:
    #         self.save_test_results(trial_num, statistics, test_dict,
    #                                euclid_plot, polar_plot, 
    #                                savepath=self.model.dirname)



    def save_test_results(self, trial_num, statistics, test_dict,
                                euclid_plot, polar_plot, savepath):
        errors_all = np.array([[x['euclid_error'], x['polar_error']] \
                                    for x in statistics])
        errors_mean = errors_all.mean(axis=0)
        errors_std  = errors_all.std(axis=0)
        num_fails = np.count_nonzero([x['fail_status'] for x in statistics])
        self.results_list.append([trial_num, statistics, 
                                  euclid_plot, polar_plot, 
                                  errors_mean, errors_std, num_fails])
        # Save statistics
        with open(savepath + "/data_test_statistics.dat", "wb") as m:
            pickle.dump(self.results_list, m, protocol=pickle.HIGHEST_PROTOCOL)
        # Save heatmaps
        uplot.plot_evals(euclid_plot, polar_plot, errors_mean, test_dict,
                         savepath=self.model.dirname, trial_num=trial_num)

