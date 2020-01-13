
"""
Author:         Nemanja Rakicevic
Date:           October 2018
Description:
                Run a grid search of the hyperparameters.
"""

import itertools
import os

hypersweep_model = ['uidf', 'entropy']
hypersweep_covs = [5, 10, 20]
hypersweep_sqlen = [0.001, 0.01, 0.1, 0.5]

test_hypersweep = [x for x in itertools.product(hypersweep_model, \
                                                hypersweep_covs, \
                                                hypersweep_sqlen)]

for i in test_hypersweep:
	print("\n>>> Running ", i)
	os.system("python simulation_experiments.py -m {} -o {} {} 1".format(
                i[0], str(i[1]), str(i[2]) ))
