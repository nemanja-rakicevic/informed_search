
import os
import numpy as np
import util_modelling as umodel
import util_experiment as uexp
import util_testing as utest
import pickle

import matplotlib.pyplot as plt


dirname = './DATA/SIMULATION/fullrange_validation/'

dirname = './DATA/SIMULATION/'

# Load all tests statistics
all_stats = []
list_models = sorted([d for d in os.listdir(dirname) if d[0:8]=='TRIAL__r'])
list_unique = np.unique([d[:39] for d in list_models])

for idx, t in enumerate(list_models):
	# for s in range(1,6)
	trial_dirname = dirname + t
	with open(trial_dirname + "/data_test_statistics.dat", "rb") as m:
		trial_stats = pickle.load(m)
	# extract the Euclidean error means
	tmp_errs = []
	tmp_fails = []
	for st in trial_stats:
		tmp_errs.append(st[2][0])
		tmp_fails.append(st[4])

	all_stats.append({	"model": t,
						"data" : tmp_errs,
						"std" : tmp_errs,
						"fails": tmp_fails})

# get random trials' mean and variance


# plot all with std

f, axarr = plt.subplots(2, sharex=True)
# Plot errors
for a in all_stats:
	mean = np.array(a['data'])
	std = 1

	lb = a['model'].split('_')[2]+' '+a['model'].split('_')[-1]
	axarr[0].plot(a['data'], label=lb)
	# axarr[0].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)

axarr[0].set_ylabel('Test error (Euclidean distance)')
axarr[0].legend()
# Plot failed trials
for a in all_stats:
	lb = a['model'].split('_')[2]+' '+a['model'].split('_')[-1]
	axarr[1].plot(a['fails'], label=lb)
axarr[1].set_xlabel('Trial number')
axarr[1].set_ylabel('Number of failed test')

plt.show()