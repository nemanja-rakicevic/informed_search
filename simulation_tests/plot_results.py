
import os
import numpy as np
import util_modelling as umodel
import util_experiment as uexp
import util_testing as utest
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl


# dirname = './DATA/SIMULATION/fullrange_useful/'

dirname = './DATA/SIMULATION/'

# Load all tests statistics
# filt = 'TRIAL__informed_res150_cov20_'
# filt = ['informed', 'RQ','cov']
filt = ['_', 'sl', 'cov']

filename = dirname+'PLOTS_{}_kernel{}_{}.svg'.format(filt[0], filt[1], filt[2])

list_models = sorted([d for d in os.listdir(dirname) if all(s in d for s in filt) and d[-3:]!='svg'])
# list_models = sorted([d for d in os.listdir(dirname) if d[0:len(filt)]==filt])
list_unique = np.unique([d[:-6] for d in list_models])

all_stats = []
for idx, t in enumerate(list_unique):
	seeds_error_mean = []
	seeds_error_std = []
	seeds_fail_mean = []
	for a in list_models:
		if t == a[:-6]:
			trial_dirname = dirname + a
			with open(trial_dirname + "/data_test_statistics.dat", "rb") as m:
				trial_stats = pickle.load(m)
			# extract the test info for every trial in the experiment
			tmp_errs = []
			tmp_stds = []
			tmp_fails = []
			for st in trial_stats:
				tmp_errs.append(st[2][0])
				tmp_stds.append(st[3][0])
				tmp_fails.append(st[4])
			# save as current seed
			seeds_error_mean.append(np.array(tmp_errs))
			seeds_error_std.append(np.array(tmp_stds))
			seeds_fail_mean.append(np.array(tmp_fails))

	# average seed for current model
	all_stats.append({	"model": t,
						"mean" : np.array([np.array(seeds_error_mean).mean(axis=0), 
									np.array(seeds_error_mean).std(axis=0)]),
						"std" :  np.array([np.array(seeds_error_std).mean(axis=0), 
									np.array(seeds_error_std).std(axis=0)]),
						"fails": np.array([np.array(seeds_fail_mean).mean(axis=0), 
									np.array(seeds_fail_mean).std(axis=0)])})


# plot all with std
# mpl.rcParams.update({'font.size': 12})
f, axarr = plt.subplots(3, sharex=True)
# plt.grid(which='major', axis='both')
f.set_size_inches(f.get_size_inches()[0]*2.5,f.get_size_inches()[1]*2)
# Plot error means
for a in all_stats:
	mean = np.array(a['mean'][0])
	std = np.array(a['mean'][1])
	lb = a['model'].split('_')[3]+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[-1]
	axarr[0].plot(mean, label=lb)
	axarr[0].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
axarr[0].set_ylabel('Test error mean') #(Euclidean distance)
axarr[0].legend()
axarr[0].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[0].set_xlim(0, 300)
# Plot error means
for a in all_stats:
	mean = np.array(a['std'][0])
	std = np.array(a['std'][1])
	lb = a['model'].split('_')[3]+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[-1]
	axarr[1].plot(mean, label=lb)
	axarr[1].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
axarr[1].set_ylabel('Test error std')
axarr[1].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[1].set_xlim(0, 300)
# Plot failed trials
for a in all_stats:
	mean = np.array(a['fails'][0])
	std = np.array(a['fails'][1])
	lb = a['model'].split('_')[3]+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[-1]
	axarr[2].plot(mean, label=lb)
	axarr[2].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
axarr[2].set_ylabel('Number of failed test')
axarr[2].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[2].set_xlim(0, 300)

plt.savefig(filename, format="svg")

plt.show()