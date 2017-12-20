
import os
import numpy as np
import util_modelling as umodel
import util_experiment as uexp
import util_testing as utest
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter


# dirname = './DATA/SIMULATION/'

dirname = './DATA/SIMULATION/5-LINK/img_paper/'
# dirname = './DATA/SIMULATION/5-LINK/selected_informed/'

# dirname = './DATA/SIMULATION/img_paper_plot/low_error/'

# Load all tests statistics
# filt = 'TRIAL__informed_res150_cov20_'
# filt = ['informed', 'RQ','cov']
filt = ['_', 'kernel', 'cov']
CUTOFF = 300
FNAME = 24
offset = 0
filename = dirname+'PLOTS_{}_kernel{}_{}_ttt.svg'.format(filt[0], filt[1], filt[2])

list_models = sorted([d for d in os.listdir(dirname) if all(s in d for s in filt) and d[-3:]!='svg'])
# list_models = sorted([d for d in os.listdir(dirname) if d[0:len(filt)]==filt])
list_unique = np.unique([d[:FNAME] for d in list_models])



all_stats = []
for idx, t in enumerate(list_unique):
	seeds_error_mean = []
	seeds_error_std = []
	seeds_fail_mean = []
	for a in list_models:
		if t == a[:FNAME]:
			trial_dirname = dirname + a
			with open(trial_dirname + "/data_test_statistics.dat", "rb") as m:
				trial_stats = pickle.load(m)
			# extract the test info for every trial in the experiment
			tmp_errs = []
			tmp_stds = []
			tmp_fails = []
			for st in trial_stats:
				tmp_errs.append(st[2+offset][0])
				tmp_stds.append(st[3+offset][0])
				tmp_fails.append(st[4+offset])
			# save as current seed
			seeds_error_mean.append(np.array(tmp_errs)[:CUTOFF])
			seeds_error_std.append(np.array(tmp_stds)[:CUTOFF])
			seeds_fail_mean.append(np.array(tmp_fails)[:CUTOFF])

	# average seed for current model
	all_stats.append({	"model": t,
						"mean" : np.array([np.array(seeds_error_mean).mean(axis=0), 
									np.array(seeds_error_mean).std(axis=0)]),
						"std" :  np.array([np.array(seeds_error_std).mean(axis=0), 
									np.array(seeds_error_std).std(axis=0)]),
						"fails": np.array([np.array(seeds_fail_mean).mean(axis=0), 
									np.array(seeds_fail_mean).std(axis=0)])
						})


clrand = ['red','maroon','deeppink']
clinfo = ['blue', 'olive', 'indigo', 'teal', 'purple']

# plot all with std
mpl.rcParams.update({'font.size': 12})
f, axarr = plt.subplots(3, sharex=True)
# plt.grid(which='major', axis='both')
# f.set_size_inches(f.get_size_inches()[0]*1.5,f.get_size_inches()[1]*3)
f.set_size_inches(f.get_size_inches()[0]*3,f.get_size_inches()[1]*2)

# PLOT ERROR MEANS
cc =[0,0]
for i,a in enumerate(all_stats):
	mean = np.array(a['mean'][0])
	std = np.array(a['mean'][1])
	if len(a['model']) < 25:
		lb = a['model'].split('_')[3]+' models'#+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[6]+' '+a['model'].split('_')[-1]
		axarr[0].plot(mean, label=lb)
		axarr[0].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)#, label=lb)
	else:
		lb = a['model'].split('_')[3]+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[6]+' '+a['model'].split('_')[-1]
		if a['model'].split('_')[3] == 'informed':
			c = clinfo[cc[0]]
			cc[0]+=1
		else:
			c = clrand[cc[1]]
			cc[1]+=1
		axarr[0].plot(mean, label=lb, color=c)
		# axarr[0].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
axarr[0].set_ylabel('Test error mean') #(Euclidean distance)
axarr[0].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[0].set_xlim(0, CUTOFF)
axarr[0].set_ylim(0, 21)

# PLOT ERROR STDs
cc =[0,0]
for i,a in enumerate(all_stats):
	mean = np.array(a['std'][0])
	std = np.array(a['std'][1])
	if len(a['model']) < 25:
		lb = a['model'].split('_')[3]+' models'#+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[6]+' '+a['model'].split('_')[-1]
		axarr[1].plot(mean, label=lb)
		axarr[1].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)#, label=lb)
	else:
		lb = a['model'].split('_')[3]+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[6]+' '+a['model'].split('_')[-1]
		if a['model'].split('_')[3] == 'informed':
			c = clinfo[cc[0]]
			cc[0]+=1
		else:
			c = clrand[cc[1]]
			cc[1]+=1
		axarr[1].plot(mean, label=lb, color=c)
		# axarr[0].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
axarr[1].set_ylabel('Test error std')
axarr[1].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[1].set_xlim(0, CUTOFF)
axarr[1].set_ylim(0, 10.5)
axarr[1].yaxis.set_major_formatter(FormatStrFormatter('%i'))

# PLOT FAILED TRIAL COUNT
cc =[0,0]
for i,a in enumerate(all_stats):
	mean = np.array(a['fails'][0])
	std = np.array(a['fails'][1])
	if len(a['model']) < 25:
		lb = a['model'].split('_')[3]+' models'#+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[6]+' '+a['model'].split('_')[-1]
		axarr[2].plot(mean, label=lb)
		axarr[2].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)#, label=lb)
	else:
		lb = a['model'].split('_')[3]+' '+a['model'].split('_')[5]+' '+a['model'].split('_')[6]+' '+a['model'].split('_')[-1]
		if a['model'].split('_')[3] == 'informed':
			c = clinfo[cc[0]]
			cc[0]+=1
		else:
			c = clrand[cc[1]]
			cc[1]+=1
		axarr[2].plot(mean, label=lb, color=c)
		# axarr[0].fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
axarr[2].set_ylabel('Number of failed tests')	
axarr[2].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[2].set_xlim(0, CUTOFF)
axarr[2].set_ylim(-1, 150)
	

handles, labels = axarr[2].get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))
# axarr[2].legend(loc='lower left', bbox_to_anchor=(0.405,0.32))  #font16
axarr[2].legend(handles, labels,loc='lower left', bbox_to_anchor=(0.8,0.54))		# font14 bbox_to_anchor=(0.675,0.165)


# plt.tight_layout(pad=0.1, w_pad=0.90, h_pad=0.90)

# SAVE and SHOW
plt.savefig(filename, format="svg")
plt.show()