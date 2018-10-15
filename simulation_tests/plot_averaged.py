"""
Author: Nemanja Rakicevic
Date  : January 2018
Description:
            Plot all models in the folder or filtered using keyworkds, 
            averaged over their hyperparameters
            Plots the 3 graphs:
            - test error mean
            - test error std
            - number of failed trials

"""

import os
import numpy as np
import util_modelling as umodel
import util_experiment as uexp
import util_testing as utest
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

# dirname = './DATA/SIMULATION/NEW_STUFF/'
dirname = './DATA/SIMULATION/5-LINK/img_paper/'

# Load all tests statistics
# filt = 'TRIAL__sim5link_uidf_res7_cov5_kernelRQsl0.1'
# filt = ['uidf', 'SE','cov10']
# filt = ['_', 'kernelRQ', 'cov5']
filt = ['']
CUTOFF = 300
FNAME = 17   # -6   24
### for new data
offset = 0

list_models = sorted([d for d in os.listdir(dirname) if all(s in d for s in filt) and d[-3:]!='svg'])
# list_models = sorted([d for d in os.listdir(dirname) if d[0:len(filt)]==filt])
list_unique = np.unique([d[:FNAME] for d in list_models])

filename = dirname+'PLOTS_comparison_{}.svg'.format('_'.join(filt))

### PROCESS STORED DATA
all_stats = []
for idx, t in enumerate(list_unique):
    seeds_error_mean = []
    seeds_error_std = []
    seeds_fail_mean = []
    for a in list_models:
        if t == a[:FNAME]:
            trial_dirname = dirname + a
            if 'rand' in a or 'informed' in a:
                offset = 0
            else:
                offset = 2
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
            tmp_name = a

    # average seed for current model
    all_stats.append({  "model": tmp_name.split('_')[3],
                        "mean" : np.array([np.array(seeds_error_mean).mean(axis=0), 
                                    np.array(seeds_error_mean).std(axis=0)]),
                        "std" :  np.array([np.array(seeds_error_std).mean(axis=0), 
                                    np.array(seeds_error_std).std(axis=0)]),
                        "fails": np.array([np.array(seeds_fail_mean).mean(axis=0), 
                                    np.array(seeds_fail_mean).std(axis=0)])
                        })



### PLOT DATA
lw = 1
## 5-link colors
clrand = ['red','maroon','deeppink']
# clrand = ['blue', 'olive', 'indigo', 'teal', 'purple', 'maroon','deeppink', 'red', 'olive', 'indigo', 'teal', 'purple', 'maroon','deeppink', 'red']
clinfo = ['blue', 'olive', 'indigo', 'teal', 'purple', 'maroon','deeppink', 'red', 'olive', 'indigo', 'teal', 'purple', 'maroon','deeppink', 'red']

# plot all with std
mpl.rcParams.update({'font.size': 18})
f, axarr = plt.subplots(3, sharex=True)
# plt.grid(which='major', axis='both')
## 5-link
f.set_size_inches(f.get_size_inches()[0]*3,f.get_size_inches()[1]*2)

# PLOT ERROR MEANS
cc =[0,0]
for i,a in enumerate(all_stats):
    mean = np.array(a['mean'][0])
    std = np.array(a['mean'][1])
    lb = a['model']+'models'
    axarr[0].plot(mean, label=lb, linewidth=lw)
    axarr[0].fill_between(range(len(mean)), (mean-std).clip(0), mean+std, alpha=0.5)#, label=lb)
axarr[0].set_ylabel('Test error mean', labelpad=14) #(Euclidean distance)
axarr[0].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[0].set_xlim(0, CUTOFF)
axarr[0].set_ylim(0, 24)
axarr[0].yaxis.set_major_locator(MultipleLocator(5))

# PLOT ERROR STDs
cc =[0,0]
for i,a in enumerate(all_stats):
    mean = np.array(a['std'][0])
    std = np.array(a['std'][1])
    lb = a['model']+' models'
    axarr[1].plot(mean, label=lb, linewidth=lw)
    axarr[1].fill_between(range(len(mean)), (mean-std).clip(0), mean+std, alpha=0.5)#, label=lb)
axarr[1].set_ylabel('Test error std', labelpad=14)
axarr[1].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[1].set_xlim(0, CUTOFF)
axarr[1].set_ylim(0, 11)
axarr[1].yaxis.set_major_formatter(FormatStrFormatter('%i'))
axarr[1].yaxis.set_major_locator(MaxNLocator(5))

# PLOT FAILED TRIAL COUNT
cc =[0,0]
for i,a in enumerate(all_stats):
    mean = np.array(a['fails'][0])
    std = np.array(a['fails'][1])
    lb = a['model']+' models'
    axarr[2].plot(mean, label=lb, linewidth=lw)
    axarr[2].fill_between(range(len(mean)), (mean-std).clip(0), mean+std, alpha=0.5)#, label=lb)
axarr[2].set_ylabel('Number of failed tests')   
axarr[2].grid(color='gray', linestyle=':', linewidth=0.8)
axarr[2].set_xlim(0, CUTOFF)
axarr[2].set_ylim(-1, 160)
axarr[2].set_xlabel('Trial number') 
    

handles, labels = axarr[2].get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0], reverse=True))

### ADD LEGEND
# # 5-link
axarr[2].legend(handles, labels,loc='lower left', bbox_to_anchor=(0.66,0.5))        # font14 bbox_to_anchor=(0.675,0.165)
# print(labels)
### MAKE IT TIGHT
# plt.tight_layout(pad=0.1, w_pad=0.90, h_pad=0.90)

### SAVE and SHOW
plt.savefig(filename, format="svg")
plt.show()