"""
Author: Nemanja Rakicevic
Date  : January 2018
Description:
            Plot heatmaps of multiple models

"""


import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pickle
import numpy as np


### Load data
filename = []
# filenameinfo
filename.append('/home/robin/robin_lab/nemanja_scripts/phd_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/5-LINK/selected_informed/TRIAL__sim5link_informed_res7_cov10_kernelRQ_sl0.001-seed1/data_test_statistics.dat')
# filenamerand
filename.append('/home/robin/robin_lab/nemanja_scripts/phd_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/5-LINK/selected_random/TRIAL__sim5link_random_res7_cov10_kernelRQ_sl0.001-seed1/data_test_statistics.dat')
# filenameuidf
filename.append('/home/robin/robin_lab/nemanja_scripts/phd_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/5-LINK/selected_uidf/TRIAL__sim5link_uidf_res7_cov5_kernelRQ_sl0.001-seed1/data_test_statistics.dat')
# filenameBO
filename.append('/home/robin/robin_lab/nemanja_scripts/phd_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/TRIAL__sim5link_BO_res7_cov5_kernelRQ_sl0.01-seed1/data_test_statistics.dat')


_TIMESTEP = 29  # 29 49  99 299

### STEP 1
euclid_plot = []
for ff in filename:
    ### STEP 1: run this for all filenames
    with open(ff, "rb") as m:
        tmp = pickle.load(m)

    dl = tmp[_TIMESTEP][1]
    test_angles = np.arange(-65, 31, 5)
    test_dist   = np.arange(5, 36, 5)
    tmp_plot = np.zeros((len(test_dist), len(test_angles)))
    for i in range(140):
        angle_s, dist_s = dl[i]['target_polar']
        if dl[i]['fail']>0:
            tmp_plot[len(test_dist) - np.argwhere(test_dist==int(dist_s))[0,0] - 1, len(test_angles) - np.argwhere(test_angles==int(angle_s))[0,0] - 1] = -1
        else:
            tmp_plot[len(test_dist) - np.argwhere(test_dist==int(dist_s))[0,0] - 1, len(test_angles) - np.argwhere(test_angles==int(angle_s))[0,0] - 1] = dl[i]['euclid_error']

    euclid_plot.append(tmp_plot)

### STEP 2 
euclid_plot = np.array(euclid_plot)


### STEP 3
mpl.rcParams.update({'font.size': 20})
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
norm1 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=euclid_plot.max())

# titles = ["Informed\nsearch", "Random\nsearch", "Inexperienced\nvolunteer", "Experienced\nvolunteer"]
xticks = np.arange(0, len(test_angles), 3)
yticks = np.arange(0, len(test_dist), 2)

fig, axes = plt.subplots(nrows=len(filename), ncols=1, figsize=None, dpi=150)
fig.set_size_inches(fig.get_size_inches()[0]*1,fig.get_size_inches()[1]*1.4)
# fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])
axes[0].set_title("Trial #{}".format(_TIMESTEP+1))

for t, ax in enumerate(axes):
    if t==len(filename)-1:
        ax.set_xlabel('angles')
    ax.set_ylabel('distances')
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in test_dist[::-2]])
    # else:
    # if t==0:
    #   ax.axes.get_xaxis().set_visible(False)
    # ax.set_title(titles[t])
    euc = ax.imshow(euclid_plot[t], cmap=cm.seismic, norm=norm1)    #, origin='upper'
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in test_angles[::-3]])

# font 15
# cb_ax = fig.add_axes([0.13, 0.1, 0.77, 0.02])
# font 22
cb_ax = fig.add_axes([0.15, 0.14, 0.72, 0.02])

cbar = fig.colorbar(euc, cax=cb_ax, orientation='horizontal')
# cbar.ax.set_xlabel()
# cbar = fig.colorbar(euc, cax=cb_ax, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, euclid_plot[0].mean().round(2), euclid_plot[1].mean().round(2), euclid_plot[2].mean().round(2), euclid_plot[3].mean().round(2), euclid_plot.max().round(2)])

### option 1
# cbar.set_ticks([-1, euclid_plot[0].mean().round(2), euclid_plot[2].mean().round(2), euclid_plot[1].mean().round(2),  euclid_plot.max().round(2)])
tick_vals = [ e.mean().round(2) for e in euclid_plot if e.mean().round(2)>0]
tick_vals.insert(0, -1)
tick_vals.append(euclid_plot.max().round(2))

print(tick_vals)

cbar.set_ticks(tick_vals)
if len(tick_vals)-2==len(filename):
    # cbar.set_ticklabels(['-1', \
    #                      'info',\
    #                      'rand',\
    #                      'uidf',\
    #                      'BO',\
    #                      'max'])
    cbar.set_ticklabels(['-1', \
                         '\ninfo ({})'.format(euclid_plot[0].mean().round(1)) , \
                         'rand ({})'.format((euclid_plot[1].mean().round(1))), \
                         'uidf ({})'.format((euclid_plot[2].mean().round(1))), \
                         'BO ({})'.format((euclid_plot[3].mean().round(1))), \
                         'max\n({})'.format(euclid_plot.max().round(1) ) ])
    cbar.ax.set_xticklabels(['-1', 'info', 'rand', 'uidf', 'BO', 'max'], rotation=-90) 
else:
    # cbar.set_ticklabels(['-1', \
    #                      'info',\
    #                      'uidf',\
    #                      'BO',\
    #                      'max'])
    cbar.set_ticklabels(['-1', \
                         'info ({})'.format(euclid_plot[0].mean().round(1)) , \
                         '\nuidf ({})'.format((euclid_plot[2].mean().round(1))), \
                         'BO ({})\n'.format((euclid_plot[3].mean().round(1))), \
                         'max\n({})'.format(euclid_plot.max().round(1) ) ])
    cbar.ax.set_xticklabels(['-1', 'info',  'uidf', 'BO','max','max'], rotation=-90) 

#.set_rotation(90)
euc.set_clim(-1.001, euclid_plot.max()+.005)
# plt.savefig(savepath + "/img_test_plots_trial_#{}.svg".format(tr_num))
plt.show()
