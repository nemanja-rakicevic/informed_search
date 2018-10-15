

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pickle
import numpy as np

# Load data

test_angles = [0, 10, 15, 20]
test_dist = [100, 120, 150, 175, 200, 250, 300]

euclid_plot = []
euclid_plot.append(model_err)
euclid_plot.append(rand_err.mean(axis=0))
euclid_plot.append(human_err.mean(axis=0))
euclid_plot.append(pro_err.mean(axis=0))

euclid_plot = np.array(euclid_plot)

# Plot data
mpl.rcParams.update({'font.size': 16})
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# norm1 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=50)
# norm2 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=50)
norm1 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=euclid_plot.max())
# norm2 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=dist_plot.max())


### Subplots for real test ###
titles = ["Informed\nsearch", "Random\nsearch", "Inexperienced\nvolunteer", "Experienced\nvolunteer"]
xticks = np.arange(0, len(test_angles), 1)
yticks = np.arange(0, len(test_dist), 1)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=None, dpi=200)
fig.set_size_inches(fig.get_size_inches()[0]*1.5,fig.get_size_inches()[1]*1.2)
# fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])

for t,ax in enumerate(axes):
    if t==0:
        ax.set_ylabel('distances')
        ax.set_xlabel('angles')
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(y) for y in test_dist[::-1]])
    else:
        ax.axes.get_yaxis().set_visible(False)
    ax.set_title(titles[t])
    euc = ax.imshow(euclid_plot[t].T[::-1], cmap=cm.seismic, norm=norm1)    #, origin='upper'
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in test_angles[::-1]])


cb_ax = fig.add_axes([0.15, 0.1, 0.72, 0.03])

cbar = fig.colorbar(euc, cax=cb_ax, orientation='horizontal')
# cbar.ax.set_xlabel()
cbar.set_ticks([-1, euclid_plot[0].mean().round(2), euclid_plot[1].mean().round(2), euclid_plot[2].mean().round(2), euclid_plot[3].mean().round(2), euclid_plot.max().round(2)])
# cbar = fig.colorbar(euc, cax=cb_ax, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, euclid_plot[0].mean().round(2), euclid_plot[1].mean().round(2), euclid_plot[2].mean().round(2), euclid_plot[3].mean().round(2), euclid_plot.max().round(2)])
cbar.set_ticklabels(['-1', 'info', 'rand', '\ninexp', 'exp', 'max\n({})'.format(euclid_plot.max().round(2))]) #.set_rotation(90)
# cbar.ax.set_xticklabels(['-1', 'info', 'rand', 'inexp', 'exp', 'max'], rotation=90) #.set_rotation(90)
euc.set_clim(-1.001, euclid_plot.max()+.005)
# plt.savefig(savepath + "/img_test_plots_trial_#{}.svg".format(tr_num))
plt.show()