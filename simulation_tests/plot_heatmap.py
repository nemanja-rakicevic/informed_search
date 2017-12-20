

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
	euc = ax.imshow(euclid_plot[t].T[::-1], cmap=cm.seismic, norm=norm1)	#, origin='upper'
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


###############################################################################
###############################################################################
# 2 link
# filenameinfo = '/home/robin/robin_lab/nemanja_scripts/hockey_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/img_paper_plot/low_error/TRIAL__sim2link_informed_res150_cov5_kernelRQ_sl0.01-seed1/data_test_statistics.dat'
# filenamerand = '/home/robin/robin_lab/nemanja_scripts/hockey_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/img_paper_plot/low_error/TRIAL__sim2link_random_res150_cov5_kernelRQ_sl0.01-seed1/data_test_statistics.dat'

# 5link
filenameinfo = '/home/robin/robin_lab/nemanja_scripts/hockey_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/5-LINK/selected_informed/TRIAL__sim5link_informed_res7_cov10_kernelRQ_sl0.001-seed1/data_test_statistics.dat'
filenamerand = '/home/robin/robin_lab/nemanja_scripts/hockey_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/5-LINK/selected_random/TRIAL__sim5link_random_res7_cov10_kernelRQ_sl0.001-seed1/data_test_statistics.dat'

with open(filenamerand, "rb") as m:
    tmp = pickle.load(m)

dl = tmp[28][1]
test_angles = np.arange(-65, 31, 5)
test_dist   = np.arange(5, 36, 5)
tmp_plot = np.zeros((len(test_dist), len(test_angles)))
for i in range(140):
	angle_s, dist_s = dl[i]['target_polar']
	if dl[i]['fail']>0:
		tmp_plot[len(test_dist) - np.argwhere(test_dist==int(dist_s))[0,0] - 1, len(test_angles) - np.argwhere(test_angles==int(angle_s))[0,0] - 1] = -1
	else:
		tmp_plot[len(test_dist) - np.argwhere(test_dist==int(dist_s))[0,0] - 1, len(test_angles) - np.argwhere(test_angles==int(angle_s))[0,0] - 1] = dl[i]['euclid_error']


euclid_plot = []
euclid_plot.append(tmp_plot)

euclid_plot = np.array(euclid_plot)


### Subplots for real test ###

mpl.rcParams.update({'font.size': 22})
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

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=None, dpi=150)
fig.set_size_inches(fig.get_size_inches()[0]*1,fig.get_size_inches()[1]*1.4)
# fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])

for t, ax in enumerate(axes):
	if t!=0:
		ax.set_xlabel('angles')
	ax.set_ylabel('distances')
	ax.set_yticks(yticks)
	ax.set_yticklabels([str(y) for y in test_dist[::-2]])
	# else:
	# if t==0:
	# 	ax.axes.get_xaxis().set_visible(False)
	# ax.set_title(titles[t])
	euc = ax.imshow(euclid_plot[t], cmap=cm.seismic, norm=norm1)	#, origin='upper'
	ax.set_xticks(xticks)
	ax.set_xticklabels([str(x) for x in test_angles[::-3]])


cb_ax = fig.add_axes([0.15, 0.1, 0.8, 0.02])

cbar = fig.colorbar(euc, cax=cb_ax, orientation='horizontal')
# cbar.ax.set_xlabel()
# cbar = fig.colorbar(euc, cax=cb_ax, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, euclid_plot[0].mean().round(2), euclid_plot[1].mean().round(2), euclid_plot[2].mean().round(2), euclid_plot[3].mean().round(2), euclid_plot.max().round(2)])

# cbar.set_ticks([-1, euclid_plot[0].mean().round(2), euclid_plot[1].mean().round(2),  euclid_plot.max().round(2)])
# cbar.set_ticklabels(['-1', 'info' , '\nrand', 'max\n({})'.format(euclid_plot.max().round(1) ) ]) #.set_rotation(90)
# cbar.set_ticklabels(['-1', 'info      \n({})      '.format(euclid_plot[0].mean().round(1)) , '      rand\n       ({})'.format((euclid_plot[1].mean().round(1))), 'max      \n({})      '.format(euclid_plot.max().round(1) ) ]) #.set_rotation(90)

cbar.set_ticks([-1, euclid_plot[0].mean().round(2),  euclid_plot.max().round(2)])
cbar.set_ticklabels(['-1', 'info\n({})'.format(euclid_plot[0].mean().round(0)), 'max      \n({})      '.format(euclid_plot.max().round(1) ) ]) #.set_rotation(90)

# cbar.ax.set_xticklabels(['-1', 'info', 'rand', 'inexp', 'exp', 'max'], rotation=90) #.set_rotation(90)
euc.set_clim(-1.001, euclid_plot.max()+.005)
# plt.savefig(savepath + "/img_test_plots_trial_#{}.svg".format(tr_num))
plt.show()









### Subplots for simulations ###
xticks = np.arange(0, len(test_angles), 2)
yticks = np.arange(0, len(test_dist), 2)
fig = plt.figure(figsize=(15,5), dpi=100)
fig.suptitle("Performance Error Plots (failed = -1)", fontsize=16)
# EUCLIDEAN ERROR
ax = plt.subplot("121")
ax.set_ylabel('distances')
ax.set_xlabel('angles')
ax.set_title("Euclidean error: "+str(errors_mean[0].round(2)))
euc = ax.imshow(euclid_plot, origin='upper', cmap=cm.seismic, norm=norm1)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([str(x) for x in test_angles[xticks-1][::-1]])
ax.set_yticklabels([str(y) for y in test_dist[yticks][::-1]])
cbar = plt.colorbar(euc, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, errors_mean[0].round(2), euclid_plot.max().round(2)])
cbar.ax.set_xticklabels(['-1', 'mean', 'max'])
euc.set_clim(-1.001, euclid_plot.max()+.005)
# POLAR ERROR
ax = plt.subplot("122")
ax.set_title("Polar coordinate error: "+str(errors_mean[1].round(2)))
ax.set_xlabel('angles')
sidf = ax.imshow(dist_plot, origin='upper', cmap=cm.seismic, norm=norm2)
# ax.set_xticks(np.arange(len(self.test_angles)))
# ax.set_yticks(np.arange(len(self.test_dist)))
# ax.set_xticklabels([str(x) for x in self.test_angles[::-1]])
# ax.set_yticklabels([str(y) for y in self.test_dist[::-1]])
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([str(x) for x in test_angles[xticks-1][::-1]])
ax.set_yticklabels([str(y) for y in test_dist[yticks][::-1]])
cbar = plt.colorbar(sidf, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, errors_mean[1].round(2), dist_plot.max().round(2)])
sidf.set_clim(-1.001, dist_plot.max()+.005)

plt.savefig(savepath + "/img_test_plots_trial_#{}.svg".format(tr_num))
if self.show_plots:
	plt.show()
else:
	plt.cla()