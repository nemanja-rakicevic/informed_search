

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pickle
import numpy as np

# Load data


###############################################################################
###############################################################################
# 2 link
# filenameinfo = '/home/robin/robin_lab/nemanja_scripts/hockey_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/img_paper_plot/low_error/TRIAL__sim2link_informed_res150_cov5_kernelRQ_sl0.01-seed1/data_test_statistics.dat'
# filenamerand = '/home/robin/robin_lab/nemanja_scripts/hockey_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/img_paper_plot/low_error/TRIAL__sim2link_random_res150_cov5_kernelRQ_sl0.01-seed1/data_test_statistics.dat'

# 5link

filename = []
# filenameinfo
filename.append('/home/robin/robin_lab/nemanja_scripts/phd_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/5-LINK/selected_informed/TRIAL__sim5link_informed_res7_cov10_kernelRQ_sl0.001-seed1/data_test_statistics.dat')
# filenamerand
filename.append('/home/robin/robin_lab/nemanja_scripts/phd_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/5-LINK/selected_random/TRIAL__sim5link_random_res7_cov10_kernelRQ_sl0.001-seed1/data_test_statistics.dat')
# filenameuidf
filename.append('/home/robin/robin_lab/nemanja_scripts/phd_projects/puckpass_hockey/simulation_tests/DATA/SIMULATION/5-LINK/selected_uidf/TRIAL__sim5link_uidf_res7_cov5_kernelRQ_sl0.001-seed1/data_test_statistics.dat')



### START ### 

TIMESTEP = 29  # 29 49  99 299

euclid_plot = []
for ff in filename:
	### STEP 1: run this for all filenames
	with open(ff, "rb") as m:
	    tmp = pickle.load(m)

	dl = tmp[TIMESTEP][1]
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

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=None, dpi=150)
fig.set_size_inches(fig.get_size_inches()[0]*1,fig.get_size_inches()[1]*1.4)
# fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1])
axes[0].set_title("Trial #{}".format(TIMESTEP+1))

for t, ax in enumerate(axes):
	if t==2:
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


cb_ax = fig.add_axes([0.12, 0.15, 0.8, 0.02])

cbar = fig.colorbar(euc, cax=cb_ax, orientation='horizontal')
# cbar.ax.set_xlabel()
# cbar = fig.colorbar(euc, cax=cb_ax, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, euclid_plot[0].mean().round(2), euclid_plot[1].mean().round(2), euclid_plot[2].mean().round(2), euclid_plot[3].mean().round(2), euclid_plot.max().round(2)])


### option 1
cbar.set_ticks([-1, euclid_plot[0].mean().round(2), euclid_plot[2].mean().round(2), euclid_plot[1].mean().round(2),  euclid_plot.max().round(2)])
# cbar.set_ticklabels(['-1', 'info' , '\nuidf', '\nrand', 'max\n({})'.format(euclid_plot.max().round(1) ) ]) #.set_rotation(90)
# cbar.set_ticklabels(['-1', 'info      \n({})      '.format(euclid_plot[0].mean().round(1)) , '      uidf\n       ({})'.format((euclid_plot[2].mean().round(1))), '      rand\n       ({})'.format((euclid_plot[1].mean().round(1))), 'max      \n({})      '.format(euclid_plot.max().round(1) ) ]) #.set_rotation(90)

# cbar.set_ticklabels(['-1', 'info' , '\nuidf', '\nrand', 'max\n({})'.format(euclid_plot.max().round(1) ) ]) #.set_rotation(90)
# cbar.set_ticklabels(['-1', 'info       \n({})       '.format(euclid_plot[0].mean().round(1)) , '       uidf\n       ({})'.format((euclid_plot[2].mean().round(1))), '       rand\n       ({})'.format((euclid_plot[1].mean().round(1))), 'max      \n({})      '.format(euclid_plot.max().round(1) ) ]) #.set_rotation(90)

# ### option 2
cbar.set_ticks([-1, euclid_plot[0].mean().round(2), euclid_plot[2].mean().round(2),  euclid_plot.max().round(2)])
cbar.set_ticklabels(['-1', '      info\n         ({})'.format(euclid_plot[0].mean().round(1)), 'uidf      \n({})         '.format(euclid_plot[2].mean().round(1)), 'max      \n({})      '.format(euclid_plot.max().round(1) ) ]) #.set_rotation(90)

# cbar.ax.set_xticklabels(['-1', 'info', 'rand', 'inexp', 'exp', 'max'], rotation=90) #.set_rotation(90)
euc.set_clim(-1.001, euclid_plot.max()+.005)
# plt.savefig(savepath + "/img_test_plots_trial_#{}.svg".format(tr_num))
plt.show()







###############################################################################
###############################################################################


# ### Subplots for simulations ###
# xticks = np.arange(0, len(test_angles), 2)
# yticks = np.arange(0, len(test_dist), 2)
# fig = plt.figure(figsize=(15,5), dpi=100)
# fig.suptitle("Performance Error Plots (failed = -1)", fontsize=16)
# # EUCLIDEAN ERROR
# ax = plt.subplot("121")
# ax.set_ylabel('distances')
# ax.set_xlabel('angles')
# ax.set_title("Euclidean error: "+str(errors_mean[0].round(2)))
# euc = ax.imshow(euclid_plot, origin='upper', cmap=cm.seismic, norm=norm1)
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
# ax.set_xticklabels([str(x) for x in test_angles[xticks-1][::-1]])
# ax.set_yticklabels([str(y) for y in test_dist[yticks][::-1]])
# cbar = plt.colorbar(euc, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, errors_mean[0].round(2), euclid_plot.max().round(2)])
# cbar.ax.set_xticklabels(['-1', 'mean', 'max'])
# euc.set_clim(-1.001, euclid_plot.max()+.005)
# # POLAR ERROR
# ax = plt.subplot("122")
# ax.set_title("Polar coordinate error: "+str(errors_mean[1].round(2)))
# ax.set_xlabel('angles')
# sidf = ax.imshow(dist_plot, origin='upper', cmap=cm.seismic, norm=norm2)
# # ax.set_xticks(np.arange(len(self.test_angles)))
# # ax.set_yticks(np.arange(len(self.test_dist)))
# # ax.set_xticklabels([str(x) for x in self.test_angles[::-1]])
# # ax.set_yticklabels([str(y) for y in self.test_dist[::-1]])
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
# ax.set_xticklabels([str(x) for x in test_angles[xticks-1][::-1]])
# ax.set_yticklabels([str(y) for y in test_dist[yticks][::-1]])
# cbar = plt.colorbar(sidf, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, errors_mean[1].round(2), dist_plot.max().round(2)])
# sidf.set_clim(-1.001, dist_plot.max()+.005)

# plt.savefig(savepath + "/img_test_plots_trial_#{}.svg".format(tr_num))
# if self.show_plots:
# 	plt.show()
# else:
# 	plt.cla()