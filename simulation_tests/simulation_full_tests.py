
import numpy as np
import itertools
import pickle
import util_modelling as umodel
import util_experiment as uexp
import matplotlib.pyplot as plt
from matplotlib import cm

# INITIALISE MODEL
print("INITIALISING MODEL\n")
experiment = uexp.SimulationExperiment(display=False, display_steps=100)
model      = umodel.InformedModel(experiment.parameter_list, experiment.type, test=True)
model.loadModel()

# DEFINE TESTING POINTS
test_angles = np.arange(-60, 26, 5)
test_dist   = np.arange(5, 36, 5)
test_cases  = np.array([xs for xs in itertools.product(test_angles, test_dist)])
dist_plot = np.zeros((len(test_dist), len(test_angles)))
euclid_plot = np.zeros((len(test_dist), len(test_angles)))

# RUN TESTS
input("\nRUN {} TESTS?".format(len(test_cases)))
statistics = []
for t in range(len(test_cases)):
    # Input goal position
    angle_s, dist_s = test_cases[t]
    print("\nTEST #{} > angle, distance: ({},{})".format(t+1, angle_s, dist_s))
    # Generate movement parameter vector
    trial_coords, trial_params = model.testModel(float(angle_s), float(dist_s), verbose=0)
    trial_info = experiment.executeTrial(0, trial_coords, trial_params, test=[float(angle_s), float(dist_s)])
    # Compile test statistics
    dist_error = np.sqrt(np.sum((trial_info['ball_polar'] - test_cases[t])**2))
    euclid_error = np.sqrt(np.sum(trial_info['observations'][-1][-2:]**2))

    statistics.append({ 'trial_num':    t+1,
                        'target_polar': test_cases[t],
                        'ball_polar':   trial_info['ball_polar'],
                        'fail':         trial_info['fail'],
                        'dist_error':   dist_error,
                        'euclid_error': euclid_error })
    if trial_info['fail']>0:
        dist_plot[len(test_dist) - np.argwhere(test_dist==int(dist_s))[0,0] - 1, len(test_angles) - np.argwhere(test_angles==int(angle_s))[0,0] - 1] = -1
        euclid_plot[len(test_dist) - np.argwhere(test_dist==int(dist_s))[0,0] - 1, len(test_angles) - np.argwhere(test_angles==int(angle_s))[0,0] - 1] = -1
    else:
        dist_plot[len(test_dist) - np.argwhere(test_dist==int(dist_s))[0,0] - 1, len(test_angles) - np.argwhere(test_angles==int(angle_s))[0,0] - 1] = dist_error
        euclid_plot[len(test_dist) - np.argwhere(test_dist==int(dist_s))[0,0] - 1, len(test_angles) - np.argwhere(test_angles==int(angle_s))[0,0] - 1] = euclid_error

# Calculate error
num_fails  = len([1 for x in statistics if x['fail']>0])
errors_all = np.array([ [x['euclid_error'], x['dist_error']] for x in statistics])
errors_mean = errors_all.mean(axis=0)
errors_std  = errors_all.std(axis=0)

print("\nTESTING COMPLETE.\nPERFORMANCE ERRORS\t> Euclidean\t(mean: {}, std:  {})".format(errors_mean[0].round(2), errors_std[0].round(2)))
print("\t\t\t> Polar\t\t(mean: {}, std:  {})".format(errors_mean[1].round(2), errors_std[1].round(2)))
print("\t\t\t> failed: {}".format(num_fails))

print("saving results...")
with open(model.trial_dirname + "/data_test_statistics.dat", "wb") as m:
    pickle.dump([statistics, errors_mean, errors_std, num_fails], m, protocol=pickle.HIGHEST_PROTOCOL)

# PLOT RESULTS
import matplotlib as mpl
class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
norm1 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=50)
norm2 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=50)
fig = plt.figure(figsize=(15,5), dpi=100)
fig.suptitle("Performance Error Plots (failed = -1)", fontsize=16)
ax = plt.subplot("121")
ax.set_ylabel('distances')
ax.set_xlabel('angles')
ax.set_title("Euclidean error: "+str(errors_mean[0].round(2)))
euc = ax.imshow(euclid_plot, origin='upper', cmap=cm.seismic, norm=norm1)
ax.set_xticks(np.arange(len(test_angles)))
ax.set_yticks(np.arange(len(test_dist)))
ax.set_xticklabels([str(x) for x in test_angles[::-1]])
ax.set_yticklabels([str(y) for y in test_dist[::-1]])
cbar = plt.colorbar(euc, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, 10, euclid_plot.max(), 50])
cbar.ax.set_xticklabels(['-1', '10', 'max', '50'])
euc.set_clim(-1.001, 50+.001)
ax = plt.subplot("122")
ax.set_title("Polar coordinate error: "+str(errors_mean[1].round(2)))
ax.set_xlabel('angles')
sidf = ax.imshow(dist_plot, origin='upper', cmap=cm.seismic, norm=norm2)
ax.set_xticks(np.arange(len(test_angles)))
ax.set_yticks(np.arange(len(test_dist)))
ax.set_xticklabels([str(x) for x in test_angles[::-1]])
ax.set_yticklabels([str(y) for y in test_dist[::-1]])
cbar = plt.colorbar(sidf, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, 10, dist_plot.max(), 50])
sidf.set_clim(-1.001, 50+.001)
plt.savefig(model.trial_dirname+"/img_test_plots.svg")
plt.show()