
import numpy as np
import numpy as np
import itertools
import pickle
import multiprocessing as mp
import logging


logger = logging.getLogger(__name__)


class FullTest:
  def __init__(self, experiment, model, show_plots=False, verbose=False):
    self.show_plots = show_plots
    self.verbose = verbose
    self.results_list = []
    self.test_angles = np.arange(-65, 31, 5)
    self.test_dist   = np.arange(5, 36, 5)
    self.test_cases  = np.array([xs for xs in itertools.product(self.test_angles, self.test_dist)])
    self.num_cpu = 4#mp.cpu_count() - 1
    self.model = model
    self.experiment = experiment

# Try to make parallel
  # def doTest(self, test_tuple):
  #   angle_s, dist_s = test_tuple
  #   trial_coords, trial_params = self.model.testModel(float(angle_s), float(dist_s), verbose=self.verbose)
  #   trial_info = self.experiment.executeTrial(0, trial_coords, trial_params, test=[float(angle_s), float(dist_s)])
  #   return trial_info

  # def runFullTests(self, tr_num, save_progress=True, heatmap=True):
  #   statistics = []
  #   dist_plot = np.zeros((len(self.test_dist), len(self.test_angles)))
  #   euclid_plot = np.zeros((len(self.test_dist), len(self.test_angles)))

  #   with mp.Pool(processes=self.num_cpu) as pool:
  #     res_pool = pool.map(self.doTest, self.test_cases)

  #   for t, tst in enumerate(res_pool):
  #     # Compile test statistics
  #     dist_error = np.sqrt(np.sum((tst['ball_polar'] - self.test_cases[t])**2))
  #     euclid_error = np.sqrt(np.sum(tst['observations'][-1][-2:]**2))

  #     statistics.append({ 'trial_num':    t+1,
  #               'target_polar': self.test_cases[t],
  #               'ball_polar':   tst['ball_polar'],
  #               'fail':         tst['fail'],
  #               'dist_error':   dist_error,
  #               'euclid_error': euclid_error })
  #     if tst['fail']>0:
  #       dist_plot[len(self.test_dist) - np.argwhere(self.test_dist==int(dist_s))[0,0] - 1, len(self.test_angles) - np.argwhere(self.test_angles==int(angle_s))[0,0] - 1] = -1
  #       euclid_plot[len(self.test_dist) - np.argwhere(self.test_dist==int(dist_s))[0,0] - 1, len(self.test_angles) - np.argwhere(self.test_angles==int(angle_s))[0,0] - 1] = -1
  #     else:
  #       dist_plot[len(self.test_dist) - np.argwhere(self.test_dist==int(dist_s))[0,0] - 1, len(self.test_angles) - np.argwhere(self.test_angles==int(angle_s))[0,0] - 1] = dist_error
  #       euclid_plot[len(self.test_dist) - np.argwhere(self.test_dist==int(dist_s))[0,0] - 1, len(self.test_angles) - np.argwhere(self.test_angles==int(angle_s))[0,0] - 1] = euclid_error

  #   # Calculate error
  #   num_fails  = len([1 for x in statistics if x['fail']>0])
  #   errors_all = np.array([ [x['euclid_error'], x['dist_error']] for x in statistics])
  #   errors_mean = errors_all.mean(axis=0)
  #   errors_std  = errors_all.std(axis=0)
  #   # Save statistics
  #   self.results_list.append([tr_num, statistics, euclid_plot, dist_plot, errors_mean, errors_std, num_fails])
  #   if save_progress:
  #     self.saveResults(model.trial_dirname)
  #   # Plot heatmaps
  #   heatmap = False
  #   if heatmap:
  #     self.plotResults(model.trial_dirname, tr_num, euclid_plot, dist_plot, errors_mean)



  def runFullTests(self, tr_num, save_progress=True, heatmap=True):
    statistics = []
    dist_plot = np.zeros((len(self.test_dist), len(self.test_angles)))
    euclid_plot = np.zeros((len(self.test_dist), len(self.test_angles)))

    
    for t in range(len(self.test_cases)):
      # Input goal position
      angle_s, dist_s = self.test_cases[t]
      if self.verbose:
        print("\nTRIAL {}\nTEST # {} > angle, distance: ({},{})".format(tr_num, t+1, angle_s, dist_s))
      # Generate movement parameter vector
      trial_coords, trial_params = self.model.testModel(float(angle_s), float(dist_s), verbose=self.verbose)
      trial_info = self.experiment.executeTrial(0, trial_coords, trial_params, test=[float(angle_s), float(dist_s)])
      # Compile test statistics
      dist_error = np.sqrt(np.sum((trial_info['ball_polar'] - self.test_cases[t])**2))
      euclid_error = np.sqrt(np.sum(trial_info['observations'][-1][-2:]**2))

      statistics.append({ 'trial_num':    t+1,
                'target_polar': self.test_cases[t],
                'ball_polar':   trial_info['ball_polar'],
                'fail':         trial_info['fail'],
                'dist_error':   dist_error,
                'euclid_error': euclid_error })
      if trial_info['fail']>0:
        dist_plot[len(self.test_dist) - np.argwhere(self.test_dist==int(dist_s))[0,0] - 1, len(self.test_angles) - np.argwhere(self.test_angles==int(angle_s))[0,0] - 1] = -1
        euclid_plot[len(self.test_dist) - np.argwhere(self.test_dist==int(dist_s))[0,0] - 1, len(self.test_angles) - np.argwhere(self.test_angles==int(angle_s))[0,0] - 1] = -1
      else:
        dist_plot[len(self.test_dist) - np.argwhere(self.test_dist==int(dist_s))[0,0] - 1, len(self.test_angles) - np.argwhere(self.test_angles==int(angle_s))[0,0] - 1] = dist_error
        euclid_plot[len(self.test_dist) - np.argwhere(self.test_dist==int(dist_s))[0,0] - 1, len(self.test_angles) - np.argwhere(self.test_angles==int(angle_s))[0,0] - 1] = euclid_error

    # Calculate error
    num_fails  = len([1 for x in statistics if x['fail']>0])
    errors_all = np.array([ [x['euclid_error'], x['dist_error']] for x in statistics])
    errors_mean = errors_all.mean(axis=0)
    errors_std  = errors_all.std(axis=0)
    # Save statistics
    self.results_list.append([tr_num, statistics, euclid_plot, dist_plot, errors_mean, errors_std, num_fails])
    if save_progress:
      self.saveResults(self.model.trial_dirname)
    # Plot heatmaps
    # heatmap = False
    if heatmap:
      self.plotResults(self.model.trial_dirname, tr_num, euclid_plot, dist_plot, errors_mean)


  def saveResults(self, savepath):
    with open(savepath + "/data_test_statistics.dat", "wb") as m:
      pickle.dump(self.results_list, m, protocol=pickle.HIGHEST_PROTOCOL)


  def plotResults(self, savepath, tr_num, euclid_plot, dist_plot, errors_mean):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 14})
    class MidpointNormalize(mpl.colors.Normalize):
      def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
      def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

    xticks = np.arange(1, len(self.test_angles), 2)
    yticks = np.arange(0, len(self.test_dist), 2)
    # norm1 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=50)
    # norm2 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=50)
    norm1 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=euclid_plot.max())
    norm2 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=dist_plot.max())
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
    ax.set_xticklabels([str(x) for x in self.test_angles[xticks-1][::-1]])
    ax.set_yticklabels([str(y) for y in self.test_dist[yticks][::-1]])
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
    ax.set_xticklabels([str(x) for x in self.test_angles[xticks-1][::-1]])
    ax.set_yticklabels([str(y) for y in self.test_dist[yticks][::-1]])
    cbar = plt.colorbar(sidf, shrink=0.7, aspect=20, pad = 0.15, orientation='horizontal', ticks=[-1, errors_mean[1].round(2), dist_plot.max().round(2)])
    sidf.set_clim(-1.001, dist_plot.max()+.005)
    plt.savefig(savepath + "/img_test_plots_trial_#{}.svg".format(tr_num))
    if self.show_plots:
      plt.show()
    else:
      plt.cla()