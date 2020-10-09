
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Plot experiment evaluations, filtered using keyworkds,
                averaged over their hyperparameters:
                Plots the three graphs:
                - test error mean
                - test error std
                - number of failed trials
"""

import os
import glob
import pickle
import argparse

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams.update({'font.size': 18})

import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator


def plot_performance(load_path,
                     filter_string,
                     savepath=None,
                     show_plots=False,
                     img_format='jpg',
                     dpi=300,
                     **kwargs):
    """Plot based on trial evaluation statistics on full test set."""
    CUTOFF = 300
    idx = -2 if load_path[-1] == '/' else -1
    graph_name = load_path.split('/')[idx] + '__plot_comparison'
    if len(filter_string):
        graph_name += '__' + filter_string

    # Extract data to plot
    filter_include = []
    filter_exclude = []
    experiments_dict = {}
    all_stats = []
    for fterm in filter_string.split('+'):
        if len(fterm) and fterm[0] == '^':
            filter_exclude += glob.glob('{}/ENV_*{}*'.format(
                load_path, fterm[1:]))
        else:
            filter_include += glob.glob('{}/ENV_*{}*'.format(
                load_path, fterm))
    filter_exp = np.setdiff1d(filter_include, filter_exclude)
    for d in filter_exp:
        exp_name = '__'.join(d.split('/')[-1].split('__')[:3])
        stats_files = glob.glob(os.path.join(d, 'S*/statistics_evaluation.pkl'))
        if exp_name in experiments_dict.keys():
            experiments_dict[exp_name] += stats_files
        else:
            experiments_dict[exp_name] = stats_files
    for stype in experiments_dict.keys():
        seeds_error_mean = []
        seeds_error_std = []
        seeds_fail_mean = []
        for exp_file in experiments_dict[stype]:
            with open(exp_file, "rb") as m:
                trial_stats = pickle.load(m)
            # extract the test info for every trial in the experiment
            tmp_errs = [ts[4][0] for ts in trial_stats]
            tmp_stds = [ts[5][0] for ts in trial_stats]
            tmp_fails = [ts[6] for ts in trial_stats]
            # save as current seed
            seeds_error_mean.append(np.array(tmp_errs)[:CUTOFF])
            seeds_error_std.append(np.array(tmp_stds)[:CUTOFF])
            seeds_fail_mean.append(np.array(tmp_fails)[:CUTOFF])
        # Cut seeds to length
        min_len = min([len(m) for m in seeds_error_mean])
        seeds_error_mean = [mm[:min_len] for mm in seeds_error_mean]
        seeds_error_std = [mm[:min_len] for mm in seeds_error_std]
        seeds_fail_mean = [mm[:min_len] for mm in seeds_fail_mean]
        # Average seed for current model
        mnames = stype.split('__')
        minfo = mnames[2].split('_')
        all_stats.append(
            {"model": '; '.join([mnames[1].split('_')[-1],
                                 'pidf={}'.format(minfo[1][4:]),
                                 'kernel={}'.format(minfo[2][6:]),
                                 '$\sigma_l^2$={}'.format(minfo[3][2:])]),
             "mean": np.array([np.array(seeds_error_mean).mean(axis=0),
                               np.array(seeds_error_mean).std(axis=0)]),
             "std": np.array([np.array(seeds_error_std).mean(axis=0),
                              np.array(seeds_error_std).std(axis=0)]),
             "fails": np.array([np.array(seeds_fail_mean).mean(axis=0),
                                np.array(seeds_fail_mean).std(axis=0)])})

    # Plot setup
    lw = 1
    clrand = ['red', 'maroon', 'deeppink']
    clinfo = ['blue', 'olive', 'indigo', 'teal', 'purple', 'maroon', 'deeppink',
              'red', 'olive', 'indigo', 'teal', 'purple', 'maroon', 'deeppink',
              'red']
    f, axarr = plt.subplots(3, sharex=True)
    f.set_size_inches(f.get_size_inches()[0] * 3, f.get_size_inches()[1] * 2)

    # Euclidean error means over full test set
    max_xaxis = 0
    max_yaxis = 0
    for i, a in enumerate(all_stats):
        mean = np.array(a['mean'][0])
        std = np.array(a['mean'][1])
        xaxis = range(len(mean))
        if len(mean) > max_xaxis:
            max_xaxis = len(mean)
        if mean.max() > max_yaxis:
            max_yaxis = mean.max()
        lb = a['model']  # + ' models'
        axarr[0].plot(mean, label=lb, linewidth=lw)
        axarr[0].fill_between(xaxis, (mean - std).clip(0), mean + std,
                              alpha=0.5)  # , label=lb)
    axarr[0].set_ylabel('Test error mean', labelpad=14)  # (Euclidean distance)
    axarr[0].grid(color='gray', linestyle=':', linewidth=0.8)
    axarr[0].set_xlim(0, min(CUTOFF, max_xaxis))
    axarr[0].set_ylim(0, max(24, max_yaxis))
    axarr[0].yaxis.set_major_locator(MultipleLocator(5))

    # Euclidean error stds over full test set
    max_yaxis = 0
    for i, a in enumerate(all_stats):
        mean = np.array(a['std'][0])
        std = np.array(a['std'][1])
        xaxis = range(len(mean))
        if mean.max() > max_yaxis:
            max_yaxis = mean.max()
        lb = a['model']  # + ' models'
        axarr[1].plot(mean, label=lb, linewidth=lw)
        axarr[1].fill_between(xaxis, (mean - std).clip(0), mean + std,
                              alpha=0.5)  # , label=lb)
    axarr[1].set_ylabel('Test error std', labelpad=14)
    axarr[1].grid(color='gray', linestyle=':', linewidth=0.8)
    axarr[0].set_xlim(0, min(CUTOFF, max_xaxis))
    axarr[1].set_ylim(0, max(11, max_yaxis))
    axarr[1].yaxis.set_major_formatter(FormatStrFormatter('%i'))
    axarr[1].yaxis.set_major_locator(MaxNLocator(5))

    # Total fail count over full test set
    max_yaxis = 0
    for i, a in enumerate(all_stats):
        mean = np.array(a['fails'][0])
        std = np.array(a['fails'][1])
        xaxis = range(len(mean))
        if mean.max() > max_yaxis:
            max_yaxis = mean.max()
        lb = a['model']  # + ' models'
        axarr[2].plot(mean, label=lb, linewidth=lw)
        axarr[2].fill_between(xaxis, (mean - std).clip(0), mean + std,
                              alpha=0.5)  # , label=lb)
    axarr[2].set_ylabel('Number of failed tests')
    axarr[2].grid(color='gray', linestyle=':', linewidth=0.8)
    axarr[0].set_xlim(0, min(CUTOFF, max_xaxis))
    axarr[2].set_ylim(-1, 160)
    axarr[2].set_xlabel('Trial number')

    # Add labels and legends
    legends = []
    handles, labels = axarr[2].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles),
                                  key=lambda t: t[0], reverse=True))
    legends.append(axarr[2].legend(handles, labels,
                                   loc='upper right', bbox_to_anchor=(1, 1)))
    # plt.tight_layout(pad=0.1, w_pad=0.90, h_pad=0.90)

    # Save/show figure
    savepath = load_path if savepath is None else savepath
    plt.savefig('{}/{}.{}'.format(savepath, graph_name, img_format),
                format=img_format, bbox_extra_artists=tuple(legends),
                bbox_inches='tight', dpi=dpi)
    if show_plots:
        plt.show()
    else:
        plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-load', '--load_path',
                        default=None, required=True,
                        help="Path to directory to plot.")
    parser.add_argument('-save', '--save_path',
                        default=None, required=False,
                        help="Path to directory where to save.")
    parser.add_argument('-f', '--filter_string',
                        default='',
                        help="Take into account experiments that contain this.")
    args = parser.parse_args()
    plot_performance(args.load_path, args.filter_string)
