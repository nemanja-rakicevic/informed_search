
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Plotting functions:
                    - evaluation heatmaps
                    - model and exploration components
                    - model and exploration components separate files
"""

import os
import logging
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.rcParams.update({'font.size': 14})

logger = logging.getLogger(__name__)


class MidpointNormalize(mpl.colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_evals(euclid_plot,
               polar_plot,
               errors_mean,
               test_dict,
               savepath=None,
               num_trial=None,
               show_plots=False,
               img_format='png',
               dpi=150):
    """Plot task model evaluation heatmap over test targets"""
    # Set axis ticks
    xticks = np.arange(1, len(test_dict['angles']), 2)
    yticks = np.arange(0, len(test_dict['dist']), 2)
    norm1 = MidpointNormalize(midpoint=0., vmin=-1, vmax=euclid_plot.max())
    norm2 = MidpointNormalize(midpoint=0., vmin=-1, vmax=polar_plot.max())
    fig = plt.figure(figsize=(15, 5), dpi=100)
    fig.suptitle("Performance Error Plots (failed = -1)", fontsize=16)
    # Eudlidean error plot
    ax = plt.subplot("121")
    ax.set_ylabel('distances')
    ax.set_xlabel('angles')
    ax.set_title("Euclidean error: {}".format(errors_mean[0].round(2)))
    euc = ax.imshow(euclid_plot, origin='upper', cmap=cm.seismic, norm=norm1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([str(x) for x in test_dict['angles'][xticks - 1][::-1]])
    ax.set_yticklabels([str(y) for y in test_dict['dist'][yticks][::-1]])
    cbar = plt.colorbar(
        euc, shrink=0.7, aspect=20, pad=0.15, orientation='horizontal',
        ticks=[-1, errors_mean[0].round(2), euclid_plot.max().round(2)])
    cbar.ax.set_xticklabels(['-1', 'mean', 'max'])
    euc.set_clim(-1.001, euclid_plot.max() + .005)
    # Polar error plot
    ax = plt.subplot("122")
    ax.set_title("Polar coordinate error: {}".format(errors_mean[1].round(2)))
    ax.set_xlabel('angles')
    sidf = ax.imshow(polar_plot, origin='upper', cmap=cm.seismic, norm=norm2)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([str(x) for x in test_dict['angles'][xticks - 1][::-1]])
    ax.set_yticklabels([str(y) for y in test_dict['dist'][yticks][::-1]])
    cbar = plt.colorbar(
        sidf, shrink=0.7, aspect=20, pad=0.15, orientation='horizontal',
        ticks=[-1, errors_mean[1].round(2), polar_plot.max().round(2)])
    sidf.set_clim(-1.001, polar_plot.max() + .005)

    savepath = os.path.join(savepath, "plots_eval")
    if savepath is not None:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        if type(num_trial) == int:
            fig_name = 'test_plots_trial_{:05d}.{}'.format(
                num_trial, img_format)
            plt.savefig('{}/{}'.format(savepath, fig_name),
                        format=img_format, dpi=dpi)
            logger.info("Figure saved: '{}'".format(fig_name))
        else:
            plt.savefig('{}/{}.{}'.format(
                        savepath, num_trial, img_format),
                        format=img_format, dpi=dpi)
    if show_plots:
        plt.show()
    else:
        plt.cla()


def plot_model(model_object,
               dimensions=(0, 1),
               savepath=None,
               num_trial=None,
               show_points=False,
               show_plots=False,
               ds_plots=True,
               img_format='png',
               dpi=150):
    """
    Plot task model components and exploration components,
    if multidimensions then along custom dims.
    """
    param_names = ['joint_{}'.format(d) for d in dimensions]
    if len(model_object.mu_alpha):
        fig = plt.figure(
            "DISTRIBUTIONs at step: {}".format(num_trial),
            figsize=None)
        fig.set_size_inches(
            fig.get_size_inches()[0] * 3,
            fig.get_size_inches()[1] * 2)
        dim1 = model_object.param_list[dimensions[0]]
        dim2 = model_object.param_list[dimensions[1]]
        # Extract values to plot
        if len(model_object.param_dims) > 2:
            if model_object.param_dims[0] > 1:
                model_alpha = model_object.mu_alpha[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
                model_L = model_object.mu_L[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
                model_pidf = model_object.pidf[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
                model_uidf = model_object.uidf[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
                model_sidf = model_object.sidf[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
            else:
                model_alpha = model_object.mu_alpha[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
                model_L = model_object.mu_L[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
                model_pidf = model_object.pidf[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
                model_uidf = model_object.uidf[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
                model_sidf = model_object.sidf[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
        else:
            model_alpha = model_object.mu_alpha
            model_L = model_object.mu_L
            model_pidf = model_object.pidf
            model_uidf = model_object.uidf
            model_sidf = model_object.sidf
        # Creat 3D plot meshgrid
        X, Y = np.meshgrid(dim2, dim1)
        # Downsample for memory contstraints
        ds1 = max(1, len(dim1) // 50) if ds_plots else 1
        ds2 = max(1, len(dim2) // 50) if ds_plots else 1
        dim1 = dim1[::ds1]
        dim2 = dim2[::ds2]
        model_alpha = model_alpha[::ds1, ::ds2]
        model_L = model_L[::ds1, ::ds2]
        model_pidf = model_pidf[::ds1, ::ds2]
        model_uidf = model_uidf[::ds1, ::ds2]
        model_sidf = model_sidf[::ds1, ::ds2]
        X = X[::ds1, ::ds2]
        Y = Y[::ds1, ::ds2]
        # Set ticks
        xticks = np.linspace(
            min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
        yticks = np.linspace(
            min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 4).round(1)
        xticks1 = np.linspace(
            min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
        yticks1 = np.linspace(
            min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(1)
        zticks_alpha = np.linspace(
            model_alpha.min(), model_alpha.max(), 5).round(2)
        zticks_L = np.linspace(
            model_L.min(), model_L.max(), 5).round(2)
        zticks_pidf = np.linspace(
            model_pidf.min(), model_pidf.max(), 7).round(2)
        zticks_uidf = np.linspace(
            model_uidf.min(), model_uidf.max(), 7).round(2)
        search_lim = (
            min((1 - model_pidf).min(), model_uidf.min(), model_sidf.min()),
            max((1 - model_pidf).max(), model_uidf.max(), model_sidf.max()))

        # Task models
        # Angle task model
        ax = fig.add_subplot(2, 3, 1, projection='3d')
        ax.set_title('ANGLE MODEL')
        ax.plot_surface(X, Y, model_alpha, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_ylabel(param_names[1], labelpad=5)
        ax.set_xlabel(param_names[0], labelpad=5)
        ax.set_zlabel('[degrees]       ', rotation='vertical', labelpad=10)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        if abs(model_alpha.max() - model_alpha.min()) >= 1:
            ax.set_zticks(zticks_alpha)
        else:
            ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)
        # Distance task model
        ax = fig.add_subplot(2, 3, 2, projection='3d')
        ax.set_title('DISTANCE MODEL')
        ax.plot_surface(X, Y, model_L, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_ylabel(param_names[1], labelpad=5)
        ax.set_xlabel(param_names[0], labelpad=5)
        ax.set_zlabel('[cm]', rotation='vertical', labelpad=10)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        if abs(model_L.max() - model_L.min()) >= 1:
            ax.set_zticks(zticks_L)
        else:
            ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)

        # Exploration components
        # Selection IDF (top view)
        ax = fig.add_subplot(2, 3, 3)
        ax.set_title('Selection function')
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        # ax.set_xlim(len(dim1), 0)
        ax.set_xlim(0, len(dim1))
        ax.set_ylim(0, len(dim2))
        # ax.set_xticks(np.linspace(len(dim1)-1, -1, 5))
        ax.set_xticks(np.linspace(-1, len(dim1), 5))
        ax.set_yticks(np.linspace(-1, len(dim2), 5))
        ax.set_xticklabels([str(x) for x in xticks])
        ax.set_yticklabels([str(y) for y in yticks1])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        sidf = ax.imshow(model_sidf, cmap=cm.summer, origin='lower')
        for spine in ax.spines.values():
            spine.set_visible(False)
        # add also the trial points
        for tr in model_object.coord_explored:
            if list(tr) in [list(x) for x in model_object.coord_failed]:
                ax.scatter(x=tr[1] // ds1, y=tr[0] // ds2, c='r', s=15)
            else:
                ax.scatter(x=tr[1] // ds1, y=tr[0] // ds2, c='c', s=15)
        cbar = plt.colorbar(
            sidf, shrink=0.5, aspect=20, pad=0.17, orientation='horizontal',
            ticks=[0.0, 0.5, 1.0])
        sidf.set_clim(-0.001, 1.001)
        # Penalisation IDF
        if 'Informed' in model_object.name:
            ax = fig.add_subplot(2, 3, 4, projection='3d')
            ax.set_title('Penalisation function: {} points'.format(
                len(model_object.coord_failed)))
            ax.plot_surface(X, Y, (1 - model_pidf), rstride=1, cstride=1,
                            cmap=cm.copper, linewidth=0, antialiased=False)
            ax.set_zlim(search_lim)
            ax.set_xlabel(param_names[0], labelpad=5)
            ax.set_ylabel(param_names[1], labelpad=5)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([str(x) for x in xticks], rotation=41)
            ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
            ax.tick_params(axis='x', direction='out', pad=-5)
            ax.tick_params(axis='y', direction='out', pad=-3)
            ax.tick_params(axis='z', direction='out', pad=2)
        # Uncertainty IDF
        if 'Random' not in model_object.name:
            ax = fig.add_subplot(2, 3, 5, projection='3d')
            ax.set_title('Model uncertainty: {:4.2f}'.format(
                model_object.uncertainty))
            ax.plot_surface(X, Y, model_uidf, rstride=1, cstride=1,
                            cmap=cm.winter, linewidth=0, antialiased=False)
            ax.set_zlim(search_lim)
            ax.set_xlabel(param_names[0], labelpad=5)
            ax.set_ylabel(param_names[1], labelpad=5)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([str(x) for x in xticks], rotation=41)
            ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
            ax.tick_params(axis='x', direction='out', pad=-5)
            ax.tick_params(axis='y', direction='out', pad=-3)
            ax.tick_params(axis='z', direction='out', pad=5)
        # Selection IDF (3D view)
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        ax.set_title('Selection function')
        ax.plot_surface(X, Y, model_sidf, rstride=1, cstride=1,
                        cmap=cm.summer, linewidth=0, antialiased=False)
        ax.set_zlim(search_lim)
        ax.set_xlabel(param_names[0], labelpad=5)
        ax.set_ylabel(param_names[1], labelpad=5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=2)
        # add also the trial points
        if show_points:
            for tr in model_object.coord_explored:
                if list(tr) in [list(x) for x in model_object.coord_failed]:
                    ax.plot([dim2[tr[1] // ds2], dim2[tr[1] // ds2]],
                            [dim1[tr[0] // ds1], dim1[tr[0] // ds1]],
                            [model_sidf.min(), model_sidf.max()],
                            linewidth=1, color='r', alpha=0.7)
                else:
                    ax.plot([dim2[tr[1] // ds2], dim2[tr[1] // ds2]],
                            [dim1[tr[0] // ds1], dim1[tr[0] // ds1]],
                            [model_sidf.min(), model_sidf.max()],
                            linewidth=1, color='c', alpha=0.7)

        savepath = os.path.join(savepath, "plots_model")
        if savepath is not None:
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            if type(num_trial) == int:
                fig_title = "Models and IDFs" \
                            "(num_iter: {}, resolution: {})".format(
                                num_trial, len(dim1))
                fig.suptitle(fig_title, fontsize=16)
                fig_name = 'model_plots_trial_{:05d}.{}'.format(
                    num_trial, img_format)
                plt.savefig('{}/{}'.format(savepath, fig_name),
                            format=img_format, dpi=dpi)
                logger.info("Figure saved: '{}'".format(fig_name))
            else:
                plt.savefig('{}/{}.{}'.format(
                            savepath, num_trial, img_format),
                            format=img_format, dpi=dpi)
        if show_plots:
            plt.show()
        else:
            plt.cla()


def plot_model_separate(model_object,
                        dimensions=(0, 1),
                        savepath=None,
                        num_trial=None,
                        show_points=False,
                        show_plots=False,
                        ds_plots=True,
                        img_format='png',
                        dpi=150):
    """
    Plot task model components and exploration components,
    each in a separate file
    """
    savepath = os.path.join(savepath, "plots_model")
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    param_names = ['joint_{}'.format(d) for d in dimensions]
    if len(model_object.mu_alpha):
        fig = plt.figure(
            "DISTRIBUTIONs at step: {}".format(num_trial),
            figsize=None)
        fig.set_size_inches(
            fig.get_size_inches()[0] * 3,
            fig.get_size_inches()[1] * 2)
        dim1 = model_object.param_list[dimensions[0]]
        dim2 = model_object.param_list[dimensions[1]]
        # Extract values to plot
        if len(model_object.param_dims) > 2:
            if model_object.param_dims[0] > 1:
                model_alpha = model_object.mu_alpha[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
                model_L = model_object.mu_L[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
                model_pidf = model_object.pidf[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
                model_uidf = model_object.uidf[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
                model_sidf = model_object.sidf[:, :, 3, 3, 4].reshape(
                    len(dim1), len(dim2))
            else:
                model_alpha = model_object.mu_alpha[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
                model_L = model_object.mu_L[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
                model_pidf = model_object.pidf[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
                model_uidf = model_object.uidf[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
                model_sidf = model_object.sidf[0, 0, :, 0, :, 0].reshape(
                    len(dim1), len(dim2))
        else:
            model_alpha = model_object.mu_alpha
            model_L = model_object.mu_L
            model_pidf = model_object.pidf
            model_uidf = model_object.uidf
            model_sidf = model_object.sidf
        # Creat 3D plot meshgrid
        X, Y = np.meshgrid(dim2, dim1)
        # Downsample for memory contstraints
        ds1 = max(1, len(dim1) // 50) if ds_plots else 1
        ds2 = max(1, len(dim2) // 50) if ds_plots else 1
        dim1 = dim1[::ds1]
        dim2 = dim2[::ds2]
        model_alpha = model_alpha[::ds1, ::ds2]
        model_L = model_L[::ds1, ::ds2]
        model_pidf = model_pidf[::ds1, ::ds2]
        model_uidf = model_uidf[::ds1, ::ds2]
        model_sidf = model_sidf[::ds1, ::ds2]
        X = X[::ds1, ::ds2]
        Y = Y[::ds1, ::ds2]
        # Set ticks
        xticks = np.linspace(
            min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
        yticks = np.linspace(
            min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 4).round(1)
        xticks1 = np.linspace(
            min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
        yticks1 = np.linspace(
            min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(1)
        zticks_alpha = np.linspace(
            model_alpha.min(), model_alpha.max(), 5).round(2)
        zticks_L = np.linspace(
            model_L.min(), model_L.max(), 5).round(2)
        zticks_pidf = np.linspace(
            model_pidf.min(), model_pidf.max(), 7).round(2)
        zticks_uidf = np.linspace(
            model_uidf.min(), model_uidf.max(), 7).round(2)
        search_lim = (
            min((1 - model_pidf).min(), model_uidf.min(), model_sidf.min()),
            max((1 - model_pidf).max(), model_uidf.max(), model_sidf.max()))

        # Task models
        # Angle task model
        fig = plt.figure("ANGLE MODEL #{}".format(num_trial), figsize=None)
        fig.set_size_inches(
            fig.get_size_inches()[0],
            fig.get_size_inches()[1])
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, model_alpha, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel(param_names[0], labelpad=10)
        ax.set_ylabel(param_names[1], labelpad=10)
        ax.set_zlabel('[degrees]       ', rotation='vertical', labelpad=10)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        if abs(model_alpha.max() - model_alpha.min()) >= 1:
            ax.set_zticks(zticks_alpha)
        else:
            ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)
        plt.savefig("{}/img_trial_{:05d}_model_angle.png".format(
            savepath, num_trial),
            format="png")
        # Distance task model
        fig = plt.figure("DISTANCE MODEL #{}".format(num_trial), figsize=None)
        fig.set_size_inches(
            fig.get_size_inches()[0],
            fig.get_size_inches()[1])
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, model_L, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel(param_names[0], labelpad=10)
        ax.set_ylabel(param_names[1], labelpad=10)
        ax.set_zlabel('[cm]', rotation='vertical', labelpad=10)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        if abs(model_L.max() - model_L.min()) >= 1:
            ax.set_zticks(zticks_L)
        else:
            ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)
        plt.savefig("{}/img_trial_{:05d}_model_dist.png".format(
            savepath, num_trial),
            format="png")

        # Exploration components
        # Selection IDF (top view)
        fig = plt.figure("SELECTION IDF #{}".format(num_trial), figsize=None)
        fig.set_size_inches(
            fig.get_size_inches()[0],
            fig.get_size_inches()[1])
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(param_names[0])
        ax1.set_ylabel(param_names[1])
        # ax1.set_xlim(len(dim1), 0)
        ax1.set_xlim(0, len(dim1))
        ax1.set_ylim(0, len(dim2))
        # ax1.set_xticks(np.linspace(len(dim1)-1, -1, 5))
        ax1.set_xticks(np.linspace(-1, len(dim1), 5))
        ax1.set_yticks(np.linspace(-1, len(dim2), 5))
        ax1.set_xticklabels([str(x) for x in xticks])
        ax1.set_yticklabels([str(y) for y in yticks1])
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        sidf = ax1.imshow(model_sidf, cmap=cm.summer, origin='lower')
        for spine in ax1.spines.values():
            spine.set_visible(False)
        # add also the trial points
        for tr in model_object.coord_explored:
            if list(tr) in [list(x) for x in model_object.coord_failed]:
                ax1.scatter(x=tr[1] // ds1, y=tr[0] // ds2, c='r', s=15)
            else:
                ax1.scatter(x=tr[1] // ds1, y=tr[0] // ds2, c='c', s=15)
        cbar = plt.colorbar(sidf, shrink=0.5, aspect=20, pad=0.17,
                            orientation='horizontal', ticks=[0.0, 0.5, 1.0])
        sidf.set_clim(-0.001, 1.001)
        plt.savefig("{}/img_trial_{:05d}_sidf_top.png".format(
            savepath, num_trial),
            format="png")
        # Penalisation IDF
        if 'Informed' in model_object.name:
            fig = plt.figure(
                "PENALISATION IDF #{}".format(num_trial), figsize=None)
            fig.set_size_inches(
                fig.get_size_inches()[0],
                fig.get_size_inches()[1])
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.plot_surface(
                X, Y, (1 - model_pidf), rstride=1, cstride=1, cmap=cm.copper,
                linewidth=0, antialiased=False)
            ax1.set_zlim(search_lim)
            ax1.set_ylabel(param_names[1], labelpad=5)
            ax1.set_xlabel(param_names[0], labelpad=5)
            ax1.set_xticks(xticks)
            ax1.set_yticks(yticks)
            ax1.set_xticklabels([str(x) for x in xticks], rotation=41)
            ax1.set_yticklabels([str(x) for x in yticks], rotation=-15)
            ax1.tick_params(axis='x', direction='out', pad=-5)
            ax1.tick_params(axis='y', direction='out', pad=-3)
            ax1.tick_params(axis='z', direction='out', pad=2)
            plt.savefig("{}/img_trial_{:05d}_pidf.png".format(
                savepath, num_trial),
                format="png")
        # Uncertainty IDF
        if 'Random' not in model_object.name:
            fig = plt.figure(
                "UNCERTAINTY IDF #{}".format(num_trial), figsize=None)
            fig.set_size_inches(
                fig.get_size_inches()[0],
                fig.get_size_inches()[1])
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.plot_surface(
                X, Y, model_uidf, rstride=1, cstride=1, cmap=cm.winter,
                linewidth=0, antialiased=False)
            ax1.set_zlim(search_lim)
            ax1.set_ylabel(param_names[1], labelpad=5)
            ax1.set_xlabel(param_names[0], labelpad=5)
            ax1.set_xticks(xticks)
            ax1.set_yticks(yticks)
            ax1.set_zticks(zticks_uidf)
            ax1.set_xticklabels([str(x) for x in xticks], rotation=41)
            ax1.set_yticklabels([str(x) for x in yticks], rotation=-15)
            ax1.tick_params(axis='x', direction='out', pad=-5)
            ax1.tick_params(axis='y', direction='out', pad=-3)
            ax1.tick_params(axis='z', direction='out', pad=5)
            plt.savefig("{}/img_trial_{:05d}_uidf.png".format(
                savepath, num_trial),
                format="png")
        # Selection IDF
        fig = plt.figure("SELECTION IDF #{}".format(num_trial), figsize=None)
        fig.set_size_inches(
            fig.get_size_inches()[0],
            fig.get_size_inches()[1])
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(X, Y, model_sidf, rstride=1, cstride=1,
                         cmap=cm.summer, linewidth=0, antialiased=False)
        ax1.set_zlim(search_lim)
        ax1.set_ylabel(param_names[1], labelpad=5)
        ax1.set_xlabel(param_names[0], labelpad=5)
        ax1.set_xticks(xticks)
        ax1.set_yticks(yticks)
        ax1.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax1.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax1.tick_params(axis='x', direction='out', pad=-5)
        ax1.tick_params(axis='y', direction='out', pad=-3)
        ax1.tick_params(axis='z', direction='out', pad=2)
        # add also the trial points
        if show_points:
            for tr in model_object.coord_explored:
                if list(tr) in [list(x) for x in model_object.coord_failed]:
                    ax1.plot([dim2[tr[1] // ds2], dim2[tr[1] // ds2]],
                             [dim1[tr[0] // ds1], dim1[tr[0] // ds1]],
                             [model_sidf.min(), model_sidf.max()],
                             linewidth=1, color='r', alpha=0.7)
                else:
                    ax1.plot([dim2[tr[1] // ds2], dim2[tr[1] // ds2]],
                             [dim1[tr[0] // ds1], dim1[tr[0] // ds1]],
                             [model_sidf.min(), model_sidf.max()],
                             linewidth=1, color='c', alpha=0.7)
        plt.savefig("{}/img_trial_{:05d}_sidf.png".format(
            savepath, num_trial),
            format="png")
