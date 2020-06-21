
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Useful plotting function
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



def plot_evals(euclid_plot, polar_plot, errors_mean, test_dict,
               savepath=None, num_trial=None, 
               show_plots=False, img_format='png', dpi=300):

    xticks = np.arange(1, len(test_dict['angles']), 2)
    yticks = np.arange(0, len(test_dict['dist']), 2)
    # norm1 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=50)
    # norm2 = MidpointNormalize(midpoint = 0., vmin=-1, vmax=50)
    norm1 = MidpointNormalize(midpoint=0., vmin=-1, vmax=euclid_plot.max())
    norm2 = MidpointNormalize(midpoint=0., vmin=-1, vmax=polar_plot.max())
    fig = plt.figure(figsize=(15,5), dpi=100)
    fig.suptitle("Performance Error Plots (failed = -1)", fontsize=16)

    # EUCLIDEAN ERROR
    ax = plt.subplot("121")
    ax.set_ylabel('distances')
    ax.set_xlabel('angles')
    ax.set_title("Euclidean error: {}".format(errors_mean[0].round(2)))
    euc = ax.imshow(euclid_plot, origin='upper', cmap=cm.seismic, norm=norm1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([str(x) for x in test_dict['angles'][xticks-1][::-1]])
    ax.set_yticklabels([str(y) for y in test_dict['dist'][yticks][::-1]])
    cbar = plt.colorbar(euc, shrink=0.7, aspect=20, pad=0.15, 
                        orientation='horizontal', 
                        ticks=[-1, errors_mean[0].round(2), 
                               euclid_plot.max().round(2)])
    cbar.ax.set_xticklabels(['-1', 'mean', 'max'])
    euc.set_clim(-1.001, euclid_plot.max()+.005)

    # POLAR ERROR
    ax = plt.subplot("122")
    ax.set_title("Polar coordinate error: {}".format(errors_mean[1].round(2)))
    ax.set_xlabel('angles')
    sidf = ax.imshow(polar_plot, origin='upper', cmap=cm.seismic, norm=norm2)
    # ax.set_xticks(np.arange(len(self.test_angles)))
    # ax.set_yticks(np.arange(len(self.test_dist)))
    # ax.set_xticklabels([str(x) for x in self.test_angles[::-1]])
    # ax.set_yticklabels([str(y) for y in self.test_dist[::-1]])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([str(x) for x in test_dict['angles'][xticks-1][::-1]])
    ax.set_yticklabels([str(y) for y in test_dict['dist'][yticks][::-1]])
    cbar = plt.colorbar(sidf, shrink=0.7, aspect=20, pad=0.15, 
                        orientation='horizontal', 
                        ticks=[-1, errors_mean[1].round(2), 
                               polar_plot.max().round(2)])
    sidf.set_clim(-1.001, polar_plot.max()+.005)

    savepath = os.path.join(savepath, "plot_eval")
    if savepath is not None:
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        if type(num_trial)==int:
            fig_name = 'test_plots_trial_{:05d}.{}'.format(
                        num_trial, img_format)
            plt.savefig('{}/{}'.format(savepath, fig_name), 
                        format=img_format, dpi=dpi, 
                        # bbox_inches='tight'
                        ) 
            logger.info("Figure saved: '{}'".format(fig_name))
        else:
            plt.savefig('{}/{}.{}'.format(
                        savepath, num_trial, img_format), 
                        format=img_format, dpi=dpi, 
                        # bbox_inches='tight'
                        ) 
    if show_plots:
        plt.show()
    else:
        plt.cla()




def plot_model(model_object, dimensions,
               savepath=None, num_trial=None, show_points=False,
               show_plots=False, img_format='png', dpi=300): 
    # if num_trial%1==0 or trial_info.fail_status==0:
    # print "<- CHECK PLOTS"     

    param_names = ['joint_{}'.format(d) for d in dimensions]


    
    if len(model_object.mu_alpha):
        fig = plt.figure("DISTRIBUTIONs at step: "+str(num_trial), figsize=None)
        fig.set_size_inches(fig.get_size_inches()[0]*3,fig.get_size_inches()[1]*2)
        dim1 = model_object.param_list[dimensions[0]]
        dim2 = model_object.param_list[dimensions[1]]
        X, Y = np.meshgrid(dim2, dim1)
        # Values to plot
        if len(model_object.param_dims)>2:
            if model_object.param_dims[0]>1:
                model_alpha  = model_object.mu_alpha[:,:,3,3,4].reshape(len(dim1),len(dim2))
                model_L      = model_object.mu_L[:,:,3,3,4].reshape(len(dim1),len(dim2))
                model_PIDF   = model_object.pidf[:,:,3,3,4].reshape(len(dim1),len(dim2))
                model_var    = model_object.uidf[:,:,3,3,4].reshape(len(dim1),len(dim2))
                model_select = model_object.sidf[:,:,3,3,4].reshape(len(dim1),len(dim2))

                # model_alpha  = model_object.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                # model_L      = model_object.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                # model_PIDF   = model_object.pidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                # model_var    = model_object.uidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                # model_select = model_object.sidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
            else:
                model_alpha  = model_object.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                model_L      = model_object.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                model_PIDF   = model_object.pidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                model_var    = model_object.uidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                model_select = model_object.sidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
        else:
            model_alpha  = model_object.mu_alpha
            model_L      = model_object.mu_L
            model_PIDF   = model_object.pidf
            model_var    = model_object.uidf
            model_select = model_object.sidf
        # Set ticks
        xticks = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
        yticks = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 4).round(1)
        # xticks1 = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
        yticks1 = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(1)
        #
        zticks_alpha = np.linspace(model_object.mu_alpha.min(), model_object.mu_alpha.max(), 4).round()
        zticks_L = np.linspace(model_object.mu_L.min(), model_object.mu_L.max(), 4).round()
        zticks_unc = np.linspace(model_object.uidf.min(), model_object.uidf.max(), 4).round(2)
        # zticks_PIDF = np.linspace(model_object.pidf.min(), model_object.pidf.max(), 7).round(1)
        # ANGLE MODEL
        ax = plt.subplot2grid((2,6),(0, 0), colspan=2, projection='3d')
        ax.set_title('ANGLE MODEL')
        ax.set_ylabel(param_names[1], labelpad=5)
        ax.set_xlabel(param_names[0], labelpad=5)
        ax.set_zlabel('[degrees]', rotation='vertical', labelpad=10)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks_alpha)
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)
        ax.plot_surface(X, Y, model_alpha, rstride=1, cstride=1, 
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # DISTANCE MODEL
        ax = plt.subplot2grid((2,6),(0, 2), colspan=2, projection='3d')
        ax.set_title('DISTANCE MODEL')
        ax.set_ylabel(param_names[1], labelpad=5)
        ax.set_xlabel(param_names[0], labelpad=5)
        ax.set_zlabel('[cm]', rotation='vertical', labelpad=10)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks_L)
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)
        ax.plot_surface(X, Y, model_L, rstride=1, cstride=1, 
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # SELECTION FUNCTION - TOP VIEW
        ax1 = plt.subplot2grid((2,6),(0, 4), colspan=2)
        ax1.set_title('Selection function')
        ax1.set_xlabel(param_names[0])
        ax1.set_ylabel(param_names[1])
        # ax1.set_xlim(len(dim1), 0)
        ax1.set_xlim(0, len(dim1))
        ax1.set_ylim(0, len(dim2))
        # ax1.set_xticks(np.linspace(len(dim1)-1, -1, 5))
        ax1.set_xticks(np.linspace(-1, len(dim1), 5))
        ax1.set_yticks(np.linspace(-1, len(dim2), 5))
        ax1.set_xticklabels([str(x) for x in xticks]), 
        ax1.set_yticklabels([str(y) for y in yticks1])
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        sidf = ax1.imshow(model_select, cmap=cm.summer, origin='lower')
        for spine in ax1.spines.values():
            spine.set_visible(False)
        # add also the trial points
        for tr in model_object.coord_explored:
            if list(tr) in [list(x) for x in model_object.coord_failed]:
                ax1.scatter(x=tr[1], y=tr[0], c='r', s=15)
            else:
                ax1.scatter(x=tr[1], y=tr[0], c='c', s=15)
        cbar = plt.colorbar(sidf, shrink=0.5, aspect=20, pad = 0.17, 
                            orientation='horizontal', ticks=[0.0, 0.5, 1.0])
        sidf.set_clim(-0.001, 1.001)

        # PENALISATION IDF
        ax = plt.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
        ax.set_title('Penalisation function: {} points'.format(
            len(model_object.coord_failed)))
        ax.set_ylabel(param_names[1], labelpad=5)
        ax.set_xlabel(param_names[0], labelpad=5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=2)
        # ax.set_zticks(zticks_PIDF)
        ax.plot_surface(X, Y, (1-model_PIDF), rstride=1, cstride=1, 
                        cmap=cm.copper, linewidth=0, antialiased=False)
        # UNCERTAINTY IDF
        ax = plt.subplot2grid((2,6),(1, 2), colspan=2, projection='3d')
        ax.set_title('Model uncertainty: {:4.2f}'.format(
            model_object.uncertainty))
        ax.set_ylabel(param_names[1], labelpad=5)
        ax.set_xlabel(param_names[0], labelpad=5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks_unc)
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)
        ax.plot_surface(X, Y, model_var, rstride=1, cstride=1, 
                        cmap=cm.winter, linewidth=0, antialiased=False)
        # SELECTION FUNCTION IDF
        ax = plt.subplot2grid((2,6),(1, 4), colspan=2, projection='3d')
        ax.set_title('Selection function')
        ax.set_ylabel(param_names[1], labelpad=5)
        ax.set_xlabel(param_names[0], labelpad=5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=2)
        surf = ax.plot_surface(X, Y, model_select, rstride=1, cstride=1, 
                               cmap=cm.summer, linewidth=0, antialiased=False)
        # add also the trial points
        if show_points:
            for tr in model_object.coord_explored:
                if list(tr) in [list(x) for x in model_object.coord_failed]:
                    ax.plot([dim2[tr[1]], dim2[tr[1]]], 
                            [dim1[tr[0]], dim1[tr[0]]], 
                            [model_select.min(), model_select.max()], 
                            linewidth=1, color='k', alpha=0.7)
                else:
                    ax.plot([dim2[tr[1]], dim2[tr[1]]], 
                            [dim1[tr[0]], dim1[tr[0]]], 
                            [model_select.min(), model_select.max()], 
                            linewidth=1, color='m', alpha=0.7)

        savepath = os.path.join(savepath, "plot_model")
        if savepath is not None:
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            if type(num_trial)==int:
                fig_title = "Models and IDFs "\
                             "(num_iter: {}, resolution: {})".format(num_trial, 
                                                                     len(dim1))
                fig.suptitle(fig_title, fontsize=16)
                fig_name = 'model_plots_trial_{:05d}.{}'.format(
                            num_trial, img_format)
                plt.savefig('{}/{}'.format(savepath, fig_name), 
                            format=img_format, dpi=dpi, 
                            # bbox_inches='tight'
                            ) 
                logger.info("Figure saved: '{}'".format(fig_name))
            else:
                plt.savefig('{}/{}.{}'.format(
                            savepath, num_trial, img_format), 
                            format=img_format, dpi=dpi, 
                            # bbox_inches='tight'
                            ) 
        if show_plots:
            plt.show()
        else:
            plt.cla()





def plot_model_separate(model_object, dimensions,
                    savepath=None, num_trial=None, show_points=False,
                    show_plots=False, img_format='png', dpi=300): 
    # if num_trial%1==0 or trial_info.fail_status==0:
    # print "<- CHECK PLOTS"     

    param_names = ['joint_{}'.format(d) for d in dimensions]
    
    if len(model_object.mu_alpha):
        dim1 = model_object.param_list[dimensions[0]]
        dim2 = model_object.param_list[dimensions[1]]
        X, Y = np.meshgrid(dim2, dim1)
        # Values to plot
        if len(model_object.param_dims)>2:
            if model_object.param_dims[0]>1:
                model_alpha  = model_object.mu_alpha[:,:,3,3,4].reshape(len(dim1),len(dim2))
                model_L      = model_object.mu_L[:,:,3,3,4].reshape(len(dim1),len(dim2))
                model_PIDF   = model_object.pidf[:,:,3,3,4].reshape(len(dim1),len(dim2))
                model_var    = model_object.uidf[:,:,3,3,4].reshape(len(dim1),len(dim2))
                model_select = model_object.sidf[:,:,3,3,4].reshape(len(dim1),len(dim2))

                # model_alpha  = model_object.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                # model_L      = model_object.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                # model_PIDF   = model_object.pidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                # model_var    = model_object.uidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                # model_select = model_object.sidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
            else:
                model_alpha  = model_object.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                model_L      = model_object.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                model_PIDF   = model_object.pidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                model_var    = model_object.uidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                model_select = model_object.sidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
        else:
            model_alpha  = model_object.mu_alpha
            model_L      = model_object.mu_L
            model_PIDF   = model_object.pidf
            model_var    = model_object.uidf
            model_select = model_object.sidf
        # Set ticks
        xticks = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
        yticks = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 4).round(1)
        # xticks1 = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
        yticks1 = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(1)
        #
        zticks_alpha = np.linspace(model_object.mu_alpha.min(), model_object.mu_alpha.max(), 4).round()
        zticks_L = np.linspace(model_object.mu_L.min(), model_object.mu_L.max(), 4).round()
        zticks_unc = np.linspace(model_object.uidf.min(), model_object.uidf.max(), 4).round(2)
        # zticks_PIDF = np.linspace(model_object.pidf.min(), model_object.pidf.max(), 7).round(1)
        

        # ANGLE MODEL
        fig = plt.figure("ANGLE MODEL"+str(num_trial), figsize=None)
        fig.set_size_inches(fig.get_size_inches()[0],fig.get_size_inches()[1])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(param_names[0], labelpad=10)
        ax.set_ylabel(param_names[1], labelpad=10)
        ax.set_zlabel('[degrees]', rotation='vertical', labelpad=10)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks_alpha)
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)
        ax.plot_surface(X, Y, model_alpha, rstride=1, cstride=1, 
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.savefig(model_object.dirname + \
                    "/img_training_trial#{:3d}_angle.png".format(num_trial), 
                    format="png")
        
        # DISTANCE MODEL
        fig = plt.figure("DISTANCE MODEL"+str(num_trial), figsize=None)
        fig.set_size_inches(fig.get_size_inches()[0],fig.get_size_inches()[1])
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(param_names[0], labelpad=10)
        ax.set_ylabel(param_names[1], labelpad=10)
        ax.set_zlabel('[cm]', rotation='vertical', labelpad=10)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks_L)
        ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        ax.tick_params(axis='x', direction='out', pad=-5)
        ax.tick_params(axis='y', direction='out', pad=-3)
        ax.tick_params(axis='z', direction='out', pad=5)
        ax.plot_surface(X, Y, model_L, rstride=1, cstride=1, 
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.savefig(model_object.dirname + \
                    "/img_training_trial#{:3d}_dist.png".format(num_trial), 
                    format="png")
        
        # SELECTION FUNCTION - TOP VIEW
        fig = plt.figure("SELECTION FCN"+str(num_trial), figsize=None)
        fig.set_size_inches(fig.get_size_inches()[0],fig.get_size_inches()[1])
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(param_names[0])
        ax1.set_ylabel(param_names[1])
        # ax1.set_xlim(len(dim1), 0)
        ax1.set_xlim(0, len(dim1))
        ax1.set_ylim(0, len(dim2))
        # ax1.set_xticks(np.linspace(len(dim1)-1, -1, 5))
        ax1.set_xticks(np.linspace(-1, len(dim1), 5))
        ax1.set_yticks(np.linspace(-1, len(dim2), 5))
        ax1.set_xticklabels([str(x) for x in xticks]), 
        ax1.set_yticklabels([str(y) for y in yticks1])
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        sidf = ax1.imshow(model_select, cmap=cm.summer, origin='lower')
        for spine in ax1.spines.values():
            spine.set_visible(False)
        # add also the trial points
        for tr in model_object.coord_explored:
            if list(tr) in [list(x) for x in model_object.coord_failed]:
                ax1.scatter(x=tr[1], y=tr[0], c='r', s=15)
            else:
                ax1.scatter(x=tr[1], y=tr[0], c='c', s=15)
        cbar = plt.colorbar(sidf, shrink=0.5, aspect=20, pad = 0.17, 
                            orientation='horizontal', ticks=[0.0, 0.5, 1.0])
        sidf.set_clim(-0.001, 1.001)
        plt.savefig(model_object.dirname + \
                    "/img_training_trial#{:3d}_select.png".format(num_trial), 
                    format="png")
        
        # # PENALISATION IDF
        # ax = plt.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
        # ax.set_title('Penalisation function: '+str(len(model_object.coord_failed))+' points')
        # ax.set_ylabel(param_names[1], labelpad=5)
        # ax.set_xlabel(param_names[0], labelpad=5)
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)
        # ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        # ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        # ax.tick_params(axis='x', direction='out', pad=-5)
        # ax.tick_params(axis='y', direction='out', pad=-3)
        # ax.tick_params(axis='z', direction='out', pad=2)
        # # ax.set_zticks(zticks_PIDF)
        # ax.plot_surface(X, Y, (1-model_PIDF), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)
        # # UNCERTAINTY IDF
        # ax = plt.subplot2grid((2,6),(1, 2), colspan=2, projection='3d')
        # ax.set_title('Model uncertainty: '+str(model_object.returnUncertainty()))
        # ax.set_ylabel(param_names[1], labelpad=5)
        # ax.set_xlabel(param_names[0], labelpad=5)
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)
        # ax.set_zticks(zticks_unc)
        # ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        # ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        # ax.tick_params(axis='x', direction='out', pad=-5)
        # ax.tick_params(axis='y', direction='out', pad=-3)
        # ax.tick_params(axis='z', direction='out', pad=5)
        # ax.plot_surface(X, Y, model_var, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
        # # SELECTION FUNCTION IDF
        # ax = plt.subplot2grid((2,6),(1, 4), colspan=2, projection='3d')
        # ax.set_title('Selection function')
        # ax.set_ylabel(param_names[1], labelpad=5)
        # ax.set_xlabel(param_names[0], labelpad=5)
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)
        # ax.set_xticklabels([str(x) for x in xticks], rotation=41)
        # ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
        # ax.tick_params(axis='x', direction='out', pad=-5)
        # ax.tick_params(axis='y', direction='out', pad=-3)
        # ax.tick_params(axis='z', direction='out', pad=2)
        # surf = ax.plot_surface(X, Y, model_select, rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)
        # # add also the trial points
        # if show_points:
        #     for tr in model_object.coord_explored:
        #         if list(tr) in [list(x) for x in model_object.coord_failed]:
        #             ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='k', alpha=0.7)
        #         else:
        #             ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='m', alpha=0.7)
        
        # SAVEFIG
        # if isinstance(num_trial, str):
        #     fig.suptitle("Models and IDFs (num_iter: {}, resolution: {})".format(num_trial, len(dim1)), fontsize=16)
        #     plt.savefig(model_object.dirname+"/img_training_trial_{}.png".format(num_trial), format="png")
        # else:
        #     plt.savefig(model_object.dirname+"/img_training_trial#{num:03d}.png".format(num=num_trial), format="png")
        
        if model_object.show_plots:
            plt.show()
        else:
            plt.cla()



