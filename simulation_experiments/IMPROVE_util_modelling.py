"""
Author: Nemanja Rakicevic
Date  : January 2018
Description:
            Classes containing model functionalities

"""

import os
import time
from numpy.core.umath_tests import inner1d
import itertools
import numpy as np
import scipy as sp
import scipy.spatial
import pickle
from heapq import nlargest, nsmallest

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# TODO separate model class versions into multiple classes
# TODO make nice private methods with unitary functionalities
# TODO make nice callables

_EPS = 1e-8

class BaseModel:

    def __init__(self, parameter_list, experiment_type,
                kernel_kwargs,
                kernel_type='SE',   # SE, MT, RQ
                seed=1,
                model_kwargs,
                is_training=False,
                show_plots=False,
                ):

        # fix seed
        np.random.seed(seed)
        # copy into private variables
        self._param_list = parameter_list
        self._is_training = is_training
        # create save_path
        self._save_path = './DATA/'+self.exp_type+'/TRIAL_'+time.strftime("%Y%m%d_%Hh%M")

        # start initialising stuff
        self._build_model(kernel_kwargs, model_kwargs)


    self._kernel_fn = self._build_kernel(kernel_kwargs)


    def _build_kernel(self, kernel_kwargs):

        def kernel_function(self, a, b):
            """ SE squared exponential kernel """
            sigsq = 1
            # siglensq = 0.01 # 1 0.5 0.3 0.1 
            siglensq = self.other[1]
            sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
            return sigsq*np.exp(-.5 *sqdist)

        return kernel_function


    def _build_model(self, kernel_kwargs, model_kwargs):
        self._cov_const = model_kwargs['cov_const']
        self._cov = self._cov_const * np.eye(len(self._param_list))
        ## build penalisation
        ## build uncertainty
        ## build selection


    def update_model():
        pass


    def generate_sample():
        pass





#################################################################
    def loadModel(self):
        list_models = sorted([d for d in os.listdir('./DATA/'+self.exp_type+'/') if d[0:6]=='TRIAL_'])
        for idx, t in enumerate(list_models):
            print("("+str(idx)+")\t", t)
        test_num = input("\nEnter number of model to load > ")
        self.trial_dirname = './DATA/'+self.exp_type+'/'+list_models[int(test_num)]
        print("Loading: ",self.trial_dirname)
        with open(self.trial_dirname + "/data_training_model.dat", "rb") as m:
            tmp = pickle.load(m)
            if len(tmp)<6:
                self.model_list = tmp
                (self.mu_alpha, self.mu_L, self.model_uncertainty, self.penal_IDF, self.selection_IDF, self.param_list) = self.model_list[-1]
            else:
                # Load last from history 
                (self.mu_alpha, self.mu_L, self.model_uncertainty, self.penal_IDF, self.selection_IDF, self.param_list) = tmp


    def testModel(self, angle_s, dist_s, verbose=False):
        # Helper functions
        thrsh = 0.1
        def sqdist(x,y):
            return np.sqrt(x**2 + y**2)
        def getMeas(M_angle, M_dist, angle_s, dist_s):
            diff_angle = M_angle - angle_s
            diff_angle = (diff_angle - diff_angle.min())/(diff_angle.max() - diff_angle.min())
            diff_dist  = M_dist - dist_s
            diff_dist  = (diff_dist - diff_dist.min())/(diff_dist.max() - diff_dist.min())
            return sqdist(diff_angle, diff_dist)
        # Calculate goodness measure to select (angle, distance) pair which is closest to the desired one
        # M_meas  = getMeas(M_angle, M_dist, angle_s, dist_s)
        M_meas = sqdist(self.mu_alpha - angle_s, self.mu_L - dist_s)
        """
        Check for known constraints/failures if the values of the penal_IDF for the selected_coords 
        are above a cetrain threshold this is probably a bad idea
        """
        src = 0
        for cnt in range(len(M_meas.ravel())):
            src += 1
            # Get candidate movement parameter vector
            best_fit        = nsmallest(src, M_meas.ravel())
            selected_coord  = np.argwhere(M_meas==best_fit[src-1])[0]
            selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(len(self.param_list))])
            error_angle     = self.mu_alpha[tuple(selected_coord)] - angle_s
            error_dist      = self.mu_L[tuple(selected_coord)]  - dist_s
            if cnt%50 == 0 and cnt > 100:
                thrsh = cnt/1000.
                src = 0
            # print(self.penal_IDF[tuple(selected_coord)], "DIFF",thrsh)
            if self.penal_IDF[tuple(selected_coord)] < thrsh:
                # print("BREAK")
                break
            else:    
                # Continue to the next smallest number
                if verbose:
                    # print("--- generated coords:", selected_coord, "-> parameters:", selected_params)
                    # print("--- ESTIMATED ERRORS > chosen (", M_angle[tuple(coords1)],",",M_dist[tuple(coords1)],") - desired (",angle_s,",",dist_s,") = error (",error_angle1,",",error_dist1,")")
                    print("--- generated coords: {} -> parameters: {} (goodness measure: {})".format(selected_coord, selected_params, round(best_fit[src-1], 4)))
                    print("--- ESTIMATED ERRORS: chosen ({}, {}) - desired ({}, {}) = error ({}, {})".format(round(self.mu_alpha[tuple(selected_coord)], 4), round(self.mu_L[tuple(selected_coord)], 4), angle_s, dist_s, round(error_angle,4), round(error_dist,4)))
                    print("--- BAD SAMPLE #{}, resampling ...".format(cnt))
    
        if verbose:
            print("--- generated coords: {} -> parameters: {} (goodness measure: {})".format(selected_coord, selected_params, round(best_fit[src-1], 4)))
            print("--- ESTIMATED ERRORS: chosen ({}, {}) - desired ({}, {}) = error ({}, {})".format(round(self.mu_alpha[tuple(selected_coord)], 4), round(self.mu_L[tuple(selected_coord)], 4), angle_s, dist_s, round(error_angle,4), round(error_dist,4)))  
        # return vector to execute
        return selected_coord, selected_params


    def plotModel(self, trial_num, dimensions, param_names, show_points=False):
        # if trial_num%1==0 or trial_info.fail_status==0:
        # print "<- CHECK PLOTS"     

        mpl.rcParams.update({'font.size': 12})
        
        if len(self.mu_alpha):
            fig = plt.figure("DISTRIBUTIONs at step: "+str(trial_num), figsize=None)
            fig.set_size_inches(fig.get_size_inches()[0]*3,fig.get_size_inches()[1]*2)
            dim1 = self.param_list[dimensions[0]]
            dim2 = self.param_list[dimensions[1]]
            X, Y = np.meshgrid(dim2, dim1)
            # Values to plot
            if len(self.param_dims)>2:
                if self.param_dims[0]>1:
                    model_alpha  = self.mu_alpha[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_L      = self.mu_L[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_PIDF   = self.penal_IDF[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_var    = self.model_uncertainty[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_select = self.selection_IDF[:,:,3,3,4].reshape(len(dim1),len(dim2))

                    # model_alpha  = self.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_L      = self.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_PIDF   = self.penal_IDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_var    = self.model_uncertainty[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_select = self.selection_IDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                else:
                    model_alpha  = self.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_L      = self.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_PIDF   = self.penal_IDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_var    = self.model_uncertainty[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_select = self.selection_IDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
            else:
                model_alpha  = self.mu_alpha
                model_L      = self.mu_L
                model_PIDF   = self.penal_IDF
                model_var    = self.model_uncertainty
                model_select = self.selection_IDF
            # Set ticks
            xticks = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
            yticks = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 4).round(1)
            # xticks1 = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
            yticks1 = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(1)
            #
            zticks_alpha = np.linspace(self.mu_alpha.min(), self.mu_alpha.max(), 4).round()
            zticks_L = np.linspace(self.mu_L.min(), self.mu_L.max(), 4).round()
            zticks_unc = np.linspace(self.model_uncertainty.min(), self.model_uncertainty.max(), 4).round(2)
            # zticks_PIDF = np.linspace(self.penal_IDF.min(), self.penal_IDF.max(), 7).round(1)
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
            ax.plot_surface(X, Y, model_alpha, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
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
            ax.plot_surface(X, Y, model_L, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
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
            for tr in self.coord_explored:
                if list(tr) in [list(x) for x in self.failed_coords]:
                    ax1.scatter(x=tr[1], y=tr[0], c='r', s=15)
                else:
                    ax1.scatter(x=tr[1], y=tr[0], c='c', s=15)
            cbar = plt.colorbar(sidf, shrink=0.5, aspect=20, pad = 0.17, orientation='horizontal', ticks=[0.0, 0.5, 1.0])
            sidf.set_clim(-0.001, 1.001)

            # PENALISATION IDF
            ax = plt.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
            ax.set_title('Penalisation function: '+str(len(self.failed_coords))+' points')
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
            ax.plot_surface(X, Y, (1-model_PIDF), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)
            # UNCERTAINTY IDF
            ax = plt.subplot2grid((2,6),(1, 2), colspan=2, projection='3d')
            ax.set_title('Model uncertainty: '+str(self.returnUncertainty()))
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
            ax.plot_surface(X, Y, model_var, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
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
            surf = ax.plot_surface(X, Y, model_select, rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)
            # add also the trial points
            if show_points:
                for tr in self.coord_explored:
                    if list(tr) in [list(x) for x in self.failed_coords]:
                        ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='k', alpha=0.7)
                    else:
                        ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='m', alpha=0.7)
            
            # SAVEFIG
            if isinstance(trial_num, str):
                fig.suptitle("Models and IDFs (num_iter: {}, resolution: {})".format(trial_num, len(dim1)), fontsize=16)
                plt.savefig(self.trial_dirname+"/img_training_trial_{}.svg".format(trial_num), format="svg")
            else:
                plt.savefig(self.trial_dirname+"/img_training_trial#{num:03d}.svg".format(num=trial_num), format="svg")
            
            if self.show_plots:
                plt.show()
            else:
                plt.cla()


    def plotModelFig(self, trial_num, dimensions, param_names, show_points=False):
        # if trial_num%1==0 or trial_info.fail_status==0:
        # print "<- CHECK PLOTS"     

        mpl.rcParams.update({'font.size': 14})
        
        if len(self.mu_alpha):
            dim1 = self.param_list[dimensions[0]]
            dim2 = self.param_list[dimensions[1]]
            X, Y = np.meshgrid(dim2, dim1)
            # Values to plot
            if len(self.param_dims)>2:
                if self.param_dims[0]>1:
                    model_alpha  = self.mu_alpha[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_L      = self.mu_L[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_PIDF   = self.penal_IDF[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_var    = self.model_uncertainty[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_select = self.selection_IDF[:,:,3,3,4].reshape(len(dim1),len(dim2))

                    # model_alpha  = self.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_L      = self.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_PIDF   = self.penal_IDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_var    = self.model_uncertainty[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_select = self.selection_IDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                else:
                    model_alpha  = self.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_L      = self.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_PIDF   = self.penal_IDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_var    = self.model_uncertainty[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_select = self.selection_IDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
            else:
                model_alpha  = self.mu_alpha
                model_L      = self.mu_L
                model_PIDF   = self.penal_IDF
                model_var    = self.model_uncertainty
                model_select = self.selection_IDF
            # Set ticks
            xticks = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
            yticks = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 4).round(1)
            # xticks1 = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
            yticks1 = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(1)
            #
            zticks_alpha = np.linspace(self.mu_alpha.min(), self.mu_alpha.max(), 4).round()
            zticks_L = np.linspace(self.mu_L.min(), self.mu_L.max(), 4).round()
            zticks_unc = np.linspace(self.model_uncertainty.min(), self.model_uncertainty.max(), 4).round(2)
            # zticks_PIDF = np.linspace(self.penal_IDF.min(), self.penal_IDF.max(), 7).round(1)
            

            # ANGLE MODEL
            fig = plt.figure("ANGLE MODEL"+str(trial_num), figsize=None)
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
            ax.plot_surface(X, Y, model_alpha, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.savefig(self.trial_dirname+"/img_training_trial#{num:03d}_angle.svg".format(num=trial_num), format="svg")
            
            # DISTANCE MODEL
            fig = plt.figure("DISTANCE MODEL"+str(trial_num), figsize=None)
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
            ax.plot_surface(X, Y, model_L, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.savefig(self.trial_dirname+"/img_training_trial#{num:03d}_dist.svg".format(num=trial_num), format="svg")
            
            # SELECTION FUNCTION - TOP VIEW
            fig = plt.figure("SELECTION FCN"+str(trial_num), figsize=None)
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
            for tr in self.coord_explored:
                if list(tr) in [list(x) for x in self.failed_coords]:
                    ax1.scatter(x=tr[1], y=tr[0], c='r', s=15)
                else:
                    ax1.scatter(x=tr[1], y=tr[0], c='c', s=15)
            cbar = plt.colorbar(sidf, shrink=0.5, aspect=20, pad = 0.17, orientation='horizontal', ticks=[0.0, 0.5, 1.0])
            sidf.set_clim(-0.001, 1.001)
            plt.savefig(self.trial_dirname+"/img_training_trial#{num:03d}_select.svg".format(num=trial_num), format="svg")
            
            # # PENALISATION IDF
            # ax = plt.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
            # ax.set_title('Penalisation function: '+str(len(self.failed_coords))+' points')
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
            # ax.set_title('Model uncertainty: '+str(self.returnUncertainty()))
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
            #     for tr in self.coord_explored:
            #         if list(tr) in [list(x) for x in self.failed_coords]:
            #             ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='k', alpha=0.7)
            #         else:
            #             ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='m', alpha=0.7)
            
            # SAVEFIG
            # if isinstance(trial_num, str):
            #     fig.suptitle("Models and IDFs (num_iter: {}, resolution: {})".format(trial_num, len(dim1)), fontsize=16)
            #     plt.savefig(self.trial_dirname+"/img_training_trial_{}.svg".format(trial_num), format="svg")
            # else:
            #     plt.savefig(self.trial_dirname+"/img_training_trial#{num:03d}.svg".format(num=trial_num), format="svg")
            
            if self.show_plots:
                plt.show()
            else:
                plt.cla()

