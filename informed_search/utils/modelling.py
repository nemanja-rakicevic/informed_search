
"""
Author:     Nemanja Rakicevic
Date  :     January 2018
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

_EPS = 1e-8   # 0.00005

class BaseModel(object):

    def __init__(self, parameter_list, 
                       dirname,
                       kernel_name='se',   # SE, MT, RQ
                       kernel_lenscale=0.1,
                       kernel_sigma=1,
                       cov_coeff=1,
                       seed=100,
                       is_training=False,
                       show_plots=False, **kwargs):
        # Fix numpy seed
        np.random.seed(seed)
        self.show_plots = show_plots

        # Generate experiment directory
        self.dirname = dirname
        os.makedirs(self.dirname+'/plots')

        # Initialise parameter space
        self.param_list = parameter_list
        self.param_space = \
                np.array([xs for xs in itertools.product(*self.param_list)])
        self.param_dims = tuple([len(i) for i in self.param_list])
        self.n_coords = len(self.param_space)
        self.n_param = len(self.param_list)


        # Initialise statistics
        self.coord_explored = []
        self.coord_failed = []

        # Initialise attributes 
        self._build_model(cov_coeff)
        self._build_kernel(kernel_name, kernel_lenscale, kernel_sigma)

    def _build_model(self, cov_coeff):
        # Prior distribution
        self.prior_init = np.ones(self.param_dims)/np.product(self.param_dims)
        # Create covariance matrix
        self.cov_coeff = cov_coeff
        self.cov_matrix = self.cov_coeff * np.eye(self.n_param)
        ## build penalisation
        ## build uncertainty
        ## build selection

        # Initialise search components
        self.pidf = self.prior_init
        self.uidf = self.prior_init
        self.sidf = self.prior_init

        # Initialise model components
        self.model_list = []
        self.mu_alpha = np.zeros(self.param_dims)
        self.mu_L = np.zeros(self.param_dims)
        self.var_alpha = np.ones(self.param_dims)
        self.var_L = np.ones(self.param_dims)
        self.model_uncertainty = np.ones(self.param_dims)
        ###
        # self.gp_class = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        self.mu_pidf = 0.5*np.ones(self.param_dims)
        self.var_pidf = np.ones(self.param_dims)
        self.mu_uncert = 0*np.ones(self.param_dims)
        self.var_uncert = np.ones(self.param_dims)
        self.current_uncertainty = self.model_uncertainty
        self.delta_uncertainty = []


### CAN ALSO DEFINE KERNELS IN A DIFFRENT FILE AND CALL BY NAME
    def _build_kernel(self, kernel_name, kernel_lenscale, kernel_sigma):
        """ Construct the kernel function and initialise Kss """
        if kernel_name == 'se':
            # SE squared exponential kernel
            def se_kernel(a, b):
                # kernel_lenscale = 0.01 # 1 0.5 0.3 0.1 
                sqdist = (1. / kernel_lenscale) * \
                         sp.spatial.distance.cdist(a, b, 'sqeuclidean')
                return kernel_sigma * np.exp(-.5 * sqdist)
            self.kernel_fn = se_kernel
        elif kernel_name == 'mt':
            def mat_kernel(a, b):
                # MT Matern 5/2 kernel
                # kernel_lenscale = 0.03 # 1
                sqdist = (1. / kernel_lenscale) * \
                         sp.spatial.distance.cdist(a, b, 'sqeuclidean')
                return kernel_sigma \
                       * (1 + np.sqrt(5 * sqdist) + 5 * sqdist / 3.) \
                       * np.exp(-np.sqrt(5.*sqdist))
            self.kernel_fn = mat_kernel
        elif kernel_name == 'rq':
            def rq_kernel(a, b):
                # RQ rational quadratic kernel
                # kernel_lenscale = 1
                alpha = a.shape[1] / 2. #a.shape[1]/2. #np.exp(1) #len(a)/2.
                sqdist = (1. / kernel_lenscale) * \
                         sp.spatial.distance.cdist(a, b, 'sqeuclidean')
                return kernel_sigma * np.power(1 + 0.5 * sqdist / alpha, -alpha)
            self.kernel_fn = rq_kernel
        else:
            raise AttributeError("Undefined kernel name!")
        # Construct Kss matrix
        self.Kss = self.kernel_fn(a=self.param_space, b=self.param_space)
        

    @property
    def uncertainty(self):
        return round(self.model_uncertainty.mean(), 4)


    @property
    def components(self):
        return self.mu_alpha, self.mu_L, self.var_alpha, self.var_L



    def save_model(self):
        self.model_list.append([self.mu_alpha, self.mu_L, self.model_uncertainty, self.pidf, self.sidf, self.param_list])
        with open(self.dirname + "/data_training_model.dat", "wb") as m:
            pickle.dump(self.model_list, m, protocol=pickle.HIGHEST_PROTOCOL)


    def load_model(self):
        list_models = sorted([d for d in os.listdir('./DATA/'+self.exp_type+'/') if d[0:6]=='TRIAL_'])
        for idx, t in enumerate(list_models):
            print("("+str(idx)+")\t", t)
        test_num = input("\nEnter number of model to load > ")
        self.dirname = './DATA/'+self.exp_type+'/'+list_models[int(test_num)]


        print("Loading: ",self.dirname)
        with open(self.dirname + "/data_training_model.dat", "rb") as m:
            tmp = pickle.load(m)
            if len(tmp)<6:
                self.model_list = tmp
                (self.mu_alpha, self.mu_L, self.model_uncertainty, self.pidf, self.sidf, self.param_list) = self.model_list[-1]
            else:
                # Load last from history 
                (self.mu_alpha, self.mu_L, self.model_uncertainty, self.pidf, self.sidf, self.param_list) = tmp
 


### MOVE TO TESTING??
    def test_model(self):
        """ Generate test parameter coordinates for performance evaluation """
        pass


    def update_model():
        raise NotImplementedError


    def generate_sample():
        raise NotImplementedError


#################################################################


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
        Check for known constraints/failures if the values of the pidf for the selected_coords 
        are above a cetrain threshold this is probably a bad idea
        """
        src = 0
        for cnt in range(len(M_meas.ravel())):
            src += 1
            # Get candidate movement parameter vector
            best_fit        = nsmallest(src, M_meas.ravel())
            selected_coord  = np.argwhere(M_meas==best_fit[src-1])[0]
            selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
            error_angle     = self.mu_alpha[tuple(selected_coord)] - angle_s
            error_dist      = self.mu_L[tuple(selected_coord)]  - dist_s
            if cnt%50 == 0 and cnt > 100:
                thrsh = cnt/1000.
                src = 0
            # print(self.pidf[tuple(selected_coord)], "DIFF",thrsh)
            if self.pidf[tuple(selected_coord)] < thrsh:
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
                    model_PIDF   = self.pidf[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_var    = self.model_uncertainty[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_select = self.sidf[:,:,3,3,4].reshape(len(dim1),len(dim2))

                    # model_alpha  = self.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_L      = self.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_PIDF   = self.pidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_var    = self.model_uncertainty[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_select = self.sidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                else:
                    model_alpha  = self.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_L      = self.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_PIDF   = self.pidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_var    = self.model_uncertainty[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_select = self.sidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
            else:
                model_alpha  = self.mu_alpha
                model_L      = self.mu_L
                model_PIDF   = self.pidf
                model_var    = self.model_uncertainty
                model_select = self.sidf
            # Set ticks
            xticks = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
            yticks = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 4).round(1)
            # xticks1 = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
            yticks1 = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(1)
            #
            zticks_alpha = np.linspace(self.mu_alpha.min(), self.mu_alpha.max(), 4).round()
            zticks_L = np.linspace(self.mu_L.min(), self.mu_L.max(), 4).round()
            zticks_unc = np.linspace(self.model_uncertainty.min(), self.model_uncertainty.max(), 4).round(2)
            # zticks_PIDF = np.linspace(self.pidf.min(), self.pidf.max(), 7).round(1)
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
                if list(tr) in [list(x) for x in self.coord_failed]:
                    ax1.scatter(x=tr[1], y=tr[0], c='r', s=15)
                else:
                    ax1.scatter(x=tr[1], y=tr[0], c='c', s=15)
            cbar = plt.colorbar(sidf, shrink=0.5, aspect=20, pad = 0.17, orientation='horizontal', ticks=[0.0, 0.5, 1.0])
            sidf.set_clim(-0.001, 1.001)

            # PENALISATION IDF
            ax = plt.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
            ax.set_title('Penalisation function: '+str(len(self.coord_failed))+' points')
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
                    if list(tr) in [list(x) for x in self.coord_failed]:
                        ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='k', alpha=0.7)
                    else:
                        ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='m', alpha=0.7)
            
            # SAVEFIG
            if isinstance(trial_num, str):
                fig.suptitle("Models and IDFs (num_iter: {}, resolution: {})".format(trial_num, len(dim1)), fontsize=16)
                plt.savefig(self.dirname+"/img_training_trial_{}.svg".format(trial_num), format="svg")
            else:
                plt.savefig(self.dirname+"/img_training_trial#{num:03d}.svg".format(num=trial_num), format="svg")
            
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
                    model_PIDF   = self.pidf[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_var    = self.model_uncertainty[:,:,3,3,4].reshape(len(dim1),len(dim2))
                    model_select = self.sidf[:,:,3,3,4].reshape(len(dim1),len(dim2))

                    # model_alpha  = self.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_L      = self.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_PIDF   = self.pidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_var    = self.model_uncertainty[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    # model_select = self.sidf[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                else:
                    model_alpha  = self.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_L      = self.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_PIDF   = self.pidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_var    = self.model_uncertainty[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_select = self.sidf[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
            else:
                model_alpha  = self.mu_alpha
                model_L      = self.mu_L
                model_PIDF   = self.pidf
                model_var    = self.model_uncertainty
                model_select = self.sidf
            # Set ticks
            xticks = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
            yticks = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 4).round(1)
            # xticks1 = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(1)
            yticks1 = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(1)
            #
            zticks_alpha = np.linspace(self.mu_alpha.min(), self.mu_alpha.max(), 4).round()
            zticks_L = np.linspace(self.mu_L.min(), self.mu_L.max(), 4).round()
            zticks_unc = np.linspace(self.model_uncertainty.min(), self.model_uncertainty.max(), 4).round(2)
            # zticks_PIDF = np.linspace(self.pidf.min(), self.pidf.max(), 7).round(1)
            

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
            plt.savefig(self.dirname+"/img_training_trial#{num:03d}_angle.svg".format(num=trial_num), format="svg")
            
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
            plt.savefig(self.dirname+"/img_training_trial#{num:03d}_dist.svg".format(num=trial_num), format="svg")
            
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
                if list(tr) in [list(x) for x in self.coord_failed]:
                    ax1.scatter(x=tr[1], y=tr[0], c='r', s=15)
                else:
                    ax1.scatter(x=tr[1], y=tr[0], c='c', s=15)
            cbar = plt.colorbar(sidf, shrink=0.5, aspect=20, pad = 0.17, orientation='horizontal', ticks=[0.0, 0.5, 1.0])
            sidf.set_clim(-0.001, 1.001)
            plt.savefig(self.dirname+"/img_training_trial#{num:03d}_select.svg".format(num=trial_num), format="svg")
            
            # # PENALISATION IDF
            # ax = plt.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
            # ax.set_title('Penalisation function: '+str(len(self.coord_failed))+' points')
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
            #         if list(tr) in [list(x) for x in self.coord_failed]:
            #             ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='k', alpha=0.7)
            #         else:
            #             ax.plot([dim2[tr[1]], dim2[tr[1]]], [dim1[tr[0]], dim1[tr[0]]], [model_select.min(), model_select.max()], linewidth=1, color='m', alpha=0.7)
            
            # SAVEFIG
            # if isinstance(trial_num, str):
            #     fig.suptitle("Models and IDFs (num_iter: {}, resolution: {})".format(trial_num, len(dim1)), fontsize=16)
            #     plt.savefig(self.dirname+"/img_training_trial_{}.svg".format(trial_num), format="svg")
            # else:
            #     plt.savefig(self.dirname+"/img_training_trial#{num:03d}.svg".format(num=trial_num), format="svg")
            
            if self.show_plots:
                plt.show()
            else:
                plt.cla()





##############################################################################
##############################################################################
##############################################################################
##############################################################################




class InformedSearch(BaseModel):
    """
        Proposed Informed Search model class
    """

    def update_model(self, info_list, save_progress=True, **kwargs):
        """
            Select successful trials to estimate the GPR model's mean and 
            variance, and the failed ones to update the penalisation IDF.
        """
        if len(info_list):

            # Successful trial: Update task models and PIDF
            if info_list[-1]['fail_status']==0:
                good_trials = np.vstack([[tr['parameters'], tr['ball_polar']] \
                                for tr in info_list if tr['fail_status']==0])
                good_params = good_trials[:, 0]
                good_fevals = good_trials[:, 1]
                # Update the Angle and Distance GPR models, as well as PIDF
                self.mu_alpha, self.var_alpha = self.update_GPR(good_params, 
                                                                good_fevals, 0)
                self.mu_L, self.var_L = self.update_GPR(good_params, 
                                                        good_fevals, 1)
                # Update PIDF
                self.update_PIDF(info_list[-1]['parameters'], failed=-1)

            # Failed trial: Update PIDF
            elif info_list[-1]['fail_status']>0:
                self.coord_failed = np.array([tr['coordinates'] for tr \
                                        in info_list if tr['fail_status']>0])
                self.update_PIDF(info_list[-1]['parameters'], failed=1)
           
            # All trials: Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.model_uncertainty = self.update_GPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.model_uncertainty = self.update_GPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_progress:
                self.saveModel()


    def update_GPR(self, Xtrain, Ytrain, label_indicator):
        """
            Perform Gaussian Process Regression using successful trials as 
            training points, and evaluate on the whole parameter space to get 
            the new model and uncertainty estimates.
        """
        # Xtrain = np.array(Xtrain).reshape(-1,self.n_param)
        Xtest = self.param_space
        # Calculate kernel matrices
        K = self.kernel_fn(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + _EPS*np.eye(len(Xtrain)))
        Ks = self.kernel_fn(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # Get posterior MU and SIGMA
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        if label_indicator != -1:
            Ytrain = Ytrain[:, label_indicator].reshape(-1,1)
            mu_post = np.dot(Lk.T, np.linalg.solve(L, Ytrain))
            # Return the matrix version
            return mu_post.reshape(self.param_dims), \
                   var_post.reshape(self.param_dims)#/np.sum(var_post)
        else:
            return var_post.reshape(self.param_dims)#/np.sum(var_post)


    def update_PIDF(self, mu, failed=1):
        """ 
            Update the PIDF based on the failed trials, by centering a
            negative multivariate Gaussian on the point in the parameter space 
            corresponding to the movement vector of the failed trial.

            For successful trials a positive Gaussian with a larger variance 
            is used.
        """
        previous_pidf = self.pidf.copy()
        # Modify the covariance matrix, depending on trial outcome  
        # Failed trial
        if failed == 1:
            """ VERSION 1
            Use most diverse samples. Make the parameters that change often change less (make cov smaller and wider), 
            and the ones which don't push to change more (make cov larger and narrower)
            """
            # fcnt = np.array([len(np.unique(np.array(self.coord_failed)[:,f])) for f in range(self.n_param) ], np.float)
            # cov_coeff = 1 + (fcnt-fcnt.mean())/(fcnt.max())
            """ VERSION 2
            Get samples with most repeated elements. Leave the parameters that change often as they are (leave cov as is),
            and the ones which don't push to change more (make cov larger and narrower) 
            """
            fcnt = np.array([max(np.bincount(self.coord_failed[:,f])) \
                                    for f in range(self.n_param)], np.float)            
            cov_coeff = 1 + (fcnt-fcnt.min())/fcnt.max()
       
        # Succesful trial
        elif failed == -1:
            fcnt = 0
            cov_coeff = 0.5*np.ones(self.n_param)
            # # Update the covariance matrix taking into account stationary parameters
            # if len(self.coord_failed)>0:
            #     good_coords = set(map(tuple, self.coord_explored)) - set(map(tuple,self.coord_failed))
            #     good_coords = np.array(list(good_coords)) 
            #     """VERSION 1: Get most diverse samples"""
            #     # fcnt = np.array([len(np.unique(np.array(self.coord_failed)[:,f])) for f in range(self.n_param) ], np.float)
            #     # cov_coeff = (1-(fcnt-fcnt.mean())/(fcnt.max()))
            #     """VERSION 2: Get samples with most repeated elements"""
            #     # fcnt = np.array([ max(np.bincount(np.array(self.good_coords)[:,f])) for f in range(self.n_param) ], np.float)            
            #     # cov_coeff = 1+(1-(fcnt)/(fcnt.max()))
            # else:
            #     cov_coeff = 0.5*np.ones(self.n_param)

        # Scale covariance diagonal elements
        for idx, cc in enumerate(cov_coeff):
            self.cov_matrix[idx,idx] = self.cov_coeff * cc
        # Estimate the contributing Gaussian
        trial_gaussian = np.reshape(self.generate_pdf_matrix(self.param_space, 
                                                             mu, 
                                                             self.cov_matrix), 
                                    self.param_dims)
        trial_gaussian *= failed
        trial_gaussian /= (trial_gaussian.max() + _EPS)

        # Update the penalisation IDF
        self.pidf = np.clip(previous_pidf+trial_gaussian, self.prior_init, 1.)

        # Log penalized points
        logger.info("PIDF: Penalised {} peaks from {} combinations.".format(
            len(np.argwhere(self.pidf.round(2)==np.max(self.pidf.round(2)))),
            len(self.coord_failed)))
        


    def generate_sample(self, info_list, **kwargs):
        """
            Generate the movement parameter vector to evaluate next, 
            based on the GPR model uncertainty and the penalisation IDF.
        """
        # Combine the model uncertainty with the PIDF 
        # model_var = (self.prior_init * self.model_uncertainty)/np.sum(self.prior_init * self.model_uncertainty)
        sidf = 1.0 * self.model_uncertainty * (1 - self.pidf)#/np.sum(1-self.pidf)
        
        # Scale the selection IDF
        # info_pdf /= np.sum(info_pdf)
        self.sidf = sidf / (sidf.max() + _EPS)

        # Check if the parameters have already been used
        temp_good = np.array([])
        cnt = 1
        while len(temp_good)==0:
            sample = np.array([sidf==c for c in nlargest(cnt*1, sidf.ravel())])
            sample = sample.reshape([-1]+list(self.param_dims))
            temp = np.argwhere(sample)[:,1:]
            temp_good = np.array(list(set(map(tuple, temp)) \
                        - set(map(tuple, self.coord_explored)))) 
            cnt += 1
            if cnt > self.n_coords:
                logger.info("All parameters have been explored!\tEXITING...")
                break

        # Convert from coordinates to parameters
        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] \
                                                for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        logger.info("Generated coords: [{}] -> parameters: [{}]".format(
                                            selected_coord, selected_params))
        # Return the next parameter vector
        return selected_coord, selected_params



    def generate_pdf_matrix(self, x_sample, mu, cov):
        """ 
            Create a multivariate gaussian over the parameter space 
        """
        tmp = np.dot((x_sample - mu), np.linalg.inv(cov))
        tmp_T = (x_sample - mu).T
        denom = np.sqrt(2 * np.pi * np.linalg.det(cov))
        f = (1 / denom) * np.exp(-0.5*inner1d(tmp, tmp_T.T))
        return f 





##############################################################################
##############################################################################
##############################################################################
##############################################################################







class UIDFSearch(BaseModel):

    def update_model(self, info_list, save_progress=True, **kwargs):
        """
        Select successful trials to estimate the GPR model mean and variance,
        and the failed ones to update the penalisation IDF.
        """
        if len(info_list):
            # Successful trial: Update task models
            if info_list[-1]['fail_status']==0:
                # good_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list if tr['fail_status']==0])
                # good_params = good_trials[:,0]
                # good_fevals = good_trials[:,1]
                good_params = np.array([tr['parameters'] for tr in info_list if tr['fail_status']==0])
                good_fevals = np.array([tr['ball_polar'] for tr in info_list if tr['fail_status']==0])
                # Estimate the Angle and Distance GPR models, as well as PIDF
                self.mu_alpha, self.var_alpha = self.update_GPR(good_params, good_fevals, 0)
                self.mu_L,     self.var_L     = self.update_GPR(good_params, good_fevals, 1)
                self.update_PIDF(info_list[-1]['parameters'], failed=-1)
            # Failed trial: Update PIDF
            elif info_list[-1]['fail_status']>0:
                self.coord_failed = [tr['coordinates'] for tr in info_list if tr['fail_status']>0]
                self.update_PIDF(info_list[-1]['parameters'], failed=1)
            # All trials: Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.model_uncertainty = self.update_GPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.model_uncertainty = self.update_GPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_progress:
                self.saveModel()


    def update_GPR(self, Xtrain, Ytrain, label_indicator):
        """
        Update GPR uncertainty over the parameter space.
        """
        if label_indicator == -1:
            # Xtrain = np.array(Xtrain).reshape(-1,self.n_param)
            Xtest = self.param_space
            # Calculate kernel matrices
            K = self.kernel_fn(Xtrain, Xtrain)
            L = np.linalg.cholesky(K + _EPS*np.eye(len(Xtrain)))
            Ks = self.kernel_fn(Xtrain, Xtest)
            Lk = np.linalg.solve(L, Ks)
            # Get overall model uncertainty
            return np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0)).reshape(self.param_dims)

        """
        Perform Gaussian Process Regression using good performed trials as training points,
        and evaluate on the whole parameter space to get the new model and uncertainty estimates.
        """
        Ytrain = Ytrain[:, label_indicator].reshape(-1,1)
        Xtest = self.param_space
        # Calculate kernel matrices
        K = self.kernel_fn(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + _EPS*np.eye(len(Xtrain)))
        Ks = self.kernel_fn(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # Get posterior MU and SIGMA
        mu_post = np.dot(Lk.T, np.linalg.solve(L, Ytrain))
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        # Return the matrix version
        return mu_post.reshape(self.param_dims), var_post.reshape(self.param_dims)#/np.sum(var_post)



    def update_PIDF(self, mu, failed=1):
        pass
        

    def generate_sample(self, info_list, **kwargs):
        """
        Generate the movement parameter vector to evaluate next, 
        based ONLY on the GPR model uncertainty.
        """
        # Combine the model uncertainty with the penalisation IDF to get the most informative point  
        # model_var = (self.prior_init * self.model_uncertainty)/np.sum(self.prior_init * self.model_uncertainty)
        sidf = 1.0 * self.model_uncertainty #* (1 - self.pidf)#/np.sum(1-self.pidf)
        # info_pdf /= np.sum(info_pdf)
        self.sidf = sidf /(sidf.max() + _EPS)
        # Check if the parameters have already been used
        temp_good = np.array([])
        cnt=1
        while len(temp_good)==0:
            temp = np.argwhere(np.array([sidf==c for c in nlargest(cnt*1, sidf.ravel())]).reshape(tuple(np.append(-1, self.param_dims))))[:,1:]
            temp_good = set(map(tuple, temp)) - set(map(tuple,self.coord_explored))
            temp_good = np.array(list(temp_good)) 
            cnt+=1
            if cnt > self.n_coords:
                print("ALL COMBINATIONS HAVE BEEN EXPLORED!\nEXITING...")
                break

        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # print("---sidf provided:", len(temp),"of which", len(temp_good),"unexplored (among the top",cnt-1,")" )
        print("--- generated coords: {}\t-> parameters: {}".format(selected_coord, selected_params))
        # return the next sample vector
        return selected_coord, selected_params






##############################################################################
##############################################################################
##############################################################################
##############################################################################





class EntropySearch(BaseModel):

    def update_model(self, info_list, save_progress=True, **kwargs):
        """
        Select successful trials to estimate the GPR model mean and variance,
        and the failed ones to update the penalisation IDF.
        """
        if len(info_list):
            # Successful trial: Update task models
            if info_list[-1]['fail_status']==0:
                # good_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list if tr['fail_status']==0])
                # good_params = good_trials[:,0]
                # good_fevals = good_trials[:,1]
                good_params = np.array([tr['parameters'] for tr in info_list if tr['fail_status']==0])
                good_fevals = np.array([tr['ball_polar'] for tr in info_list if tr['fail_status']==0])
                # Estimate the Angle and Distance GPR models, as well as PIDF
                self.mu_alpha, self.var_alpha = self.update_GPR(good_params, good_fevals, 0)
                self.mu_L,     self.var_L     = self.update_GPR(good_params, good_fevals, 1)
                self.update_PIDF(info_list[-1]['parameters'], failed=-1)
            # Failed trial: Update PIDF
            elif info_list[-1]['fail_status']>0:
                self.coord_failed = [tr['coordinates'] for tr in info_list if tr['fail_status']>0]
                self.update_PIDF(info_list[-1]['parameters'], failed=1)
            # All trials: Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.model_uncertainty = self.update_GPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.model_uncertainty = self.update_GPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_progress:
                self.saveModel()


    def update_GPR(self, Xtrain, Ytrain, label_indicator):
        """
        Update GPR uncertainty over the parameter space.
        """
        if label_indicator == -1:
            # Xtrain = np.array(Xtrain).reshape(-1,self.n_param)
            Xtest = self.param_space
            # Calculate kernel matrices
            K = self.kernel_fn(Xtrain, Xtrain)
            L = np.linalg.cholesky(K + self.eps_var*np.eye(len(Xtrain)))
            Ks = self.kernel_fn(Xtrain, Xtest)
            Lk = np.linalg.solve(L, Ks)
            # Get overall model uncertainty
            return np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0)).reshape(self.param_dims)

        """
        Perform Gaussian Process Regression using good performed trials as training points,
        and evaluate on the whole parameter space to get the new model and uncertainty estimates.
        """
        Ytrain = Ytrain[:, label_indicator].reshape(-1,1)
        Xtest = self.param_space
        # Calculate kernel matrices
        K = self.kernel_fn(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + self.eps_var*np.eye(len(Xtrain)))
        Ks = self.kernel_fn(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # Get posterior MU and SIGMA
        mu_post = np.dot(Lk.T, np.linalg.solve(L, Ytrain))
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        # Return the matrix version
        return mu_post.reshape(self.param_dims), var_post.reshape(self.param_dims)#/np.sum(var_post)



    def update_PIDF(self):
        pass
        

    def generate_sample(self, info_list, **kwargs):
        """
        Generate the movement parameter vector to evaluate next, 
        based ONLY on the posterior distributions entropy
        """
        # Combine the model uncertainty with the penalisation IDF to get the most informative point  
        # model_var = (self.prior_init * self.model_uncertainty)/np.sum(self.prior_init * self.model_uncertainty)
        sidf = 0.5 * np.log(2 * np.pi * np.e * self.model_uncertainty )#* (1 - self.pidf)#/np.sum(1-self.pidf)
        # info_pdf /= np.sum(info_pdf)
        self.sidf = sidf /(sidf.max() + _EPS)
        # Check if the parameters have already been used
        temp_good = np.array([])
        cnt=1
        while len(temp_good)==0:
            temp = np.argwhere(np.array([sidf==c for c in nlargest(cnt*1, sidf.ravel())]).reshape(tuple(np.append(-1, self.param_dims))))[:,1:]
            temp_good = set(map(tuple, temp)) - set(map(tuple,self.coord_explored))
            temp_good = np.array(list(temp_good)) 
            cnt+=1
            if cnt > self.n_coords:
                print("ALL COMBINATIONS HAVE BEEN EXPLORED!\nEXITING...")
                break

        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # print("---sidf provided:", len(temp),"of which", len(temp_good),"unexplored (among the top",cnt-1,")" )
        print("--- generated coords: {}\t-> parameters: {}".format(selected_coord, selected_params))
        # return the next sample vector
        return selected_coord, selected_params









##############################################################################
##############################################################################
##############################################################################
##############################################################################





class RandomSearch(BaseModel):

    def update_model(self, info_list, save_progress=True, **kwargs):
        """
        Select successful trials to estimate the GPR model mean and variance,
        and the failed ones to update the penalisation IDF.
        """
        if len(info_list):
            # Successful trial: Update task models
            if info_list[-1]['fail_status']==0:
                # good_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list if tr['fail_status']==0])
                # good_params = good_trials[:,0]
                # good_fevals = good_trials[:,1]
                good_params = np.array([tr['parameters'] for tr in info_list if tr['fail_status']==0])
                good_fevals = np.array([tr['ball_polar'] for tr in info_list if tr['fail_status']==0])
                # Estimate the Angle and Distance GPR models, as well as PIDF
                self.mu_alpha, self.var_alpha = self.update_GPR(good_params, good_fevals, 0)
                self.mu_L,     self.var_L     = self.update_GPR(good_params, good_fevals, 1)
                self.update_PIDF(info_list[-1]['parameters'], failed=-1)
            # Failed trial: Update PIDF
            elif info_list[-1]['fail_status']>0:
                self.coord_failed = [tr['coordinates'] for tr in info_list if tr['fail_status']>0]
                self.update_PIDF(info_list[-1]['parameters'], failed=1)
            # All trials: Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.model_uncertainty = self.update_GPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.model_uncertainty = self.update_GPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_progress:
                self.saveModel()


    def update_GPR(self, Xtrain, Ytrain, label_indicator):
        """
        Update GPR uncertainty over the parameter space.
        """
        if label_indicator == -1:
            # Xtrain = np.array(Xtrain).reshape(-1,self.n_param)
            Xtest = self.param_space
            # Calculate kernel matrices
            K = self.kernel_fn(Xtrain, Xtrain)
            L = np.linalg.cholesky(K + _EPS*np.eye(len(Xtrain)))
            Ks = self.kernel_fn(Xtrain, Xtest)
            Lk = np.linalg.solve(L, Ks)
            # Get overall model uncertainty
            return np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0)).reshape(self.param_dims)

        """
        Perform Gaussian Process Regression using good performed trials as training points,
        and evaluate on the whole parameter space to get the new model and uncertainty estimates.
        """
        Ytrain = Ytrain[:, label_indicator].reshape(-1,1)
        Xtest = self.param_space
        # Calculate kernel matrices
        K = self.kernel_fn(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + _EPS*np.eye(len(Xtrain)))
        Ks = self.kernel_fn(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # Get posterior MU and SIGMA
        mu_post = np.dot(Lk.T, np.linalg.solve(L, Ytrain))
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        # Return the matrix version
        return mu_post.reshape(self.param_dims), var_post.reshape(self.param_dims)#/np.sum(var_post)



    def update_PIDF(self, mu, failed=1):
        pass
        

    def generate_sample(self, *args, **kwargs):
        """
        Generate the random movement parameter vector to evaluate next. (with replacement)
        """
        param_sizes = [range(i) for i in self.param_dims]
        temp = np.array([xs for xs in itertools.product(*param_sizes)])
        
        temp_good = np.array([])
        cnt=1
        while len(temp_good)==0:
            temp_sel = np.array([temp[np.random.choice(len(temp)),:]])
            temp_good = set(map(tuple, temp_sel)) - set(map(tuple,self.coord_explored))
            temp_good = np.array(list(temp_good))
            cnt+=1
            if cnt > self.n_coords:
                print("ALL COMBINATIONS HAVE BEEN EXPLORED!\nEXITING...")
                break

        selected_coord = temp_good[0]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # print("---random sampling provided:", len(temp),"of which", len(temp_good))#,"unexplored (among the top",cnt-1,")" 
        print("--- generated coords: {}\t-> parameters: {}".format(selected_coord, selected_params))
        # return the next sample vector
        return selected_coord, selected_params




##############################################################################
##############################################################################
##############################################################################
##############################################################################






class REVIEWERSearch(BaseModel):

    def update_model(self, info_list, save_progress=True, **kwargs):
        """
        Select successful trials to estimate the GPR model mean and variance,
        and the failed ones to update the penalisation IDF.
        """
        if len(info_list):
            # Successful trial: Update task models
            if info_list[-1]['fail_status']==0:
                # good_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list if tr['fail_status']==0])
                # good_params = good_trials[:,0]
                # good_fevals = good_trials[:,1]
                good_params = np.array([tr['parameters'] for tr in info_list if tr['fail_status']==0])
                good_fevals = np.array([tr['ball_polar'] for tr in info_list if tr['fail_status']==0])
                # Estimate the Angle and Distance GPR models, as well as PIDF
                self.mu_alpha, self.var_alpha = self.update_GPR(good_params, good_fevals, 0)
                self.mu_L,     self.var_L     = self.update_GPR(good_params, good_fevals, 1)
                # self.update_PIDF(info_list[-1]['parameters'], failed=-1)
            # Failed trial: Update PIDF
            # elif info_list[-1]['fail_status']>0:
            #     self.coord_failed = [tr['coordinates'] for tr in info_list if tr['fail_status']>0]
                # self.update_PIDF(info_list[-1]['parameters'], failed=1)

            # All trials: Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.model_uncertainty = self.update_GPR(all_trials, None, -1)
            # GPC for fail/success 
            all_nonfails = np.array([ 1 if tr['fail_status']>0 else 0 for tr in info_list])
            self.mu_pidf,     self.var_pidf     = self.update_GPR_reviewer(all_trials, all_nonfails)

            temp = self.returnUncertainty()
            self.delta_uncertainty.append(self.current_uncertainty - temp)
            self.current_uncertainty = temp
            assert len(self.delta_uncertainty)==len(info_list)
            self.mu_uncert,     self.var_uncert     = self.update_GPR_reviewer(all_trials, np.array(self.delta_uncertainty))


            # SAVE CURRENT MODEL
            if save_progress:
                self.saveModel()


    def update_GPR(self, Xtrain, Ytrain, label_indicator):
        """
        Update GPR uncertainty over the parameter space.
        """
        if label_indicator == -1:
            # Xtrain = np.array(Xtrain).reshape(-1,self.n_param)
            Xtest = self.param_space
            # Calculate kernel matrices
            K = self.kernel_fn(Xtrain, Xtrain)
            L = np.linalg.cholesky(K + _EPS*np.eye(len(Xtrain)))
            Ks = self.kernel_fn(Xtrain, Xtest)
            Lk = np.linalg.solve(L, Ks)
            # Get overall model uncertainty
            return np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0)).reshape(self.param_dims)

        """
        Perform Gaussian Process Regression using good performed trials as training points,
        and evaluate on the whole parameter space to get the new model and uncertainty estimates.
        """
        Ytrain = Ytrain[:, label_indicator].reshape(-1,1)
        Xtest = self.param_space
        # Calculate kernel matrices
        K = self.kernel_fn(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + _EPS*np.eye(len(Xtrain)))
        Ks = self.kernel_fn(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # Get posterior MU and SIGMA
        mu_post = np.dot(Lk.T, np.linalg.solve(L, Ytrain))
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        # Return the matrix version
        return mu_post.reshape(self.param_dims), var_post.reshape(self.param_dims)#/np.sum(var_post)



    def update_GPR_reviewer(self, Xtrain, Ytrain):
        Xtest = self.param_space
        # Calculate kernel matrices
        K = self.kernel_fn(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + _EPS*np.eye(len(Xtrain)))
        Ks = self.kernel_fn(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # Get posterior MU and SIGMA
        mu_post = np.dot(Lk.T, np.linalg.solve(L, Ytrain))
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        # Return the matrix version
        return mu_post.reshape(self.param_dims), var_post.reshape(self.param_dims)#/np.sum(var_post)


    def update_PIDF(self):
        pass
        

    def generate_sample(self, info_list, **kwargs):
        """
        - calculate only PI for samples classified as successful, 
        - for those on border, use their variance
        """
        sidf = np.zeros(self.param_dims)
        if len(self.delta_uncertainty)>0:
            part_PI = norm.cdf((self.mu_uncert - np.array(self.delta_uncertainty).max()) / np.sqrt(self.var_uncert))
        else:
            part_PI = norm.cdf((self.mu_uncert) / np.sqrt(self.var_uncert))

        part_PI[part_PI<=0.5] = 0.
        part_Vgs = self.var_pidf.copy()
        part_Vgs[self.mu_pidf != 0.5] = 0

        sidf += part_PI
        sidf += part_Vgs
        # info_pdf /= np.sum(info_pdf)
        self.sidf = sidf #/(sidf.max() + _EPS)
        # Check if the parameters have already been used
        temp_good = np.array([])
        cnt=1
        while len(temp_good)==0:
            temp = np.argwhere(np.array([sidf==c for c in nlargest(cnt*1, sidf.ravel())]).reshape(tuple(np.append(-1, self.param_dims))))[:,1:]
            temp_good = set(map(tuple, temp)) - set(map(tuple,self.coord_explored))
            temp_good = np.array(list(temp_good)) 
            cnt+=1
            if cnt > self.n_coords:
                print("ALL COMBINATIONS HAVE BEEN EXPLORED!\nEXITING...")
                break

        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # print("---sidf provided:", len(temp),"of which", len(temp_good),"unexplored (among the top",cnt-1,")" )
        print("--- generated coords: {}\t-> parameters: {}".format(selected_coord, selected_params))
        # return the next sample vector
        return selected_coord, selected_params





















