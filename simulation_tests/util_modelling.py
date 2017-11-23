
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


# COVARIANCE
# COV = 5


class InformedModel:
    def __init__(self, parameter_list, experiment_type, other=[5, 0.1, 1], test=False, show_plots=False, folder_name=False):
        self.exp_type = experiment_type
        self.show_plots = show_plots
        self.other = other
        if not test:
            self.param_list = parameter_list
            self.param_space = np.array([xs for xs in itertools.product(*self.param_list)])
            self.param_dims = np.array([len(i) for i in self.param_list])
            # Initialise to uniform distribution
            self.prior_init = np.ones(tuple(self.param_dims))/(np.product(self.param_dims))
            self.COVARIANCE = other[0]
            self.cov = self.COVARIANCE * np.eye(len(self.param_list))
            self.eps_var = 0.00005
            # Calculate Kernel for the whole parameter (test) spacece, self.param_space)
            self.Kss = self.kernel(self.param_space, self.param_space)
            #
            self.coord_explored = []
            self.failed_coords = []
            # self.failed_params = []
            # self.good_params = []
            # self.good_fevals = []
            ###
            self.penal_IDF = self.prior_init
            self.selection_IDF = self.prior_init
            ###
            self.model_list = []
            self.mu_alpha = np.zeros(tuple(self.param_dims))
            self.mu_L = np.zeros(tuple(self.param_dims))
            self.var_alpha = np.ones(tuple(self.param_dims))
            self.var_L = np.ones(tuple(self.param_dims))
            self.model_uncertainty = np.ones(tuple(self.param_dims))
            ###
            if folder_name:
                self.trial_dirname = './DATA/'+self.exp_type+'/TRIAL__'+folder_name
            else:
                self.trial_dirname = './DATA/'+self.exp_type+'/TRIAL_'+time.strftime("%Y%m%d_%Hh%M")
            os.makedirs(self.trial_dirname)
        np.random.seed(210*int(other[2])) #840 4

#### SELECT KERNEL ####

    def kernel(self, a, b):
        """ SE squared exponential kernel """
        sigsq = 1
        # siglensq = 0.01 # 1 0.5 0.3 0.1 
        siglensq = self.other[1]
        sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
        return sigsq*np.exp(-.5 *sqdist)

    # def kernel(self, a, b):
    #     """ MT Matern 5/2 kernel: """
    #     sigsq = 1
    #     # siglensq = 0.03 # 1
    #     siglensq = self.other[1]
    #     sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
    #     return sigsq * (1 + np.sqrt(5*sqdist) + 5*sqdist/3.) * np.exp(-np.sqrt(5.*sqdist))

    # def kernel(self, a, b):
    #     """ RQ rational quadratic kernel """
    #     sigsq = 1
    #     # siglensq = 1
    #     siglensq = self.other[1]
    #     alpha = a.shape[1]/2. #a.shape[1]/2. #np.exp(1) #len(a)/2.
    #     # print alpha
    #     sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
    #     return sigsq * np.power(1 + 0.5*sqdist/alpha, -alpha)

######################## 


    def generatePDF_matrix(self, x_sample, mu, cov):
        """ Make a multinomial gaussian over the parameter space """
        tmp = np.dot((x_sample - mu), cov)
        tmp_T = (x_sample - mu).T
        f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*inner1d(tmp,tmp_T.T))
        return f 


    def updateModel(self, info_list, save_progress=True):
        """
        Select successful trials to estimate the GPR model mean and variance,
        and the failed ones to update the penalisation IDF.
        """
        if len(info_list):
            # Successful trial: Update task models
            if info_list[-1]['fail']==0:
                # good_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list if tr['fail']==0])
                # good_params = good_trials[:,0]
                # good_fevals = good_trials[:,1]
                good_params = np.array([tr['parameters'] for tr in info_list if tr['fail']==0])
                good_fevals = np.array([tr['ball_polar'] for tr in info_list if tr['fail']==0])
                # Estimate the Angle and Distance GPR models, as well as PIDF
                self.mu_alpha, self.var_alpha = self.updateGPR(good_params, good_fevals, 0)
                self.mu_L,     self.var_L     = self.updateGPR(good_params, good_fevals, 1)
                self.updatePIDF(info_list[-1]['parameters'], failed=-1)
            # Failed trial: Update PIDF
            elif info_list[-1]['fail']>0:
                self.failed_coords = [tr['coordinates'] for tr in info_list if tr['fail']>0]
                self.updatePIDF(info_list[-1]['parameters'], failed=1)
            # All trials: Update UIDF
            all_trials = np.array([tr['parameters'] for tr in info_list])
            self.model_uncertainty = self.updateGPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.model_uncertainty = self.updateGPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_progress:
                self.saveModel()


    def updateGPR(self, Xtrain, Ytrain, label_indicator):
        """
        Update GPR uncertainty over the parameter space.
        """
        if label_indicator == -1:
            # Xtrain = np.array(Xtrain).reshape(-1,len(self.param_list))
            Xtest = self.param_space
            # Calculate kernel matrices
            K = self.kernel(Xtrain, Xtrain)
            L = np.linalg.cholesky(K + self.eps_var*np.eye(len(Xtrain)))
            Ks = self.kernel(Xtrain, Xtest)
            Lk = np.linalg.solve(L, Ks)
            # Get overall model uncertainty
            return np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0)).reshape(tuple(self.param_dims))

        """
        Perform Gaussian Process Regression using good performed trials as training points,
        and evaluate on the whole parameter space to get the new model and uncertainty estimates.
        """
        Ytrain = Ytrain[:, label_indicator].reshape(-1,1)
        Xtest = self.param_space
        # Calculate kernel matrices
        K = self.kernel(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + self.eps_var*np.eye(len(Xtrain)))
        Ks = self.kernel(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # Get posterior MU and SIGMA
        mu_post = np.dot(Lk.T, np.linalg.solve(L, Ytrain))
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        # Return the matrix version
        return mu_post.reshape(tuple(self.param_dims)), var_post.reshape(tuple(self.param_dims))#/np.sum(var_post)


    def updatePIDF(self, mu, failed=1):
        """ 
        Update the penalisation PDF based on the failed trials, by centering a
        negative multivariate Gaussian in the point in the parameter space 
        corresponding to the movement vector of the failed trial.
        Similar is done for the successful trials with a positive Gaussian, but wider.
        """
        previous_pidf = self.penal_IDF.copy()
        # Modify the covariance matrix, depending on trial outcome  
        # Failed trial
        if failed == 1:
            """ VERSION 1
            Use most diverse samples. Make the parameters that change often change less (make cov smaller and wider), 
            and the ones which don't push to change more (make cov larger and narrower)
            """
            # fl_var = np.array([len(np.unique(np.array(self.failed_coords)[:,f])) for f in range(len(self.param_list)) ], np.float)
            # cov_coeff = 1 + (fl_var-fl_var.mean())/(fl_var.max())
            """ VERSION 2
            Get samples with most repeated elements. Leave the parameters that change often as they are (leave cov as is),
            and the ones which don't push to change more (make cov larger and narrower) 
            """
            fl_var = np.array([ max(np.bincount(np.array(self.failed_coords)[:,f])) for f in range(len(self.param_list)) ], np.float)            
            cov_coeff = 1 + (fl_var-fl_var.min())/fl_var.max()
        # Succesful trial
        elif failed == -1:
            fl_var = 0
            cov_coeff = 0.5*np.ones(len(self.param_list))
            # # Update the covariance matrix taking into account stationary parameters
            # if len(self.failed_coords)>0:
            #     good_coords = set(map(tuple, self.coord_explored)) - set(map(tuple,self.failed_coords))
            #     good_coords = np.array(list(good_coords)) 
            #     """VERSION 1: Get most diverse samples"""
            #     # fl_var = np.array([len(np.unique(np.array(self.failed_coords)[:,f])) for f in range(len(self.param_list)) ], np.float)
            #     # cov_coeff = (1-(fl_var-fl_var.mean())/(fl_var.max()))
            #     """VERSION 2: Get samples with most repeated elements"""
            #     # fl_var = np.array([ max(np.bincount(np.array(self.good_coords)[:,f])) for f in range(len(self.param_list)) ], np.float)            
            #     # cov_coeff = 1+(1-(fl_var)/(fl_var.max()))
            # else:
            #     cov_coeff = 0.5*np.ones(len(self.param_list))
        # Update covariance diagonal elements
        for idx, cc in enumerate(cov_coeff):
            self.cov[idx,idx] = self.COVARIANCE * cc
        # Estimate the contributing Gaussian
        trial_gaussian = failed * np.reshape(self.generatePDF_matrix(self.param_space, mu, self.cov), tuple(self.param_dims))
        trial_gaussian /= (trial_gaussian.max() + self.eps_var)
        # Update the penalisation IDF
        self.penal_IDF = np.clip(previous_pidf + trial_gaussian, self.prior_init, 1.)
        # print("---check, penalisation updates: ")
        # print(fl_var)
        # print(np.diag(self.cov))
        print("--- penalised", len(np.argwhere(self.penal_IDF.round(2)==np.max(self.penal_IDF.round(2)))),"peaks from",len(self.failed_coords),"combinations.")
        

    def generateInformedSample(self, info_list):
        """
        Generate the movement parameter vector to evaluate next, 
        based on the GPR model uncertainty and the penalisation IDF.
        """
        # Combine the model uncertainty with the penalisation IDF to get the most informative point  
        # model_var = (self.prior_init * self.model_uncertainty)/np.sum(self.prior_init * self.model_uncertainty)
        selection_IDF = 1.0 * self.model_uncertainty * (1 - self.penal_IDF)#/np.sum(1-self.penal_IDF)
        # info_pdf /= np.sum(info_pdf)
        self.selection_IDF = selection_IDF /(selection_IDF.max() + self.eps_var)
        # Check if the parameters have already been used
        temp_good = np.array([])
        cnt=1
        while len(temp_good)==0:
            temp = np.argwhere(np.array([selection_IDF==c for c in nlargest(cnt*1, selection_IDF.ravel())]).reshape(tuple(np.append(-1, self.param_dims))))[:,1:]
            temp_good = set(map(tuple, temp)) - set(map(tuple,self.coord_explored))
            temp_good = np.array(list(temp_good)) 
            cnt+=1
            if cnt > len(self.penal_IDF.ravel()):
                print("ALL COMBINATIONS HAVE BEEN EXPLORED!\nEXITING...")
                break

        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(len(self.param_list))])
        self.coord_explored.append(selected_coord)
        # print("---selection_IDF provided:", len(temp),"of which", len(temp_good),"unexplored (among the top",cnt-1,")" )
        print("--- generated coords: {}\t-> parameters: {}".format(selected_coord, selected_params))
        # return the next sample vector
        return selected_coord, selected_params


    def generateRandomSample(self):
        """
        Generate the random movement parameter vector to evaluate next.
        """
        param_sizes = [range(i) for i in self.param_dims]
        temp = np.array([xs for xs in itertools.product(*param_sizes)])
        selected_coord = temp[np.random.choice(len(temp)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(len(self.param_list))])
        self.coord_explored.append(selected_coord)
        # print("---random sampling provided:", len(temp),"of which", len(temp_good))#,"unexplored (among the top",cnt-1,")" 
        print("--- generated coords: {}\t-> parameters: {}".format(selected_coord, selected_params))
        # return the next sample vector
        return selected_coord, selected_params


    def returnModel(self):
        return self.mu_alpha, self.mu_L, self.var_alpha, self.var_L


    def returnUncertainty(self):
        return round(self.model_uncertainty.mean(), 4)


    def saveModel(self):
        self.model_list.append([self.mu_alpha, self.mu_L, self.model_uncertainty, self.penal_IDF, self.selection_IDF, self.param_list])
        with open(self.trial_dirname + "/data_training_model.dat", "wb") as m:
            pickle.dump(self.model_list, m, protocol=pickle.HIGHEST_PROTOCOL)


########### TESTING 

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
            cbar = plt.colorbar(sidf, shrink=0.5, aspect=20, pad = 0.15, orientation='horizontal', ticks=[0.0, 0.5, 1.0])
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