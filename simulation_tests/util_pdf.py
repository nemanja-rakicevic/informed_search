
import os
import time
from numpy.core.umath_tests import inner1d
import itertools
import numpy as np
import scipy as sp
import scipy.spatial
import pickle
from heapq import nlargest

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# COVARIANCE
COV = 1000

class InformedModel:
    def __init__(self, parameter_list):
        self.param_list = parameter_list
        self.param_space = np.array([xs for xs in itertools.product(*self.param_list)])
        self.param_dims = np.array([len(i) for i in self.param_list])
        # Initialise to uniform distribution
        self.prior_init = np.ones(tuple(self.param_dims))/(np.product(self.param_dims))
        self.cov = COV * np.eye(len(self.param_list))
        self.eps_var = 0.00005
        # Calculate Kernel for the whole parameter (test) spacece, self.param_space)
        self.Kss = self.kernel(self.param_space, self.param_space)
        #
        self.coord_explored = []
        self.failed_coords = []
        self.failed_params = []
        self.good_params = []
        self.good_fevals = []
        ###
        self.penal_IDF = self.prior_init
        ###
        self.mu_alpha = np.array([])
        self.mu_L = np.array([])
        self.var_alpha = np.ones(tuple(self.param_dims))
        self.var_L = np.ones(tuple(self.param_dims))
        # ###
        self.trial_dirname = 'DATA/SIMULATION/TRIAL_'+time.strftime("%Y%m%d_%Hh%M")
        # if self.param_dims[0]>1:
        #     self.trial_dirname = 'TRIALS_FULL/TRIAL_'+time.strftime("%Y%m%d_%Hh%M")
        # else:
        #     self.trial_dirname = 'TRIALS_2D/TRIAL_'+time.strftime("%Y%m%d_%Hh%M")
        os.makedirs(self.trial_dirname)
        np.random.seed(210)

#### SELECT KERNEL ####

    # def kernel(self, a, b):
    #     """ GP squared exponential kernel """
    #     sigsq = 1
    #     siglensq = 0.03
    #     sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
    #     return sigsq*np.exp(-.5 *sqdist)

    # def kernel(self, a, b):
    #     """ GP Matern 5/2 kernel: """
    #     sigsq = 1
    #     siglensq = 1
    #     sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
    #     return sigsq * (1 + np.sqrt(5*sqdist) + 5*sqdist/3.) * np.exp(-np.sqrt(5.*sqdist))

    def kernel(self, a, b):
        """ GP rational quadratic kernel """
        sigsq = 1
        siglensq = 1
        alpha = a.shape[1]/2. #a.shape[1]/2. #np.exp(1) #len(a)/2.
        # print alpha
        sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
        return sigsq * np.power(1 + 0.5*sqdist/alpha, -alpha)

######################## 


    def generatePDF_matrix(self, x_sample, mu, cov):
        """ Make a multinomial gaussian over the parameter space """
        tmp = np.dot((x_sample - mu), cov)
        tmp_T = (x_sample - mu).T
        f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*inner1d(tmp,tmp_T.T))
        return f 


    def updateModel(self, info_list):
        """
        Select successful trials to estimate the GPR model mean and variance,
        and the failed ones to update the penalisation IDF.
        """
        if len(info_list):
            # Successful trial
            if info_list[-1]['fail']==0:
                good_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list if tr['fail']==0])
                self.good_params = good_trials[:,0]
                self.good_fevals = good_trials[:,1]
                # Estimate the Angle and Distance GPR models, as well as PIDF
                self.mu_alpha, self.var_alpha = self.updateGP(0)
                self.mu_L,     self.var_L     = self.updateGP(1)
                self.updatePIDF(info_list[-1]['parameters'], failed=-1)
            # Failed trial
            elif info_list[-1]['fail']>0:
                self.failed_coords = np.array([tr['coordinates'] for tr in info_list if tr['fail']>0])
                self.updatePIDF(info_list[-1]['parameters'], failed=1)
            # SAVE CURRENT MODEL
            self.saveModel()


    def updateGP(self, label_indicator):
        """
        Perform Gaussian Process Regression using good performed trials as training points,
        and evaluate on the whole parameter space to get the new model and uncertainty estimates.
        """
        Xtrain = self.good_params
        Ytrain = self.good_fevals[:, label_indicator].reshape(-1,1)
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
        pdf = self.penal_IDF.copy()
        # Modify the covariance matrix, depending on trial outcome  
        # Failed trial
        if failed == 1:
            self.failed_coords.append(self.coord)
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
            #     good_coords = np.array(map(list, good_coords)) 
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
            self.cov[idx,idx] = COV * cc
        # Apply Bayes rule
        likelihood = np.reshape(self.generatePDF_matrix(self.param_space, mu, self.cov), tuple(self.param_dims))
        posterior = failed * (self.prior_init * likelihood)/np.sum(self.prior_init * likelihood)
        # Normalise posterior distribution and add it to the previous one
        shift = (self.prior_init + posterior)#/np.sum(self.prior_init+posterior)  
        # Update with the penalisation IDF
        self.penal_IDF = np.clip(pdf + shift, self.prior_init, 1.)

        print("---check, penalisation updates: ")
        print(fl_var)
        print(np.diag(self.cov))
        print("\n---penalised", len(np.argwhere(self.penal_IDF.round(2)==np.max(self.penal_IDF.round(2))))," ponts and",len(self.failed_coords),"combinations.")
        

    def generateInformedSample(self, info_list):
        """
        Generate the movement parameter vector to evaluate next, 
        based on the GPR model uncertainty and the penalisation IDF.
        """
        # Combine the model uncertainty with the penalisation IDF to get the most informative point   
        # DO NOT NORMALIZE ! LOG PROBABILITY ?
        # model_var = (self.prior_init * self.var_alpha)/np.sum(self.prior_init * self.var_alpha)
        selection_IDF = 1.0 * self.var_alpha * (1 - self.penal_IDF)#/np.sum(1-self.penal_IDF)
        # info_pdf /= np.sum(info_pdf)
        self.selection_IDF = selection_IDF
        # Check if the parameters have already been used
        temp_good = []
        cnt=1
        while not len(temp_good):
            temp = np.argwhere(np.array([selection_IDF==c for c in nlargest(cnt*1, selection_IDF.ravel())]).reshape(tuple(np.append(-1, self.param_dims))))[:,1:]
            temp_good = set(map(tuple, temp)) - set(map(tuple,self.coord_explored))
            temp_good = np.array(map(list, temp_good))            
            cnt+=1

        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        self.coord_explored.append(selected_coord)
        print("---selection_IDF provided:", len(temp),"of which", len(temp_good),"unexplored (among the top",cnt-1,")" )
        print("---generated coords:", selected_coord)
        # return the next sample vector
        return selected_coord, np.array([self.param_list[i][selected_coord[i]] for i in range(len(self.param_list))])


    def generateRandomSample(self):
        """
        Generate the random movement parameter vector to evaluate next.
        """
        param_sizes = [range(i) for i in self.param_dims]
        temp = np.array([xs for xs in itertools.product(*param_sizes)])
        selected_coord = temp[np.random.choice(len(temp)),:]
        self.coord_explored.append(selected_coord)
        print("---random sampling provided:", len(temp),"of which", len(temp_good))#,"unexplored (among the top",cnt-1,")" 
        print("---generated coords:", self.coord)
        # return the next sample vector
        return selected_coord, np.array([self.param_list[i][selected_coord[i]] for i in range(len(self.param_list))])

 

    def returnModel(self):
        return self.mu_alpha, self.mu_L, self.var_alpha, self.var_L


    def returnUncertainty(self):
        return round(self.var_alpha.mean(), 4)


    def saveModel(self):
        with open(self.trial_dirname + "/DATA_HCK_model_checkpoint.dat", "wb") as m:
            pickle.dump([self.mu_alpha, self.mu_L, self.var_alpha, self.penal_IDF, self.selection_IDF, self.param_list], m, protocol=pickle.HIGHEST_PROTOCOL)



    def plotModel(self, trial_num, dimensions, param_names):
        # if trial_num%1==0 or trial_info.fail_status==0:
        # print "<- CHECK PLOTS"     
        if len(self.mu_alpha) and len(self.mu_L):
            fig = plt.figure("DISTRIBUTIONs at step: "+str(trial_num), figsize=None)
            fig.set_size_inches(fig.get_size_inches()[0]*2,fig.get_size_inches()[1]*2)
            dim1 = self.param_list[dimensions[0]]
            dim2 = self.param_list[dimensions[1]]
            X, Y = np.meshgrid(dim2, dim1)
            # Values to plot
            if len(dimensions)>2:
                if self.param_dims[0]>1:
                    model_alpha  = self.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    model_L      = self.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    model_PIDF   = self.penal_IDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    model_var    = self.var_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                    model_select = self.selection_IDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2))
                else:
                    model_alpha  = self.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_L      = self.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_PIDF   = self.penal_IDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_var    = self.var_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
                    model_select = self.selection_IDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))
            else:
                model_alpha  = self.mu_alpha
                model_L      = self.mu_L
                model_PIDF   = self.penal_IDF
                model_var    = self.var_alpha
                model_select = self.selection_IDF
            # ANGLE MODEL
            ax = plt.subplot2grid((2,6),(0, 0), colspan=3, projection='3d')
            ax.set_title('ANGLE MODEL')
            ax.set_ylabel(param_names[1])
            ax.set_xlabel(param_names[0])
            ax.set_zlabel('[degrees]', rotation='vertical')
            ax.plot_surface(X, Y, model_alpha, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # DISTANCE MODEL
            ax = plt.subplot2grid((2,6),(0, 3), colspan=3, projection='3d')
            ax.set_title('DISTANCE MODEL')
            ax.set_ylabel(param_names[1])
            ax.set_xlabel(param_names[0])
            ax.set_zlabel('[cm]', rotation='vertical')
            ax.plot_surface(X, Y, model_L, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # PENALISATION PDF
            ax = plt.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
            ax.set_title('Penalisation function: '+str(len(self.failed_params))+' points')
            ax.set_ylabel(param_names[1])
            ax.set_xlabel(param_names[0])
            ax.plot_surface(X, Y, model_PIDF, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # UNCERTAINTY
            ax = plt.subplot2grid((2,6),(1, 2), colspan=2, projection='3d')
            ax.set_title('Model uncertainty: '+str(self.returnUncertainty()))
            ax.set_ylabel(param_names[1])
            ax.set_xlabel(param_names[0])
            ax.plot_surface(X, Y, model_var, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # SELECTION FUNCTION
            ax = plt.subplot2grid((2,6),(1, 4), colspan=2, projection='3d')
            ax.set_title('Selection function')
            ax.set_ylabel(param_names[1])
            ax.set_xlabel(param_names[0])
            ax.plot_surface(X, Y, model_select, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # SAVEFIG
            plt.savefig(self.trial_dirname+"/IMG_HCK_distributions_trial#"+str(trial_num)+".png")
            plt.show()