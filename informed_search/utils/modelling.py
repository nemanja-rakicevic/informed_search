
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Classes containing different model implementations
"""

import os
import time
import logging
from numpy.core.umath_tests import inner1d

import pickle
import logging
import itertools
import numpy as np
import scipy as sp
import scipy.spatial
from heapq import nlargest, nsmallest


import utils.plotting as uplot
from utils.misc import _EPS


logger = logging.getLogger(__name__)



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

        # Initialise informed search components
        self.pidf = self.prior_init
        self.uidf = self.prior_init
        self.sidf = self.prior_init  # np.ones(tuple(self.param_dims))

        # Initialise model components
        self.model_list = []
        self.mu_alpha = np.zeros(self.param_dims)
        self.mu_L = np.zeros(self.param_dims)
        self.var_alpha = np.ones(self.param_dims)
        self.var_L = np.ones(self.param_dims)
        self.uidf = np.ones(self.param_dims)
        #
        self.mu_pidf = 0.5*np.ones(self.param_dims)
        self.var_pidf = np.ones(self.param_dims)
        self.mu_uncert = 0*np.ones(self.param_dims)
        self.var_uncert = np.ones(self.param_dims)
        self.previous_uncertainty = self.uncertainty
        self.delta_uncertainty = []


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
        

    def query_target(self, angle_s, dist_s, verbose=False):
        """ 
            Generate test parameter coordinates for given target polar coords.
        """
        thrsh = 0.1

        def elementwise_sqdist(x,y):
            return np.sqrt(x**2 + y**2)

        def scaled_sqdist(M_angle, M_dist, angle_s, dist_s):
            diff_angle = M_angle - angle_s
            diff_angle = (diff_angle - diff_angle.min()) \
                            /(diff_angle.max() - diff_angle.min())
            diff_dist  = M_dist - dist_s
            diff_dist  = (diff_dist - diff_dist.min()) \
                            /(diff_dist.max() - diff_dist.min())
            return sqdist(diff_angle, diff_dist)

        # Calculate model error for target point
        M_meas = elementwise_sqdist(self.mu_alpha-angle_s, self.mu_L-dist_s)
        # M_meas  = scaled_sqdist(M_angle, M_dist, angle_s, dist_s)

        # Generate optimal parameters to reached the target point
        src = 0
        for cnt in range(len(M_meas.ravel())):
            src += 1
            # get candidate movement parameter vector with smalles model error
            best_fit        = nsmallest(src, M_meas.ravel())
            selected_coord  = np.argwhere(M_meas==best_fit[src-1])[0]
            selected_params = np.array([self.param_list[i][selected_coord[i]] \
                                            for i in range(self.n_param)])
            selected_mu_alpha = self.mu_alpha[tuple(selected_coord)]
            selected_mu_L = self.mu_L[tuple(selected_coord)]
            error_angle     = selected_mu_alpha - angle_s
            error_dist      = selected_mu_L - dist_s
            if cnt%50 == 0 and cnt > 100:
                thrsh = cnt/1000.
                src = 0
            # check pidf constraint is below threshold (if close to failure)
            pidf_value = self.pidf[tuple(selected_coord)]
            if pidf_value < thrsh:
                break
            else:    
                # continue to the next smallest model error
                if verbose:
                    print("\n(Iteration: {}; search {}) "
                          "--- BAD SAMPLE, resampling ..."
                          "\n - PIDF: {:4.2}; UIDF: {:4.2}; SIDF: {:4.2};"
                          "\n".format(cnt, src, 
                                      pidf_value, uidf_value, sidf_value))
        if verbose:
            uidf_value = self.uidf[tuple(selected_coord)]
            sidf_value = self.sidf[tuple(selected_coord)]
            print("\n(Iteration: {})"
                  "\n - PIDF: {:4.2}; UIDF: {:4.2}; SIDF: {:4.2};"
                  "\n - Generated coords: {} -> parameters: {} "
                  "\n - Model polar error: {:4.2f}"
                  "\n --- |selected ({:4.2f}, {:4.2f}) - target ({}, {})| = "
                  "error ({:4.2f}, {:4.2f})\n".format(
                                        cnt, pidf_value, uidf_value, sidf_value,
                                        selected_coord, selected_params, 
                                        best_fit[src-1],
                                        selected_mu_alpha, selected_mu_L,
                                        angle_s, dist_s,
                                        error_angle, error_dist))
        # Return vector to execute as well as estimated model polar error
        return selected_coord, selected_params, best_fit[src-1]


    def update_model():
        raise NotImplementedError


    def generate_sample():
        raise NotImplementedError

    @property
    def uncertainty(self):
        return self.uidf.mean()


    @property
    def components(self):
        return self.mu_alpha, self.mu_L, self.var_alpha, self.var_L


    def save_model(self, num_trial, save_plots=True, save_data=True, **kwargs):
        if save_plots:
            uplot.plot_model(model_object=self,
                             dimensions=[0,1],
                             num_trial=num_trial, 
                             savepath=self.dirname)  
                             # param_names=['joint_1', 'joint_0'])
        if save_data:
            self.model_list.append([self.mu_alpha, self.mu_L, 
                                    self.uidf, self.pidf, self.sidf, 
                                    self.param_list])
            with open(self.dirname+"/model_checkpoints.pickle", "wb") as f:
                pickle.dump(self.model_list, f, 
                            protocol=pickle.HIGHEST_PROTOCOL)


    def load_model(self, loadpath):
        # list_models = sorted([d for d in \
        #     os.listdir('./DATA/'+self.exp_type+'/') if d[0:6]=='TRIAL_'])
        # for idx, t in enumerate(list_models):
        #     print("("+str(idx)+")\t", t)
        # test_num = input("\nEnter number of model to load > ")
        # self.dirname = './DATA/'+self.exp_type+'/'+list_models[int(test_num)]
        logger.info("Loading: ", loadpath)
        with open(loadpath + "/model_checkpoints.pickle", "rb") as f:
            self.model_list = pickle.load(f)
            self.mu_alpha, self.mu_L, \
            self.uidf, self.pidf, self.sidf, \
            self.param_list = self.model_list[-1]



##############################################################################
##############################################################################
##############################################################################
##############################################################################




class InformedSearch(BaseModel):
    """
        Proposed Informed Search model class; 
        Uses the penalisation and uncertainty IDF's to determine the next trial.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
        """
            Select successful trials to estimate the GPR model's mean and 
            variance, and the failed ones to update the penalisation IDF.
        """
        if len(info_list):

            # Successful trial: Update task models and PIDF
            if info_list[-1]['fail_status']==0:
                good_trials = np.vstack([[tr['parameters'], tr['ball_polar']] \
                                        for tr in info_list if tr['fail_status']==0])
                good_params = np.vstack(good_trials[:, 0])
                good_fevals = np.vstack(good_trials[:, 1])


                # pdb.set_trace()

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
            self.uidf = self.update_GPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.uidf = self.update_GPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_model_progress: self.save_model()


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
        # logger.info("PIDF: Penalised {} peaks from {} combinations.".format(
        #     len(np.argwhere(self.pidf.round(2)==np.max(self.pidf.round(2)))),
        #     len(self.coord_failed)))
        


    def generate_sample(self, info_list=None, **kwargs):
        """
            Generate the movement parameter vector to evaluate next, 
            based on the GPR model uncertainty and the penalisation IDF.
        """
        # Combine the model uncertainty with the PIDF 
        # model_var = (self.prior_init * self.uidf)/np.sum(self.prior_init * self.uidf)
        sidf = 1.0 * self.uidf * (1 - self.pidf)#/np.sum(1-self.pidf)
        
        # Scale the selection IDF
        # info_pdf /= np.sum(info_pdf)
        self.sidf = sidf / (sidf.max() + _EPS)

        # Check if the parameters have already been used
        temp_good = np.array([])
        cnt = 1
        while len(temp_good)==0:
            sample = np.array([sidf==c for c in nlargest(cnt*1, sidf.ravel())])
            sample = sample.reshape([-1]+list(self.param_dims))
            sample_idx = np.argwhere(sample)[:,1:]
            temp_good = np.array(list(set(map(tuple, sample_idx)) \
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
    """
        Uncertainty-only model;
        Uses only the uncertainty IDF to select the next trial. Does not update
        the penalisation IDF.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
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
            self.uidf = self.update_GPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.uidf = self.update_GPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_model_progress: self.save_model()


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
        

    def generate_sample(self, info_list=None, **kwargs):
        """
        Generate the movement parameter vector to evaluate next, 
        based ONLY on the GPR model uncertainty.
        """
        # Combine the model uncertainty with the penalisation IDF to get the most informative point  
        # model_var = (self.prior_init * self.uidf)/np.sum(self.prior_init * self.uidf)
        sidf = 1.0 * self.uidf #* (1 - self.pidf)#/np.sum(1-self.pidf)
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
                logger.info("ALL COMBINATIONS HAVE BEEN EXPLORED! EXITING...")
                break

        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # return the next sample vector
        return selected_coord, selected_params






##############################################################################
##############################################################################
##############################################################################
##############################################################################





class EntropySearch(BaseModel):
    """
        Entropy-based model;
        Uses the entropy of uncertainty IDF to select the next trial. 
        Does not update the penalisation IDF.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
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
            self.uidf = self.update_GPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.uidf = self.update_GPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_model_progress: self.save_model()


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
        

    def generate_sample(self, info_list=None, **kwargs):
        """
        Generate the movement parameter vector to evaluate next, 
        based ONLY on the posterior distributions entropy
        """
        # Combine the model uncertainty with the penalisation IDF to get the most informative point  
        # model_var = (self.prior_init * self.uidf)/np.sum(self.prior_init * self.uidf)
        sidf = 0.5 * np.log(2 * np.pi * np.e * self.uidf )#* (1 - self.pidf)#/np.sum(1-self.pidf)
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
                logger.info("ALL COMBINATIONS HAVE BEEN EXPLORED! EXITING...")
                break

        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # return the next sample vector
        return selected_coord, selected_params








##############################################################################
##############################################################################
##############################################################################
##############################################################################



# REVIEWER

class BOSearch(BaseModel):
    """
        A modified BO approach from Englert and Toussaint (2016), adapted to 
        next trial selection.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
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
            self.uidf = self.update_GPR(all_trials, None, -1)
            # GPC for fail/success 
            all_nonfails = np.array([ 1 if tr['fail_status']>0 else 0 for tr in info_list])
            self.mu_pidf,     self.var_pidf     = self.update_GPR_reviewer(all_trials, all_nonfails)

            temp = self.uncertainty
            self.delta_uncertainty.append(self.previous_uncertainty - temp)
            self.previous_uncertainty = temp
            assert len(self.delta_uncertainty)==len(info_list)
            self.mu_uncert,     self.var_uncert     = self.update_GPR_reviewer(all_trials, np.array(self.delta_uncertainty))


            # SAVE CURRENT MODEL
            if save_model_progress: self.save_model()


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
        

    def generate_sample(self, info_list=None, **kwargs):
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
                logger.info("ALL COMBINATIONS HAVE BEEN EXPLORED! EXITING...")
                break

        selected_coord = temp_good[np.random.choice(len(temp_good)),:]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # return the next sample vector
        return selected_coord, selected_params







##############################################################################
##############################################################################
##############################################################################
##############################################################################





class RandomSearch(BaseModel):
    """
        Purely random search in the parameter space, without replacement.
    """

    def update_model(self, info_list, save_model_progress=False, **kwargs):
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
            self.uidf = self.update_GPR(all_trials, None, -1)
            # There's some crazy bug here when using 5-link version, canot figure it out...
            # all_trials = np.array([[tr['parameters'], tr['ball_polar']] for tr in info_list])
            # self.uidf = self.update_GPR(all_trials[:,0], all_trials[:,1], -1)
            # SAVE CURRENT MODEL
            if save_model_progress: self.save_model()


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
                logger.info("ALL COMBINATIONS HAVE BEEN EXPLORED! EXITING...")
                break

        selected_coord = temp_good[0]
        selected_params = np.array([self.param_list[i][selected_coord[i]] for i in range(self.n_param)])
        self.coord_explored.append(selected_coord)
        # return the next sample vector
        return selected_coord, selected_params

