
import os
import time
from numpy.core.umath_tests import inner1d
import itertools
import numpy as np
import scipy as sp
import scipy.spatial
import cPickle as pickle
from heapq import nlargest

##################################################################
# CONSTANTS - speed 
SPEED_MIN = 0.5
SPEED_MAX = 1
# CONSTANTS - left arm
LEFT_X_MIN = -0.3   #(-0.35)
LEFT_X_MAX = 0.1    #(0.12)
LEFT_Y_MIN = -0.1   #(-0.8)
LEFT_Y_MAX = 0.1   #(0.30)
# CONSTANTS - left wrist
WRIST_MIN = -0.97   #(max = -3.) lean front
WRIST_MAX = 0.4     #(max = +3.) lean back
# CONSTANTS - right arm
RIGHT_X_MIN = 0.0   #(-0.05)
RIGHT_X_MAX = 0.17  #(0.20)
RIGHT_Y_MIN = -0.1  #(-0.5)
RIGHT_Y_MAX = 0.5   #(0.5)
# COVARIANCE
COV = 1000

##################################################################
## max length of combination vector should be 25000 - 8/7/8/7/8
# ### FULL MOTION SPACE
range_l_dx = np.round(np.linspace(LEFT_X_MIN, LEFT_X_MAX, 5), 3)
range_l_dy = np.round(np.linspace(LEFT_Y_MIN, LEFT_Y_MAX, 5), 3)
range_r_dx = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 5), 3)
range_r_dy = np.round(np.linspace(RIGHT_Y_MIN, RIGHT_Y_MAX, 5), 3)
range_wrist = np.round(np.linspace(WRIST_MIN, WRIST_MAX, 6), 3)
range_speed = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 5), 3)
################################################################(-0.3, 0.1, 0.05, 0.4, w=-0.97, speed=s) #(-0.1,0, 0.2,0, s)

# ### PARTIAL JOINT SPACE
# range_l_dx = np.round(np.linspace(-0.3, -0.3, 1), 3)
# range_l_dy = np.round(np.linspace(0.1, 0.1, 1), 3)
# range_r_dx = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 5), 3)
# range_r_dy = np.round(np.linspace(0.4, 0.4, 1), 3)
# range_wrist = np.round(np.linspace(WRIST_MIN, WRIST_MAX, 6), 3)
# range_speed = np.round(np.linspace(1, 1, 1), 3)
# ##################################################################
    
class PDFoperations:

    def __init__(self):
        
        self.param_list = np.array([range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_wrist, range_speed])
        self.param_space = np.array([xs for xs in itertools.product(range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_wrist, range_speed)])
        self.param_dims = np.array([len(i) for i in self.param_list])
        # Initialise to uniform distribution
        self.prior_init = np.ones(tuple(self.param_dims))/(np.product(self.param_dims))
        self.cov = COV * np.eye(len(self.param_list))
        self.eps_var = 0.00005
        # Calculate Kernel for the whole parameter (test) space
        # try:
        #     self.Kss = pickle.load(open("DATA_Kss1_matrix.dat", "rb"))
        # except:
        #     self.Kss = self.kernel(self.param_space, self.param_space)
        #     with open('DATA_Kss1_matrix.dat', "wb") as f:
        #         pickle.dump(self.Kss, f)
        self.Kss = self.kernel(self.param_space, self.param_space)
        ###
        self.trial_list = []
        self.f_eval_list = []
        #
        self.failed_params = []
        self.failed_coords = []
        self.coord_list = []
        ###
        self.penal_PDF = self.prior_init
        ###
        self.mu_alpha = np.array([])
        self.mu_L = np.array([])
        self.var_alpha = np.ones(tuple(self.param_dims))
        self.var_L = np.ones(tuple(self.param_dims))
        # ###
        self.trial_dirname = 'TRIALS_RETRAINED/TRIAL_'+time.strftime("%Y%m%d_%Hh%M")
        # if self.param_dims[0]>1:
        #     self.trial_dirname = 'TRIALS_FULL/TRIAL_'+time.strftime("%Y%m%d_%Hh%M")
        # else:
        #     self.trial_dirname = 'TRIALS_2D/TRIAL_'+time.strftime("%Y%m%d_%Hh%M")
        os.makedirs(self.trial_dirname)
        np.random.seed(210)

#### SELECT KERNEL
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


    # Make a multinomial gaussian 
    def generatePDF_matrix(self, x_sample, mu, cov):
        tmp = np.dot((x_sample - mu), cov)
        tmp_T = (x_sample - mu).T
        f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*inner1d(tmp,tmp_T.T))
        return f 


    # Update the penalisation PDF based on the failed trials
    def updatePDF(self, mu, sign=1):
        pdf = self.penal_PDF.copy()
        ### Modify the covariance matrix
        if sign == -1:
            fl_var = 0
            cov_coeff = 0.5*np.ones(len(self.param_list))
            # ### Update the covariance matrix taking into account stationary parameters:
            # if len(self.failed_coords)>0:
            #     good_coords = set(map(tuple, self.coord_list)) - set(map(tuple,self.failed_coords))
            #     good_coords = np.array(map(list, good_coords)) 
            #     ### Get most diverse samples: 
            #     # fl_var = np.array([len(np.unique(np.array(self.failed_coords)[:,f])) for f in range(len(self.param_list)) ], np.float)
            #     ### Get samples with most repeated elements:
            #     fl_var = np.array([ max(np.bincount(np.array(self.good_coords)[:,f])) for f in range(len(self.param_list)) ], np.float)            
            #     # ### VERSION 1
            #     # cov_coeff = (1-(fl_var-fl_var.mean())/(fl_var.max()))
            #     # ### VERSION 2
            #     # cov_coeff = 1+(1-(fl_var)/(fl_var.max()))
            # else:
            #     cov_coeff = 0.5*np.ones(len(self.param_list))
                
        elif sign == 1:
            self.failed_params.append(mu)
            self.failed_coords.append(self.coord)
            ### Get most diverse samples: 
            # fl_var = np.array([len(np.unique(np.array(self.failed_coords)[:,f])) for f in range(len(self.param_list)) ], np.float)
            ### Get samples with most repeated elements:
            fl_var = np.array([ max(np.bincount(np.array(self.failed_coords)[:,f])) for f in range(len(self.param_list)) ], np.float)            
            ### VERSION 1
            # Make the ones that change often change less (make cov smaller and wider), 
            # and the ones which don't push to change more (make cov larger and narrower)
            # cov_coeff = 1+(fl_var-fl_var.mean())/(fl_var.max())
            # ### VERSION 2
            # # Leave the ones that change often as they are (leave cov as is),
            # # and the ones which don't push to change more (make cov larger and narrower)
            cov_coeff = 1+(fl_var-fl_var.min())/fl_var.max()

        ### Update covariance diagonal elements
        for idx, cc in enumerate(cov_coeff):
            self.cov[idx,idx] = COV * cc
        print "---check, penalisation updates: "
        print fl_var
        print np.diag(self.cov)
        likelihood = np.reshape(self.generatePDF_matrix(self.param_space, mu, self.cov), tuple(self.param_dims))
        
        ### Apply Bayes rule
        posterior = sign * (self.prior_init * likelihood)/np.sum(self.prior_init * likelihood)
        ### Normalise posterior distribution and add it to the previous one
        shift = (self.prior_init + posterior)#/np.sum(self.prior_init+posterior)  

        # Normalise the final pdf
        # if (pdf==self.prior_init).all():
        #     final_pdf = shift
        # else:
        #     final_pdf = (pdf + shift)/np.sum(pdf + shift)

        ### Update with the penalisation gaussian likelihood
        self.penal_PDF = np.clip(pdf + shift, self.prior_init, 1.)
        # self.penal_PDF = pdf + shift

        print "\n---penalised", len(np.argwhere(self.penal_PDF.round(2)==np.max(self.penal_PDF.round(2))))," ponts and",len(self.failed_params),"combinations."
        return self.penal_PDF


    # Perform Gaussian Process Regression based on performed trials
    def updateGP(self, lab_num):
        Xtrain = self.trials
        y = self.f_evals[:, lab_num].reshape(-1,1)
        Xtest = self.param_space
        # calculate covariance matrices
        K = self.kernel(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + self.eps_var*np.eye(len(self.trials)))
        Ks = self.kernel(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # get posterior MU and SIGMA
        mu = np.dot(Lk.T, np.linalg.solve(L, y))
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        # return the matrix version
        return mu.reshape(tuple(self.param_dims)), var_post.reshape(tuple(self.param_dims))#/np.sum(var_post)


    # Generate the next parameter vector to evaluate
    def generateSample(self, trial_list, f_eval_list):
        # take only trials which produced correct labels
        # if f_eval_list.size:
        if not (f_eval_list==np.array(None)).all():
            good_trials = f_eval_list[:,0]!=np.array(None)
            self.trials = trial_list[good_trials]
            self.f_evals = np.array(f_eval_list[good_trials,:], dtype=float)
            # estimate the Angle GP
            self.mu_alpha, self.var_alpha = self.updateGP(0)
            # estimate the Distance GP
            self.mu_L, self.var_L = self.updateGP(1)
            # SAVE CURRENT MODEL
            # with open('DATA_trial_checkpoint.dat', "wb") as f:
            #         pickle.dump([self.trial_list,self.f_eval_list], f)
            with open(self.trial_dirname+"/DATA_HCK_model_checkpoint_"+str(len(trial_list))+".dat", "wb") as m:
                    pickle.dump([self.mu_alpha, self.mu_L, self.var_alpha, self.penal_PDF, self.param_list], m)
        # multiply the above's uncertainties to get the most informative point
        # DO NOT NORMALIZE ?!
        model_var = self.var_alpha
        # model_var = (self.prior_init * self.var_alpha)/np.sum(self.prior_init * self.var_alpha)
        info_pdf = 1.0 * model_var * (1-self.penal_PDF)#/np.sum(1-self.penal_PDF)
       
        # info_pdf /= np.sum(info_pdf)
       
        self.info_pdf = info_pdf
        temp_good = []
        cnt=1
        while not len(temp_good):
            # get positions of highest uncertainty 
            # temp = np.argwhere(info_pdf==np.max(info_pdf))
            # temp = np.argwhere((info_pdf==nlargest(cnt, info_pdf.ravel())).reshape(tuple(np.append(self.param_dims, -1))))[:,0:-1]
            temp = np.argwhere(np.array([info_pdf==c for c in nlargest(cnt*1, info_pdf.ravel())]).reshape(tuple(np.append(-1, self.param_dims))))[:,1:]
            # check and take those which have not been explored
            temp_good = set(map(tuple, temp)) - set(map(tuple,self.coord_list))
            temp_good = np.array(map(list, temp_good))            
            cnt+=1
            print "cnt: ", cnt-1
            print "tmp: ", len(temp_good)
            # # FILTER 2D
            # temp_good = np.array([c for c in temp_good if c[0]==0 and c[1]==0 and c[3]==0 and c[5]==0])
            # if cnt-1 > 100:
            #     print 'ALL options exhausted...Quitting'
            #     break
            # print "\nsamples:"
            # print len(temp)
            # print cnt
            # print len(self.coord_list)
            # print 'selected:'
            # print len(temp_good1),'---',len(temp_good),"\n"

        self.coord = temp_good[np.random.choice(len(temp_good)),:]
        self.coord_list.append(self.coord)
        print "---info_pdf provided:", len(temp),"of which", len(temp_good),"unexplored (among the top",cnt-1,")" 
        print "---generated coords:", self.coord
        # return the next sample vector
        return np.array([self.param_list[i][self.coord[i]] for i in range(len(self.param_list))])

####################################################################################################
    # ###### RANDOM MODEL
    #  # Generate the next parameter vector to evaluate
    # def generateSample(self, trial_list, f_eval_list):
    #     # take only trials which produced correct labels
    #     # if f_eval_list.size:
    #     if not (f_eval_list==np.array(None)).all():
    #         good_trials = f_eval_list[:,0]!=np.array(None)
    #         self.trials = trial_list[good_trials]
    #         self.f_evals = np.array(f_eval_list[good_trials,:], dtype=float)
    #         # estimate the Angle GP
    #         self.mu_alpha, self.var_alpha = self.updateGP(0)
    #         # estimate the Distance GP
    #         self.mu_L, self.var_L = self.updateGP(1)
    #         # SAVE CURRENT MODEL
    #         # with open('DATA_trial_checkpoint.dat', "wb") as f:
    #         #         pickle.dump([self.trial_list,self.f_eval_list], f)
    #         with open(self.trial_dirname+"/DATA_HCK_model_checkpoint.dat", "wb") as m:
    #                 pickle.dump([self.mu_alpha, self.mu_L, self.var_alpha, self.penal_PDF, self.param_list], m)

    #     temp_good = []
    #     # get random points 
    #     temp = np.array([xs for xs in itertools.product(range(self.param_dims[0]), range(self.param_dims[1]), range(self.param_dims[2]), range(self.param_dims[3]), range(self.param_dims[4]), range(self.param_dims[5]))])
    #     # # check and take those which have not been explored
    #     # temp_good = set(map(tuple, temp)) - set(map(tuple,self.coord_list))
    #     # temp_good = np.array(map(list, temp_good))  
    #     # Take whichever
    #     temp_good = temp          
    #     # cnt+=1
    #     # print "cnt: ", cnt-1
    #     print "tmp: ", len(temp_good)

    #     self.coord = temp_good[np.random.choice(len(temp_good)),:]
    #     self.coord_list.append(self.coord)
    #     print "---info_pdf provided:", len(temp),"of which", len(temp_good)#,"unexplored (among the top",cnt-1,")" 
    #     print "---generated coords:", self.coord
    #     # return the next sample vector
    #     return np.array([self.param_list[i][self.coord[i]] for i in range(len(self.param_list))])
 ####################################################################################################      


    def returnModel(self):
        return self.mu_alpha, self.mu_L, self.var_alpha, self.var_L


    def returnUncertainty(self):
        return round(self.var_alpha.mean(), 4), round(self.var_L.mean(), 4)

