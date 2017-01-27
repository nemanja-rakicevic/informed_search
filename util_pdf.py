
from numpy.core.umath_tests import inner1d
import itertools
import numpy as np
import pickle
from heapq import nlargest

##################################################################
# # CONSTANTS - thresholds
# THRSH_START = 10
# THRSH_FORCE = 40
# THRSH_POS = 0.01
# THRSH_SPEED = 0.1
# # CONSTANTS - stick length
# STICK_X_MIN = 0
# STICK_X_MAX = 0.35
# STICK_Y_MIN = 0
# STICK_Y_MAX = 0.55
# CONSTANTS - speed 
SPEED_MIN = 0.4
SPEED_MAX = 1
# CONSTANTS - left arms
LEFT_X_MIN = -0.4
LEFT_X_MAX = 1.

LEFT_Y_MIN = -0.8
LEFT_Y_MAX = 0.0
# CONSTANTS - right arm
RIGHT_X_MIN = 0.
RIGHT_X_MAX = 0.2

RIGHT_Y_MIN = -0.8
RIGHT_Y_MAX = 0.
##################################################################
## max length of combination vector should be 25000 - 8/7/8/7/8
range_l_dx = np.round(np.linspace(LEFT_X_MIN, LEFT_X_MAX, 8), 3)
range_l_dy = np.round(np.linspace(LEFT_Y_MIN, LEFT_Y_MAX, 7), 3)
range_r_dx = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 8), 3)
range_r_dy = np.round(np.linspace(RIGHT_Y_MIN, RIGHT_Y_MAX, 7), 3)
range_v = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 8), 3)
##################################################################(-0.1,0, 0.2,0, s)
# range_l_dx = np.round(np.linspace(-0.2, -0.2, 1), 4)
# range_l_dy = np.round(np.linspace(0, 0, 1), 4)
# range_r_dx = np.round(np.linspace(0.1, 0.1, 1), 4)
# range_r_dy = np.round(np.linspace(0, 0, 1), 4)
# range_v = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 20), 3)
##################################################################
    
class PDFoperations:

    def __init__(self):
        

        # self.param_list = np.array([range_r_dx, range_r_dy, range_v])
        # self.param_space = np.array([xs for xs in itertools.product(range_r_dx, range_r_dy, range_v)])
        # self.param_dims = np.array([len(i) for i in self.param_list])


        self.param_list = np.array([range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v])
        self.param_space = np.array([xs for xs in itertools.product(range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v)])
        self.param_dims = np.array([len(i) for i in self.param_list])


        # Initialise to uniform distribution
        self.prior_init = np.ones(tuple(self.param_dims))/(np.product(self.param_dims))
        self.cov = 5000*np.eye(len(self.param_list))
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
        self.failed_list = []
        self.coord_list = []
        ###
        self.penal_PDF = self.prior_init
        ###
        self.mu_alpha = np.array([])
        self.mu_L = np.array([])
        self.var_alpha = np.ones(tuple(self.param_dims))
        self.var_L = np.ones(tuple(self.param_dims))
        #
        self.date = "_20170127"

    # Define the kernel 1
    def kernel(self, a, b):
        """ GP squared exponential kernel """
        kernelParameter = 1
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        return np.exp(-.5 * (1/kernelParameter) * sqdist)


    # # Define the kernel 2
    # def kernel(self, a, b):
    #     """ GP Matern 5/2 kernel: """
    #     kernelParameter = 1
    #     sqdist = (1/kernelParameter) * np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    #     return (1+np.sqrt(5*sqdist)+5*sqdist/3.) * np.exp(-np.sqrt(5*sqdist))


    # Make a multinomial gaussian 
    def generatePDF_matrix(self, x_sample, mu, cov):
        tmp = np.dot((x_sample - mu), cov)
        tmp_T = (x_sample - mu).T
        f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*inner1d(tmp,tmp_T.T))
        return f 


    # Update the penalisation PDF based on the failed trials
    def updatePDF(self, mu):
        self.failed_list.append(mu)
        pdf = self.penal_PDF
        # Update with the gaussian likelihood
        likelihood = np.reshape(self.generatePDF_matrix(self.param_space, mu, self.cov), tuple(self.param_dims))
        # Apply Bayes rule
        posterior = (self.prior_init * likelihood)/np.sum(self.prior_init * likelihood)
        # Normalise posterior distribution and add it to the previous one
        shift = (self.prior_init + posterior)/np.sum(self.prior_init+posterior)
        # Normalise the final pdf
        # if (pdf==self.prior_init).all():
        #     final_pdf = shift
        # else:
        #     final_pdf = (pdf + shift)/np.sum(pdf + shift)
        self.penal_PDF = pdf + shift
        print "\n---penalised", len(np.argwhere(self.penal_PDF.round(2)==np.max(self.penal_PDF.round(2))))," ponts and",len(self.failed_list),"combinations."
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
        return mu.reshape(tuple(self.param_dims)), var_post.reshape(tuple(self.param_dims)) #/np.sum(var_post)


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
            with open("DATA_HCK_model_checkpoint"+self.date+".dat", "wb") as m:
                    pickle.dump([self.mu_alpha, self.mu_L, self.var_alpha, self.var_L], m)

        # multiply the above's uncertainties to get the most informative point
        info_pdf = self.var_alpha * self.var_L * (1-self.penal_PDF)/np.sum(1-self.penal_PDF)
        info_pdf /= np.sum(info_pdf)

        # get positions of highest uncertainty 
        temp = np.argwhere(info_pdf==np.max(info_pdf))
        # check and take those which have not been explored
        temp_good = set(map(tuple, temp)) - set(map(tuple, self.coord_list))
        temp_good = np.array(map(list, temp_good))
        # temp_good = np.array([c for c in np.array(temp) if c not in np.array(self.coord_list)])
        
        cnt=1
        # print len(temp), len(temp_good), len(self.coord_list)
        while not len(temp_good):
            temp_good = np.array([c for c in np.argwhere(nlargest(cnt, info_pdf)) if c not in np.array(self.coord_list)])
            print "\nsamples:",temp_good,"\n"
            cnt+=1

        self.coord = temp_good[np.random.choice(len(temp_good)),:]
        self.coord_list.append(self.coord)
        print "---info_pdf provided:", len(temp),"of which", len(temp_good),"unexplored." 
        print "---generated coords:", self.coord
        # return the next sample vector
        return np.array([self.param_list[i][self.coord[i]] for i in range(len(self.param_list))])


    def returnModel(self):
        return self.mu_alpha, self.mu_L, self.var_alpha, self.var_L


    def returnUncertainty(self):
        return round(self.var_alpha.mean(), 4), round(self.var_L.mean(), 4)

