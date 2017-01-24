
from numpy.core.umath_tests import inner1d
import itertools
import numpy as np
import matplotlib.pyplot as pl
import pickle

##################################################################
# CONSTANTS - thresholds
THRSH_START = 10
THRSH_FORCE = 40
THRSH_POS = 0.01
THRSH_SPEED = 0.1
# CONSTANTS - stick length
STICK_X_MIN = 0
STICK_X_MAX = 0.35
STICK_Y_MIN = 0
STICK_Y_MAX = 0.55
# CONSTANTS - speed 
SPEED_MIN = 0.3
SPEED_MAX = 1
# CONSTANTS - left arms
LEFT_X_MIN = -0.3
LEFT_X_MAX = 0.1
LEFT_Y_MIN = -0.8
LEFT_Y_MAX = 0.05
# CONSTANTS - right arm
RIGHT_X_MIN = -0.05
RIGHT_X_MAX = 0.15
RIGHT_Y_MIN = -0.8
RIGHT_Y_MAX = 0.1
##################################################################

class PDFoperations:

    # max length of combination vector should be 25000
    range_l_dx = np.round(np.linspace(LEFT_X_MIN, LEFT_X_MAX, 3), 3)
    range_l_dy = np.round(np.linspace(LEFT_Y_MIN, LEFT_Y_MAX, 3), 3)
    range_r_dx = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 3), 3)
    range_r_dy = np.round(np.linspace(RIGHT_Y_MIN, RIGHT_Y_MAX, 3), 3)
    range_v = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 3), 3)

    def __init__(self):
        self.param_list = np.array([range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v])
        self.param_space = np.array([xs for xs in itertools.product(range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v)])
        self.param_dims = np.array([len(i) for i in self.param_list])
        # Initialise to uniform distribution
        self.prior_init = np.ones(tuple(self.param_dims))/(np.product(self.param_dims))
        self.cov = 10000*np.eye(len(self.param_list))
        self.eps_var = 0.00005
        # Calculate Kernel for the whole parameter (test) space
        self.Kss = self.kernel(self.param_space, self.param_space)
        ###
        self.trial_list = []
        self.f_eval_list = []
        ###
        self.penal_PDF = self.prior_init
        ###
        self.mu_alpha = np.array([])
        self.mu_L = np.array([])
        self.var_alpha = np.ones(tuple(self.param_dims))
        self.var_L = np.ones(tuple(self.param_dims))

    # Define the kernel
    def kernel(self, a, b):
        """ GP squared exponential kernel """
        kernelParameter = 1
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        return np.exp(-.5 * (1/kernelParameter) * sqdist)
    # # # Taken from intro to bayes opt
    def kernel_2(self, a, b):
        """ GP Matern 5/2 kernel: """
        kernelParameter = 1
        sqdist = (1/kernelParameter) * np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        return (1+np.sqrt(5*sqdist)+5*sqdist/3.) * np.exp(-np.sqrt(5*sqdist))


    # Make a multinomial gaussian 
    def generatePDF_matrix(x_sample, mu, cov):
        tmp = np.dot((x_sample - mu), cov)
        tmp_T = (x_sample - mu).T
        f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*inner1d(tmp,tmp_T.T))
        return f 


    # Update the penalisation PDF based on the failed trials
    def updatePDF(self, pdf, mu):
        # Update with the gaussian likelihood
        likelihood = np.reshape(self.generatePDF_matrix(self.param_space, mu, self.cov), tuple(self.param_dims))
        # Apply Bayes rule
        posterior = (self.prior_init * likelihood)/np.sum(self.prior_init * likelihood)
        # Normalise posterior distribution and add it to the previous one
        shift = (self.prior_init + posterior)/np.sum(self.prior_init+posterior)
        # Normalise the final pdf
        if (pdf==self.prior_init).all():
            final_pdf = (pdf + shift)/np.sum(pdf + shift)
        else:
            final_pdf = shift
        return final_pdf


    # Perform Gaussian Process Regression based on performed trials
    def updateGP(self, lab_num):
        Xtrain = self.trials
        y = self.f_evals[:, lab_num]
        Xtest = self.param_space
        # 
        K = kernel(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + self.eps_var*np.eye(len(trials)))
        Ks = kernel(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # get posterior MU and SIGMA
        mu = np.dot(Lk.T, np.linalg.solve(L, y))
        var_post = np.sqrt(np.diag(self.Kss) - np.sum(Lk**2, axis=0))
        # return the matrix version
        return mu.reshape(tuple(self.param_dims)), var_post.reshape(tuple(self.param_dims))/np.sum(var_post)


    # Generate the next parameter vector to evaluate
    def generateSample(self, trial, f_eval):
        # Add and save progress so far:
        self.trial_list.append(trial)
        self.f_eval_list.append(f_eval)
        with open('hockey_checkpoint.dat', "wb") as f:
                pickle.dump([self.trial_list,self.f_eval_list], f)

        # update the penalisation PDF if the f_eval is  failure (None)
        if f_eval.any():
            self.penal_PDF = self.updatePDF(self.penal_PDF, trials)
        else:
            # estimate the Angle GP
            self.mu_alpha, self.var_alpha = self.updateGP(0)
            # estimate the Distance GP
            self.mu_L, self.var_L = self.updateGP(1)

        # multiply the above's uncertainties to get the most informative point
        info_pdf = self.var_alpha * self.var_L * (1-self.penal_PDF)/np.sum(1-self.penal_PDF)
        info_pdf /= np.sum(info_pdf)
        # get poition of highest uncertainty
        coord = np.argwhere(info_pdf==np.max(info_pdf))[0]
        # return the next sample vector
        return np.array([self.param_list[i][coord[i]] for i in range(len(self.param_list))])
        