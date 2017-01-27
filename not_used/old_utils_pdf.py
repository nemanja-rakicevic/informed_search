
from numpy.core.umath_tests import inner1d
import itertools
import numpy as np
import matplotlib.pyplot as pl
import pickle

##################################################################3
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
##################################################################3

def sh():
    print RIGHT_X_MIN

# max length of combination vector should be 25000
range_l_dx = np.round(np.linspace(LEFT_X_MIN, LEFT_X_MAX, 100), 3)
range_l_dy = np.round(np.linspace(LEFT_Y_MIN, LEFT_Y_MAX, 100), 3)
range_r_dx = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 3), 3)
range_r_dy = np.round(np.linspace(RIGHT_Y_MIN, RIGHT_Y_MAX, 100), 3)
range_v = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 3), 3)

# 1D
dims = [len(range_r_dy)]
param_space = range_r_dy.reshape(-1,1)
param_list = np.array([range_r_dy])

# 2D
dims = [len(range_l_dx), len(range_l_dy)]
param_space = np.array([xs for xs in itertools.product(range_l_dx, range_l_dy)])
param_list = np.array([range_l_dx, range_l_dy])

# dims = [len(range_l_dx), len(range_l_dy), len(range_r_dx), len(range_r_dy), len(range_v)]
# param_space = np.array([xs for xs in itertools.product(range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v)])
# param_list = np.array([range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v])

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)
# # # Taken from intro to bayes opt
# def kernel(a, b):
#     """ GP squared exponential kernel """
#     kernelParameter = 1
#     sqdist = (1/kernelParameter) * np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#     return (1+np.sqrt(5*sqdist)+5*sqdist/3.) * np.exp(-np.sqrt(5*sqdist))


Kss = kernel(param_space, param_space)

mu = -0.2*np.ones(len(param_list))
cov = 1000*np.eye(len(param_list)) # increasing cov diagonal value makes the gaussian narrower

# Initialise to uniform distribution
prior_init = np.ones(tuple(dims))/(np.product(dims))
pdf = prior_init
# posterior = np.zeros(tuple(dims))
    

# Make a multinomial gaussian 
def generatePDF_matrix(x_sample, mu, cov):
    tmp = np.dot((x_sample - mu), cov)
    tmp_T = (x_sample - mu).T
    f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*inner1d(tmp,tmp_T.T))
    return f  


# GENERATING POSTERIOR
    # decide how to integrate with rest of the code
    # range of data - within or forwarded????
    # later PRIOR = POSTERIOR
    ########################

# class PDFoperations:

# def __init__(self):
#     self.dims = [len(range_l_dx), len(range_l_dy), len(range_r_dx), len(range_r_dy), len(range_v)]
#     self.param_space = np.array([xs for xs in itertools.product(range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v)])
#     self.param_list = np.array([range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v])

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)
# # # Taken from intro to bayes opt
def kernel_2(a, b):
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
def updatePDF(param_space, pdf, mu, cov):
    # Update with the gaussian likelihood
    likelihood = np.reshape(generatePDF_matrix(param_space, mu, cov), tuple(dims))
    # Apply Bayes rule
    posterior = (prior_init * likelihood)/np.sum(prior_init * likelihood)
    # Normalise posterior distribution and add it to the previous one
    shift = (prior_init + posterior)/np.sum(prior_init+posterior)
    # Normalise the final pdf

    final_pdf = pdf + shift
    # if (pdf!=prior_init).all():
    #     final_pdf = (pdf + shift)/np.sum(pdf + shift)
    #     final_pdf = (pdf + shift)
    # else:
    #     final_pdf = shift
    return final_pdf

pl.plot(param_space, posterior)
pl.plot(param_space, final_pdf)

cov = 1000*np.eye(len(param_list))
likelihood = np.reshape(generatePDF_matrix(param_space, mu, cov), tuple(dims))
pl.plot(param_space, likelihood)
pl.show()





pdf = prior_init

mu = -0.5*np.ones(len(param_list))
p_pdf = updatePDF(param_space, p_pdf, mu, cov)
pl.plot(param_space, p_pdf/p_pdf.sum())
pl.plot(param_space, p_pdf)
pl.show()
#####

m,v, s = updateGP(param_space, Kss, tr, yy/100.)

pl.plot(param_space, v1)
pl.plot(param_space, m1)
pl.show()

X, Y = np.meshgrid(range_l_dx, range_l_dy)


fig1 = pl.figure()
ax1 = fig1.gca(projection='3d')
surf = ax1.plot_surface(X, Y, p_pdf, cmap=cm.coolwarm,linewidth=0, antialiased=False)
pl.show()
#

# Perform Gaussian Process Regression based on performed trials
def updateGP(param_space, Kss, trials, f_evals):
    Xtrain = trials
    y = f_evals
    eps_var = 0.00005
    Xtest = param_space
    # 
    K = kernel(Xtrain, Xtrain)
    L = np.linalg.cholesky(K + eps_var*np.eye(len(trials)))
    Ks = kernel(Xtrain, Xtest)
    Lk = np.linalg.solve(L, Ks)
    # get posterior MU and SIGMA
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    var_post = np.sqrt(np.diag(Kss) - np.sum(Lk**2, axis=0))
    # return the matrix version
    return mu.reshape(tuple(dims)), var_post.reshape(tuple(dims)), np.sum(var_post)#/np.sum(var_post)


# Generate the next parameter vector to evaluate
def generateSample():
# update the penalisation PDF
p_pdf = updatePDF(param_space, pdf, mu, cov)
# estimate the Angle GP
mu_alpha, var_alpha = updateGP(param_space, Kss, trials, f_evals)
# estimate the Distance GP
mu_L, var_L = updateGP(param_space, Kss, trials, f_evals)
# multiply the above's uncertainties to get the most informative point
info_pdf = var_alpha * var_L * (1-p_pdf)/np.sum(1-p_pdf)
info_pdf /= np.sum(info_pdf)
# get poition of highest uncertainty
coord = np.argwhere(info_pdf==np.max(info_pdf))[0]
# return the next sample vector
    return np.array([param_list[i][coord[i]] for i in range(len(param_list))])



#######################################################
### HELPER CODE ###
#######################################################

########################
# GETTING SAMPLES (Probability Integral Transform)
# calculate cdf
# solve for x, by putting u instead of F(x)
# u is then the draw from the uniform distribution (which is transformed)
########################

### Sampling the penalisation PDF uniformly
# def samplePDF(pdf, lower=True):
#     u = np.random.uniform(np.min(pdf),np.max(pdf))
#     if lower:
#         m = np.argwhere(pdf<=u)   # "<="" is because those are the good/unexplored parameters
#     else:
#         m = np.argwhere(pdf>=u)
#     mu = m[np.random.choice(len(m))]
#     return mu

### LOOP version of the penalisaiton updatePDF function 
# def updatePDF(pdf, mu, cov):
#     x_range = np.append(np.linspace(-0.5,-0.2,10), np.linspace(0.2,0.5,10))
#     v_range = np.linspace(0.3,1,10)
#     prior_init = np.ones((len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)))/(np.product([len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)]))
#     posterior = np.zeros((len(x_range),len(x_range),len(x_range),len(x_range),len(v_range),len(v_range)))
#     # Update with the gaussian likelihood
#     for idx1, x1 in enumerate(x_range):
#         for idx2, x2 in enumerate(x_range):
#             for idx3, x3 in enumerate(x_range):
#                 for idx4, x4 in enumerate(x_range):
#                     for idx5, x5 in enumerate(v_range):
#                         for idx6, x6 in enumerate(v_range):
#                             posterior[idx1, idx2, idx3, idx4, idx5, idx6] = prior[idx1, idx2, idx3, idx4, idx5, idx6] * generatePDF(np.array([x1,x2,x3,x4,x5,x6]), mu, cov)
#     # Normalise posterior distribution and add it to the previous one
#     posterior = pdf + posterior/np.sum(posterior)
#     # Normalise the final pdf
#     posterior = posterior/np.sum(posterior)
#     return posterior

### Covariance matrix with off-diagonal dependencies
# def makeCovarianceMatrix(num_el, decay, coef=1):
#     m = np.eye(num_el)
#     for i in range(1,num_el):
#         val = [1-decay*i]*(num_el-i)
#         m += np.diag(coef*val,+i) + np.diag(coef*val,-i)
#     return m

### Generating gaussian pdf
# def generatePDF(x_sample, mu, cov):
#     f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*np.dot(np.dot((x_sample - mu), cov), (x_sample - mu).T))
#     return f