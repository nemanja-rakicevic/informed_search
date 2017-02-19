


import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import os
import math
import time
import cPickle as pickle

from numpy.core.umath_tests import inner1d
import itertools
import numpy as np
import scipy as sp
import scipy.spatial
import cPickle as pickle
from heapq import nlargest

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
eps_var = 0.00005
# ##################################################################
# ## max length of combination vector should be 25000 - 8/7/8/7/8
# # ### FULL MOTION SPACE
range_l_dx = np.round(np.linspace(LEFT_X_MIN, LEFT_X_MAX, 5), 3)
range_l_dy = np.round(np.linspace(LEFT_Y_MIN, LEFT_Y_MAX, 5), 3)
range_r_dx = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 5), 3)
range_r_dy = np.round(np.linspace(RIGHT_Y_MIN, RIGHT_Y_MAX, 5), 3)
range_wrist = np.round(np.linspace(WRIST_MIN, WRIST_MAX, 6), 3)
range_speed = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 5), 3)

param_list = np.array([range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_wrist, range_speed])
param_space = np.array([xs for xs in itertools.product(range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_wrist, range_speed)])
param_dims = np.array([len(i) for i in param_list])



#### SELECT KERNEL
# def kernel( a, b):
#     """ GP squared exponential kernel """
#     sigsq = 1
#     siglensq = 0.03
#     sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
#     return sigsq*np.exp(-.5 *sqdist)

# def kernel( a, b):
#     """ GP Matern 5/2 kernel: """
#     sigsq = 1
#     siglensq = 1
#     sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
#     return sigsq * (1 + np.sqrt(5*sqdist) + 5*sqdist/3.) * np.exp(-np.sqrt(5.*sqdist))

def kernel(a, b):
    """ GP rational quadratic kernel """
    sigsq = 1
    siglensq = 0.3
    alpha = a.shape[1]/2. #a.shape[1]/2. #np.exp(1) #len(a)/2.
    # print alpha
    sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
    return sigsq * np.power(1 + 0.5*sqdist/alpha, -alpha)


Kss = kernel(param_space, param_space)


class TrialInfo:
    def __init__(self, num):
        self.num = num
        self.traj_jnt = [[],[]] 
        self.traj_cart = [[],[]]

test_v = input("\nEnter which test to run:\n(1) for 2D\n(2) for FULL test\n")
if test_v==1:
    list_models = [d for d in os.listdir('./TRIALS_2D') if d[0:6]=='TRIAL_']
    for idx, t in enumerate(list_models):
        print "("+str(idx)+")\t", t
    test_num = input("\nEnter number of model to load > ")
    trialname = "TRIALS_2D/"+list_models[test_num]
    print "Loading: ",trialname
elif test_v==2:
    list_models = [d for d in os.listdir('./TRIALS_FULL') if d[0:6]=='TRIAL_']
    for idx, t in enumerate(list_models):
        print "("+str(idx)+")\t", t
    test_num = input("\nEnter number of model to load > ")
    trialname = "TRIALS_FULL/"+list_models[test_num]
    print "Loading: ",trialname

(M_angle, M_dist, var_angle, penal_PDF, param_list) = pickle.load(open(trialname+'/DATA_HCK_model_checkpoint.dat', "rb"))
(trials_list, labels_list, info_list, failed_params) = pickle.load(open(trialname+'/DATA_HCK_trial_checkpoint.dat', "rb"))

def updateGP( lab_num):
    good_trials = np.array(labels_list)[:,0]!=np.array(None)
    Xtrain = np.array(trials_list)[good_trials]
    f_evals = np.array(np.array(labels_list)[good_trials,:], dtype=float)
    y = f_evals[:, lab_num].reshape(-1,1)
    Xtest = param_space
    # calculate covariance matrices
    K = kernel(Xtrain, Xtrain)
    L = np.linalg.cholesky(K + eps_var*np.eye(len(Xtrain)))
    Ks = kernel(Xtrain, Xtest)
    Lk = np.linalg.solve(L, Ks)
    # get posterior MU and SIGMA
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    var_post = np.sqrt(np.diag(Kss) - np.sum(Lk**2, axis=0))
    # return the matrix version
    return mu.reshape(tuple(param_dims)), var_post.reshape(tuple(param_dims))#/np.sum(var_post)

# estimate the Angle GP
new_mu_alpha, new_var_alpha = updateGP(0)
# estimate the Distance GP
new_mu_L, new_var_L = updateGP(1)


# VISUALISE
dim1 = param_list[2]
dim2 = param_list[4]
X, Y = np.meshgrid(dim2, dim1)
fig = pl.figure("DISTRIBUTIONs at step: ", figsize=None)
fig.set_size_inches(fig.get_size_inches()[0]*2,fig.get_size_inches()[1]*2)
# ANGLE MODEL
ax = pl.subplot2grid((2,6),(0, 0), colspan=3, projection='3d')
ax.set_title('ANGLE MODEL')
ax.set_ylabel('right dx')
ax.set_xlabel('wrist angle')
ax.set_zlabel('[degrees]', rotation='vertical')
if param_dims[0]>1:
    ax.plot_surface(X, Y, new_mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, new_mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# DISTANCE MODEL
ax = pl.subplot2grid((2,6),(0, 3), colspan=3, projection='3d')
ax.set_title('DISTANCE MODEL')
ax.set_ylabel('right dx')
ax.set_xlabel('wrist angle')
ax.set_zlabel('[cm]', rotation='vertical')
if param_dims[0]>1:
    ax.plot_surface(X, Y, new_mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, new_mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# PENALISATION PDF
ax = pl.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
ax.set_title('Penalisation function: '+str(len(failed_params))+' points')
ax.set_ylabel('right dx')
ax.set_xlabel('wrist angle')
if param_dims[0]>1:
    ax.plot_surface(X, Y, penal_PDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, (1-penal_PDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)
# UNCERTAINTY
ax = pl.subplot2grid((2,6),(1, 2), colspan=2, projection='3d')
ax.set_ylabel('right dx')
ax.set_xlabel('wrist angle')
ax.set_title('Model uncertainty: '+str(round(new_var_alpha.mean(),4)))
if param_dims[0]>1:
    ax.plot_surface(X, Y, new_var_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, new_var_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
# SELECTION FUNCTION
# ax = pl.subplot2grid((2,6),(1, 4), colspan=2, projection='3d')
# ax.set_title('Selection function')
# ax.set_ylabel('right dx')
# ax.set_xlabel('wrist angle')
# if param_dims[0]>1:
#     ax.plot_surface(X, Y, info_pdf[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)
# else:
#     ax.plot_surface(X, Y, info_pdf[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)
# SAVEFIG
# pl.savefig(model.trial_dirname+"/IMG_HCK_distributions_step"+str(tr)+".png")
pl.show()