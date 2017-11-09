

########################################################
# - CHECK DIMENSIONALITY
# - EDIT util_pdf.py
# - EDIT THE PRINTING PART, FONTSIZE
########################################################

import os
import math
import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import util_pdf as updf

########################################################
# LOAD
########################################################

# Selection function evolution
trial_name = '/media/robin/DATA/HOCKEY_PuckPassing/HOCKEY_DATA/TRIALS_2D/TRIAL_20170221_12h23_nice_plots'

# Actual model
trial_name = '/media/robin/DATA/HOCKEY_PuckPassing/HOCKEY_DATA/TRIALS_FULL/TRIAL_20170218_17h09_RQ_plusrand100'

class TrialInfo:
    def __init__(self, num):
        self.num = num
        self.traj_jnt = [[],[]] 
        self.traj_cart = [[],[]]

# LOAD Model info
with open(trial_name+'/DATA_HCK_model_checkpoint.dat', 'rb') as f:
    (M_angle, M_dist, var_angle, penal_PDF, param_list) = pickle.load(f) 


# LOAD Trial info
with open(trial_name+'/DATA_HCK_trial_checkpoint.dat', 'rb') as f:
    (trials_list, labels_list, info_list, failed_params) = pickle.load(f) 
    

########################################################
# REPLICATE tests
########################################################

# desired_trial = 8
# trials_list = trials_list[:desired_trial]
# labels_list = labels_list[:desired_trial]

model = updf.PDFoperations()
model.generateSample(np.array([]), np.array([]))
for i, tr in enumerate(trials_list):
    # model.coord = list(tr)
    if labels_list[i][0] == None:
        model.updatePDF(tr)
    else:
        model.updatePDF(tr, -1)

    model.generateSample(np.array(trials_list[:i]), np.array(labels_list[:i]))


avar, lvar = model.returnUncertainty()
tr = len(trials_list)



########################################################
# PLOT stuff
########################################################

# matplotlib.rcParams.update({'font.size': 12})

dim1 = model.param_list[2]
dim2 = model.param_list[4]
X, Y = np.meshgrid(dim2, dim1)
fig = pl.figure("DISTRIBUTIONs at step: "+str(tr), figsize=None)
fig.set_size_inches(fig.get_size_inches()[0]*2,fig.get_size_inches()[1]*2)

xticks = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 6).round(2)
xticks1 = np.linspace(min(dim2[0], dim2[-1]), max(dim2[0], dim2[-1]), 5).round(2)

yticks = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 6).round(2)
yticks1 = np.linspace(min(dim1[0], dim1[-1]), max(dim1[0], dim1[-1]), 5).round(2)

zticks_alpha = np.linspace(0, model.mu_alpha.max(), 5).round(1)
zticks_L = np.linspace(150, 350, 5).round(1)  
# zticks_alpha = np.linspace(0, model.mu_alpha.max(), 5).round(1)
# zticks_L = np.linspace(model.mu_L.min(), model.mu_L.max(), 5).round(1)     

# ANGLE MODEL
ax = pl.subplot2grid((2,6),(0, 0), colspan=3, projection='3d')
ax.set_title('ANGLE MODEL')
ax.set_ylabel('right dx', labelpad=5)
ax.set_xlabel('wrist angle', labelpad=8)
ax.set_zlabel('[degrees]', rotation='vertical', labelpad=10)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_zticks(zticks_alpha)
ax.set_zlim(zticks_alpha[0], zticks_alpha[-1])
ax.set_xticklabels([str(x) for x in xticks], rotation=41)
ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
# ax.set_yticklabels([str(x) for x in zticks_alpha])
ax.tick_params(axis='x', direction='out', pad=-5)
ax.tick_params(axis='y', direction='out', pad=-3)
ax.tick_params(axis='z', direction='out', pad=5)
if model.param_dims[0]>1:
    ax.plot_surface(X, Y, model.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, model.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# DISTANCE MODEL
ax = pl.subplot2grid((2,6),(0, 3), colspan=3, projection='3d')
ax.set_title('DISTANCE MODEL')
ax.set_ylabel('right dx', labelpad=5)
ax.set_xlabel('wrist angle', labelpad=8)
ax.set_zlabel('[cm]', rotation='vertical', labelpad=10)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_zticks(zticks_L)
ax.set_zlim(zticks_L[0], zticks_L[-1])
ax.set_xticklabels([str(x) for x in xticks], rotation=41)
ax.set_yticklabels([str(x) for x in yticks], rotation=-15)
# ax.set_yticklabels([str(x) for x in zticks_L])
ax.tick_params(axis='x', direction='out', pad=-5)
ax.tick_params(axis='y', direction='out', pad=-3)
ax.tick_params(axis='z', direction='out', pad=5)
if model.param_dims[0]>1:
    ax.plot_surface(X, Y, model.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, model.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# PENALISATION PDF
ax = pl.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
ax.set_title('Penalisation function: '+str(len(model.failed_params))+' points')
ax.set_ylabel('right dx', labelpad=5)
ax.set_xlabel('wrist angle', labelpad=5)
ax.set_xticks(xticks)
# ax.set_yticks(yticks)
ax.set_xticklabels([str(x) for x in xticks], rotation=50)
ax.set_yticklabels([str(x) for x in yticks], rotation=-20)
ax.tick_params(axis='x', direction='out', pad=-5)
ax.tick_params(axis='y', direction='out', pad=-3)
ax.tick_params(axis='z', direction='out', pad=2)
if model.param_dims[0]>1:
    ax.plot_surface(X, Y, (1-model.penal_PDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2))), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, (1-model.penal_PDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)

# UNCERTAINTY
ax = pl.subplot2grid((2,6),(1, 2), colspan=2, projection='3d')
ax.set_title('Model uncertainty: '+str(round(avar,4)))
ax.set_ylabel('right dx', labelpad=5)
ax.set_xlabel('wrist angle', labelpad=5)
ax.set_xticks(xticks)
# ax.set_yticks(yticks1)
ax.set_xticklabels([str(x) for x in xticks], rotation=50)
ax.set_yticklabels([str(x) for x in yticks], rotation=-20)
ax.tick_params(axis='x', direction='out', pad=-5)
ax.tick_params(axis='y', direction='out', pad=-3)
ax.tick_params(axis='z', direction='out', pad=7)
if model.param_dims[0]>1:
    ax.plot_surface(X, Y, model.var_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, model.var_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)

# SELECTION FUNCTION
ax = pl.subplot2grid((2,6),(1, 4), colspan=2, projection='3d')
ax.set_title('Selection function')
ax.set_ylabel('right dx', labelpad=5)
ax.set_xlabel('wrist angle', labelpad=5)
ax.set_xticks(xticks)
# ax.set_yticks(yticks)
ax.set_xticklabels([str(x) for x in xticks], rotation=50)
ax.set_yticklabels([str(x) for x in yticks], rotation=-20)
ax.tick_params(axis='x', direction='out', pad=-5)
ax.tick_params(axis='y', direction='out', pad=-3)
ax.tick_params(axis='z', direction='out', pad=5)
if model.param_dims[0]>1:
    ax.plot_surface(X, Y, model.info_pdf[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)
else:
    ax.plot_surface(X, Y, model.info_pdf[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)

# SAVEFIG
pl.savefig("Fig_for_paper_.svg")
pl.show()