
import os
import math
import time
import rospy
import pickle
import numpy as np
import ik_solver
import baxter_interface as BI

from geometry_msgs.msg import (PoseStamped,Pose,Point,Quaternion)
# from std_msgs.msg import (Float32MultiArray,UInt64MultiArray)
# from sensor_msgs.msg import Image

import util_pdf as updf

flag = True
##################################################################
# CONSTANTS - thresholds
THRSH_START = 10
THRSH_FORCE = 40
THRSH_POS = 0.01
THRSH_SPEED = 0.1
# CONSTANTS - stick length
STICK_X_MIN = 0.
STICK_X_MAX = 0.7

STICK_Y_MIN = 0.
STICK_Y_MAX = 0.7
# CONSTANTS - speed 
SPEED_MIN = 0.4
SPEED_MAX = 1
# CONSTANTS - left arms
LEFT_X_MIN = -0.4
LEFT_X_MAX = 0.
LEFT_Y_MIN = -0.8
LEFT_Y_MAX = 0.0
# CONSTANTS - right arm
RIGHT_X_MIN = 0.
RIGHT_X_MAX = 0.2
RIGHT_Y_MIN = -0.8
RIGHT_Y_MAX = 0.
##################################################################
## max length of combination vector should be 25000 - 8/7/8/7/8
# range_l_dx = np.round(np.linspace(LEFT_X_MIN, LEFT_X_MAX, 8), 3)
# range_l_dy = np.round(np.linspace(LEFT_Y_MIN, LEFT_Y_MAX, 7), 3)
# range_r_dx = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 8), 3)
# range_r_dy = np.round(np.linspace(RIGHT_Y_MIN, RIGHT_Y_MAX, 7), 3)
# range_v = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 8), 3)
##################################################################(-0.1,0, 0.2,0, s)
range_l_dx = np.round(np.linspace(-0.2, -0.2, 1), 4)
range_l_dy = np.round(np.linspace(0, 0, 1), 4)
range_r_dx = np.round(np.linspace(0.1, 0.1, 1), 4)
range_r_dy = np.round(np.linspace(0, 0, 1), 4)
range_v = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 20), 3)

param_list = np.array([range_l_dx, range_l_dy, range_r_dx, range_r_dy, range_v])

##################################################################
(angle, L, var_angle, var_L) = pickle.load(open("DATA_HCK_model_checkpoint.dat", "rb"))

def sqdist(x,y):
    return np.sqrt(x**2 + y**2)

def coord2vals(coord, param_list):
    return np.array([param_list[i][coord[i]] for i in range(len(param_list))])

### OPTION 1
# Select (angle, L) pair which is closest to the desired one
meas = (angle-angle_s)* (L-L_s)
# Get the indices of closest
meas_idx = np.argwhere(info_pdf==np.max(info_pdf))
# locally refine

### OPTION 2
# check if L*alpha is in range around the goal


# Execute obtained parameters
