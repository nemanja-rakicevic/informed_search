
import os
import math
import time
import rospy
import pickle
import numpy as np
import ik_solver
import baxter_interface as BI
from heapq import nsmallest


import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from geometry_msgs.msg import (PoseStamped,Pose,Point,Quaternion)
# from std_msgs.msg import (Float32MultiArray,UInt64MultiArray)
# from sensor_msgs.msg import Image

# import util_pdf as updf

##################################################################
# CONSTANTS - thresholds
THRSH_START = 10
THRSH_FORCE = 40
THRSH_POS = 0.01
THRSH_SPEED = 0.1
# CONSTANTS - stick length
STICK_X_MIN = 0.
STICK_X_MAX = 1.
STICK_Y_MIN = 0.
STICK_Y_MAX = 1.
##################################################################
# INITIAL POSE 
# # v6C, bigger angle span
initial_left = {'left_w0': -1.9017526817809416, 'left_w1': -1.5098205904762185, 'left_w2': 0.0, 'left_e0': -2.111141059327301, 'left_e1': 1.5926555530220308, 'left_s0': 0.3861796633501529, 'left_s1': 0.25349032519806464}
# initial_right = {'right_s0': -0.37155194764034827, 'right_s1': 0.947, 'right_w0': -1.9226134104693988, 'right_w1': 1.2720872783823325, 'right_w2': 0.9336995649992517, 'right_e0': 1.4905379406793826, 'right_e1': 1.8962337509333822}
initial_right = {'right_s0': -0.18604712804257897, 'right_s1': 0.7366307125790943, 'right_w0': -1.6419219445615647, 'right_w1': 1.4053479234536634, 'right_w2': 1.0948321182211216, 'right_e0': 1.5301695974547347, 'right_e1': 1.8269783292751778}

#####################################################################
#####################################################################

def sqdist(x,y):
    return np.sqrt(x**2 + y**2)


def checkStickConstraints(left_dx, left_dy, right_dx, right_dy, *_k):
    # Check if the parameters comply with stick dimension constraints  
    tmp_left = limb_left.endpoint_pose()['position']
    tmp_right = limb_right.endpoint_pose()['position']
    dx = abs((tmp_left.x + left_dx) - (tmp_right.x + right_dx))
    dy = abs((tmp_left.y + left_dy) - (tmp_right.y + right_dy))
    # print "/// STICK ", dx < STICK_X_MAX, dx < STICK_X_MAX and STICK_Y_MIN < dy < STICK_Y_MAX
    # print "/// STICK ", round(dx,2), round(dy,2)
    if dx <= STICK_X_MAX and dy <= STICK_Y_MAX:
         # abs(left_dx)>10*THRSH_POS and abs(left_dy)>10*THRSH_POS and\
         # abs(right_dx)>10*THRSH_POS and abs(right_dy)>10*THRSH_POS:
        return False
    else:
        return True


def getNewPose(left_dx, left_dy, right_dx, right_dy, w, speed):   
    # Get current position
    pose_tmp_left = limb_left.endpoint_pose()
    pose_tmp_right = limb_right.endpoint_pose()
    # Set new position
    new_pos_left = limb_left.Point( 
        x = pose_tmp_left['position'].x + left_dx, 
        y = pose_tmp_left['position'].y + left_dy, 
        z = pose_tmp_left['position'].z ) 
    new_pos_right = limb_right.Point( 
        x = pose_tmp_right['position'].x + right_dx, 
        y = pose_tmp_right['position'].y + right_dy, 
        z = pose_tmp_right['position'].z ) 
    # Get Joint positions
    joint_values_left = ik_solver.ik_solve('left', new_pos_left, pose_tmp_left['orientation'], limb_left.joint_angles())
    joint_values_right = ik_solver.ik_solve('right', new_pos_right, pose_tmp_right['orientation'], limb_right.joint_angles()) 
    # Set joint speed
    left_dL = sqdist(left_dx,left_dy)
    right_dL = sqdist(right_dx,right_dy) 
    if left_dL>right_dL:
        speed_left = speed
        try:
            speed_right = speed*right_dL/left_dL
        except:
            speed_right = 0
    else:
        speed_right = speed
        speed_left = speed*left_dL/right_dL
    # print speed_left, speed_right
    # return the joint values
    return joint_values_left, joint_values_right, speed_left, speed_right, new_pos_left, new_pos_right


def executeTrial(params):  
    # CHECK 1) Stick constraints
    if checkStickConstraints(*params):
        print '>> FAILED (Error 1: Stick constraints)'   
        return 0
    print "> TRIAL CHECK 1): OK Stick constraint"

    # GET IK SOLUTION
    joint_values_left, joint_values_right, speed_left, speed_right, new_pos_left, new_pos_right = getNewPose(*params) 
    # joint_values_left['left_w2'] = params[4]
    # CHECK 2) Inverse Kinematic solution
    if joint_values_left == -1 or joint_values_right == -1:
        trial.fail_status = 2
        trial.traj_cart[0].append(None)
        trial.traj_cart[1].append(None)
        trial.traj_jnt[0].append(None)
        trial.traj_jnt[1].append(None)  
        print '>> TRIAL # - FAILED (Error 2: No IK solution)'
        return 0
    print "> TRIAL CHECK 2): OK IK solution"

    # Passed constraint check ready to execute
    raw_input(">>> Ready to execute configuration: "+str((params))+"?\n")
    os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Stand clear'\"")   
    time.sleep(1)
    # EXECUTE MOTION
    # Set tip hit angle
    angle_left = limb_left.joint_angles()
    angle_left['left_w2'] = params[4]
    limb_left.set_joint_position_speed(1)
    limb_left.move_to_joint_positions(angle_left, timeout=1)
    # Set the speeds
    limb_left.set_joint_position_speed(speed_left)
    limb_right.set_joint_position_speed(speed_right)
    # EXECUTE MOTION and save/track progress
    # while not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) and \
    #     not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS)):
    cnt = 0
    while (not (tuple(np.array(joint_values_left.values())-THRSH_POS) <= tuple(limb_left.joint_angles().values()) <= tuple(np.array(joint_values_left.values())+THRSH_POS)) or \
        not (tuple(np.array(joint_values_right.values())-THRSH_POS) <= tuple(limb_right.joint_angles().values()) <= tuple(np.array(joint_values_right.values())+THRSH_POS))) and cnt <30000:
        cnt+=1
        # send joint commands
        limb_left.set_joint_positions(joint_values_left)
        limb_right.set_joint_positions(joint_values_right)

    return 1


def coord2vals(coord, param_list):
    return np.array([param_list[i][coord[i]] for i in range(len(param_list))])

###########################################################################################
# ROS Initialisation
rospy.init_node('HCK_PC_TEST_node')
rate = rospy.Rate(1000)

# Baxter initialisation
if not BI.RobotEnable().state().enabled:
    print("Enabling robot... ")
    BI.RobotEnable().enable()

limb_left = BI.Limb("left")
limb_right = BI.Limb("right")
############################################################################################

limb_left.move_to_joint_positions(initial_left, timeout=5)
limb_right.move_to_joint_positions(initial_right, timeout=5)

trialname = "TRIAL_20170206_17h13"
(M_angle, M_dist, var_angle, penal_PDF, param_list) = pickle.load(open(trialname+'/DATA_HCK_model_checkpoint.dat', "rb"))

# The dimensions which are plotted
p1 = 2
p2 = 4

angle_s, dist_s = input("\nEnter GOAL angle, distance: ")
### Do paired cartesian sqrt distance
# Select (angle, L) pair which is closest to the desired one
M_meas = sqdist(M_angle-angle_s, M_dist-dist_s)


# HOW TO DEAL WITH PENALISED PARAMETER COMBINATIONS?!?!!?
done = False
cnt = 0
while not done:
    cnt += 1
    # Get the indices of closest
    meas_idx = np.argwhere(M_meas==nsmallest(cnt, M_meas.ravel()))
    coords = meas_idx[cnt-1]
    exec_params = coord2vals(coords, param_list)

    error_angle = M_angle[tuple(coords)] - angle_s
    error_dist = M_dist[tuple(coords)] - dist_s
    print "\nERRORS>  angle:", error_angle, " distance:", error_dist

#### VISUALISATION
    dim1 = param_list[p1]
    dim2 = param_list[p2]
    X, Y = np.meshgrid(dim1, dim2)
    fig = pl.figure("PREDICTION", figsize=None)
    fig.set_size_inches(fig.get_size_inches()[0]*2,fig.get_size_inches()[1]*2)
    # ANGLE MODEL
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title('ANGLE MODEL')
    ax.set_xlabel('right dx')
    ax.set_ylabel('wrist angle')
    ax.set_zlabel('[degrees]', rotation='vertical')
    angle_2d = M_angle.reshape(len(dim1),len(dim2))
    #
    ax.plot_surface(X, Y, angle_2d, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_wireframe(X, Y, angle_2d, rstride=1, cstride=1)
    ax.plot([exec_params[p1],exec_params[p1]], [exec_params[p2],exec_params[p2]], [0, 35], linewidth=3)
    ax.plot([exec_params[p1],exec_params[p1]], [param_list[p2][0],param_list[p2][-1]], [angle_2d[coords[p1], coords[p2]], angle_2d[coords[p1], coords[p2]]], linewidth=3)
    ax.plot([param_list[p1][0],param_list[p1][-1]], [exec_params[p2],exec_params[p2]], [angle_2d[coords[p1], coords[p2]], angle_2d[coords[p1], coords[p2]]], linewidth=3)
    ax.scatter(exec_params[p1], exec_params[p2], angle_2d[coords[p1], coords[p2]], s=100, c='g')
    ax.scatter(exec_params[p1], exec_params[p2], angle_s, s=100, c='r')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # DISTANCE MODEL
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title('DISTANCE MODEL')
    ax.set_xlabel('right dx')
    ax.set_ylabel('wrist angle')
    ax.set_zlabel('[cm]', rotation='vertical')
    dist_2d = M_dist.reshape(len(dim1),len(dim2)).round()
    # ax.plot_surface(X, Y, dist_2d, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_wireframe(X, Y, dist_2d, rstride=1, cstride=1)
    ax.scatter(exec_params[p1], exec_params[p2], dist_2d[coords[p1], coords[p2]], s=100, c='g')
    # PENALISATION PDF
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title('Penalisation PDF')
    ax.set_xlabel('right dx')
    ax.set_ylabel('wrist angle')
    ax.plot_surface(X, Y, penal_PDF.reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)
    # UNCERTAINTY
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_xlabel('right dx')
    ax.set_ylabel('wrist angle')
    ax.set_title('Model uncertainty:'+str(round(var_angle.mean(),4)))
    ax.plot_surface(X, Y, var_angle.reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
    # SAVEFIG
    # pl.savefig(model.trial_dirname+"/IMG_HCK_distributions_step"+str(tr)+".png")
    pl.show()

#### EXECUTION
    done = executeTrial(exec_params)

print "\nEXECUTION DONE"




