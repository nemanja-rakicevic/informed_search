#!/usr/bin/env python

#################################################
# MOBILE BASE
# ssh pi@192.168.2.105
# cd ~/ros_catkin_ws/src/mob_base_controls/src
# python movement_control.py 
#################################################
# LAPTOP
# ssh petar@192.168.0.2
# cd ~/ros_ws/robin_lab/nemanja/hockey_demo
# python HCK_DENIRO_camera.py 
#################################################



import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import os
import math
import time
import rospy
import cPickle as pickle
import numpy as np
import ik_solver
import baxter_interface as BI

from geometry_msgs.msg import (PoseStamped,Pose,Point,Quaternion)
# from std_msgs.msg import (Float32MultiArray,UInt64MultiArray)
# from sensor_msgs.msg import Image

import util_pdf as updf

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
STICK_Y_MAX = 0.7
##################################################################
# INITIAL POSE 
# # v6C, bigger angle span
initial_left = {'left_w0': -1.9017526817809416, 'left_w1': -1.5098205904762185, 'left_w2': 0.0, 'left_e0': -2.111141059327301, 'left_e1': 1.5926555530220308, 'left_s0': 0.3861796633501529, 'left_s1': 0.25349032519806464}
# initial_right = {'right_s0': -0.37155194764034827, 'right_s1': 0.947, 'right_w0': -1.9226134104693988, 'right_w1': 1.2720872783823325, 'right_w2': 0.9336995649992517, 'right_e0': 1.4905379406793826, 'right_e1': 1.8962337509333822}
initial_right = {'right_s0': -0.18604712804257897, 'right_s1': 0.7366307125790943, 'right_w0': -1.6419219445615647, 'right_w1': 1.4053479234536634, 'right_w2': 1.0948321182211216, 'right_e0': 1.5301695974547347, 'right_e1': 1.8269783292751778}

#####################################################################
#####################################################################

class TrialInfo:
    def __init__(self, num):
        self.num = num
        self.traj_jnt = [[],[]] 
        self.traj_cart = [[],[]]


def cleanup_on_shutdown():
    # cleanup, close any open windows
    print "\nSaving at exit...\n"
    with open(model.trial_dirname+"/DATA_HCK_trial_checkpoint.dat", "wb") as f:
        pickle.dump((trials_list, labels_list, info_list, model.failed_params), f)

    BI.RobotEnable().disable()
    cv2.destroyAllWindows()


def sqdist(x,y):
    return np.sqrt(x**2 + y**2)


def checkStickConstraints(left_dx, left_dy, right_dx, right_dy, *_k):
    # Check if the parameters comply with stick dimension constraints  
    tmp_left = limb_left.endpoint_pose()['position']
    tmp_right = limb_right.endpoint_pose()['position']
    dx = abs((tmp_left.x + left_dx) - (tmp_right.x + right_dx))
    dy = abs((tmp_left.y + left_dy) - (tmp_right.y + right_dy))
    # print "/// STICK ", dx < STICK_X_MAX, dx < STICK_X_MAX and STICK_Y_MIN < dy < STICK_Y_MAX
    print "/// STICK ", round(dx,2), round(dy,2)
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


def executeTrial(trialnum, params):  
    resp = -1
    while True:

        if resp==5:
            limb_right.move_to_joint_positions(initial_right, timeout=5)
            limb_left.move_to_joint_positions(initial_left, timeout=5)

        trial = TrialInfo(trialnum)
        # CHECK 1) Stick constraints
        if checkStickConstraints(*params):
            trial.fail_status = 1
            trial.traj_cart[0].append(None)
            trial.traj_cart[1].append(None)
            trial.traj_jnt[0].append(None)
            trial.traj_jnt[1].append(None)
            print '>> TRIAL #',trialnum," - FAILED (Error 1: Stick constraints)"   
            return trial
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
            print '>> TRIAL #',trialnum," - FAILED (Error 2: No IK solution)" 
            return trial
        print "> TRIAL CHECK 2): OK IK solution"

        # Passed constraint check ready to execute
        raw_input(">>> Ready to execute configuration: "+str((params))+"?\n")
        # os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Stand clear'\"")   
        time.sleep(1)
        # EXECUTE MOTION
        # Set tip hit angle
        angle_left = limb_left.joint_angles()
        angle_left['left_w2'] = params[4]
        limb_left.set_joint_position_speed(1)
        limb_left.move_to_joint_positions(angle_left, timeout=2)
        # Set the speeds
        limb_left.set_joint_position_speed(speed_left)
        limb_right.set_joint_position_speed(speed_right)
        #
        # joint_values_left['left_w2'] = params[4]

        # # ## EXECUTE MOTION and save/track progress
        # # while not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) and \
        # #     not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS)):
        # cnt = 0
        # while (not (tuple(np.array(joint_values_left.values())-THRSH_POS) <= tuple(limb_left.joint_angles().values()) <= tuple(np.array(joint_values_left.values())+THRSH_POS)) or \
        #     not (tuple(np.array(joint_values_right.values())-THRSH_POS) <= tuple(limb_right.joint_angles().values()) <= tuple(np.array(joint_values_right.values())+THRSH_POS))) and cnt <20000:
        #     cnt+=1
        #     # send joint commands
        #     limb_left.set_joint_positions(joint_values_left)
        #     limb_right.set_joint_positions(joint_values_right)
        #     # save joint movements
        #     trial.traj_jnt[0].append(limb_left.joint_angles())
        #     trial.traj_jnt[1].append(limb_right.joint_angles())
        #     # save end-effector movements
        #     trial.traj_cart[0].append(limb_left.endpoint_pose()['position'][0:3])
        #     trial.traj_cart[1].append(limb_right.endpoint_pose()['position'][0:3])


        # CHECK 3) PHYSICAL EFFECT
        while True:
            try:
                resp = input("\n> TRIAL CHECK 3): What happened? [(0): All OK; (1): Fuse broke; (2): No effect; (5): Repeat trial] ")
                if not(resp==0 or resp==1 or resp==2 or resp==5):
                    continue
            except:
                continue
            else:
                break
        if resp==5:
            print "Repeating trial ...\n"
            continue            
        elif resp==0:
            trial.fail_status = 0
            print '>> TRIAL:',trialnum,' - SUCCESS'
            return trial
        elif resp==1:
            print '>> TRIAL #',trialnum," - FAILED (Error 3: Fuse broke)"
            trial.fail_status = 3
            return trial
        elif resp==2:
            print '>> TRIAL #',trialnum," - FAILED (Error 4: Puck didn't move)"
            trial.fail_status = 4
            return trial


#####################################################################
#####################################################################

# ROS Initialisation
rospy.init_node('HCK_PC_main_node')
rate = rospy.Rate(1000)

# Baxter initialisation
if not BI.RobotEnable().state().enabled:
    print("Enabling robot... ")
    BI.RobotEnable().enable()

limb_left = BI.Limb("left")
limb_right = BI.Limb("right")


#####################################################################
# Main loop
#####################################################################

# while not rospy.is_shutdown():

# Get into initial position
limb_left.move_to_joint_positions(initial_left, timeout=5)
limb_right.move_to_joint_positions(initial_right, timeout=5)
# time.sleep(5)
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Clear'\"")  
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Ready my master!'\"") 

trials_list = []
info_list = []
labels_list = []

print "Calculating Kss ... \n"
model = updf.PDFoperations()

#################################################################
# for tr in range(100):
tr = 0
while True:
    tr+=1
    limb_right.move_to_joint_positions(initial_right, timeout=2)
    limb_left.move_to_joint_positions(initial_left, timeout=2)
    limb_right.move_to_joint_positions(initial_right, timeout=3)
    limb_left.move_to_joint_positions(initial_left, timeout=3)

##### GENERATE SAMPLE
    print "\n==================="
    print "===== Step", tr,"====="
    print "==================="
    avar, lvar = model.returnUncertainty()
    print "- Generating new trial parameters...\n- Current model entropy:"+str(round(avar,4))
    trial_params = model.generateSample(np.array(trials_list), np.array(labels_list))
    trials_list.append(trial_params)
    print "---generated params:", trial_params

##### EXECUTE TRIAL      
    # (left_dx, left_dy, right_dx, right_dy, speed)
    trial_info = executeTrial(tr, trial_params)
    info_list.append(trial_info)

##### MANUALLY ENTER LABELS when ready
    if trial_info.fail_status==0:    
        while True:
            try:
                trial_label = input("\n- Enter angle, distance: ")
                if len(trial_label)!=2:
                    continue
            except:
                continue
            else:
                break
        labels_list.append(trial_label)
        model.updatePDF(trial_params, -1)
    else:
        labels_list.append([None, None])
        ### if task has failed, remember where it failed
        model.updatePDF(trial_params)
######## VISUALISE PREDICTIONS
    if tr%1==0 or trial_info.fail_status==0:
        print "<- CHECK PLOTS"     
        if len(model.mu_alpha) and len(model.mu_L):
            dim1 = model.param_list[2]
            dim2 = model.param_list[4]
            X, Y = np.meshgrid(dim2, dim1)
            fig = pl.figure("DISTRIBUTIONs at step: "+str(tr), figsize=None)
            fig.set_size_inches(fig.get_size_inches()[0]*2,fig.get_size_inches()[1]*2)
            # ANGLE MODEL
            ax = pl.subplot2grid((2,6),(0, 0), colspan=3, projection='3d')
            ax.set_title('ANGLE MODEL')
            ax.set_ylabel('right dx')
            ax.set_xlabel('wrist angle')
            ax.set_zlabel('[degrees]', rotation='vertical')
            if model.param_dims[0]>1:
                ax.plot_surface(X, Y, model.mu_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            else:
                ax.plot_surface(X, Y, model.mu_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            # DISTANCE MODEL
            ax = pl.subplot2grid((2,6),(0, 3), colspan=3, projection='3d')
            ax.set_title('DISTANCE MODEL')
            ax.set_ylabel('right dx')
            ax.set_xlabel('wrist angle')
            ax.set_zlabel('[cm]', rotation='vertical')
            if model.param_dims[0]>1:
                ax.plot_surface(X, Y, model.mu_L[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            else:
                ax.plot_surface(X, Y, model.mu_L[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # PENALISATION PDF
            ax = pl.subplot2grid((2,6),(1, 0), colspan=2, projection='3d')
            ax.set_title('Penalisation function: '+str(len(model.failed_params))+' points')
            ax.set_ylabel('right dx')
            ax.set_xlabel('wrist angle')
            if model.param_dims[0]>1:
                ax.plot_surface(X, Y, model.penal_PDF[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)
            else:
                ax.plot_surface(X, Y, (1-model.penal_PDF[0,0,:,0,:,0].reshape(len(dim1),len(dim2))), rstride=1, cstride=1, cmap=cm.copper, linewidth=0, antialiased=False)
            # UNCERTAINTY
            ax = pl.subplot2grid((2,6),(1, 2), colspan=2, projection='3d')
            ax.set_ylabel('right dx')
            ax.set_xlabel('wrist angle')
            ax.set_title('Model uncertainty: '+str(round(avar,4)))
            if model.param_dims[0]>1:
                ax.plot_surface(X, Y, model.var_alpha[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
            else:
                ax.plot_surface(X, Y, model.var_alpha[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=False)
            # SELECTION FUNCTION
            ax = pl.subplot2grid((2,6),(1, 4), colspan=2, projection='3d')
            ax.set_title('Selection function')
            ax.set_ylabel('right dx')
            ax.set_xlabel('wrist angle')
            if model.param_dims[0]>1:
                ax.plot_surface(X, Y, model.info_pdf[0,3,:,3,:,4].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)
            else:
                ax.plot_surface(X, Y, model.info_pdf[0,0,:,0,:,0].reshape(len(dim1),len(dim2)), rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)
            # SAVEFIG
            pl.savefig(model.trial_dirname+"/IMG_HCK_distributions_step"+str(tr)+".png")
            pl.show()

##### SAVE FEATURES and LABELS
    if tr%1==0:
        print "\nSaving progress...\n"
        with open(model.trial_dirname+"/DATA_HCK_trial_checkpoint.dat", "wb") as f:
            pickle.dump((trials_list, labels_list, info_list, model.failed_params), f)

    # PROMPT for continuation
    if tr%100==0:
        quit = raw_input("\n\n=============== Continue 100 more iterations? [x to quit]\n")
        if quit == 'x' or  quit == 'X':
            break

    # CHECK if the parameter space has been exhaustively searched
    if not tr<len(model.param_space):
        print "========================================"
        print " ALL POSSIBLE COMBINATIONS TRIED. EXIT."
        print "========================================"

#################################################################

print '\nDONE! saving..'
with open(model.trial_dirname+"/DATA_HCK_trial_checkpoint.dat", "wb") as f:
    pickle.dump((trials_list, labels_list, info_list, model.failed_params), f)


rospy.on_shutdown(cleanup_on_shutdown)
# rate.sleep()
# rospy.spin()

