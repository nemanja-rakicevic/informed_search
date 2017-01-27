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
# # CONSTANTS - speed 
# SPEED_MIN = 0.4
# SPEED_MAX = 1
# # CONSTANTS - left arms
# LEFT_X_MIN = -0.3
# LEFT_X_MAX = 0.1
# LEFT_Y_MIN = -0.8
# LEFT_Y_MAX = 0.05
# # CONSTANTS - right arm
# RIGHT_X_MIN = -0.05
# RIGHT_X_MAX = 0.15
# RIGHT_Y_MIN = -0.8
# RIGHT_Y_MAX = 0.1
##################################################################
# INITIAL POSE 
# #v3 - (30, 5)
# initial_left = {'left_w0': -2.6461168591023387, 'left_w1': -0.9595049828223263, 'left_w2': -0.5668059011236604, 'left_e0': -1.9493060862053895, 'left_e1': 1.2202817167628466, 'left_s0': 0.9437816797465008, 'left_s1': -0.08321845774278369}
# initial_right = {'right_s0': 1.032759546843563, 'right_s1': 0.4353110245074654, 'right_w0': -2.248856247084555, 'right_w1': 1.018974816124536, 'right_w2': 1.9044125747349738, 'right_e0': 1.8281707804907916, 'right_e1': 1.3464587499877072}

# #v4 - (35,20)
# initial_left = {'left_w0': -2.89577223233069, 'left_w1': -1.1600729708383442, 'left_w2': -1.6582332317041322, 'left_e0': -1.448461358960802, 'left_e1': 0.7669903939427068, 'left_s0': 0.27113110425874687, 'left_s1': -0.7060146576242616}
# # initial_right = {'right_s0': 0.4415068194922188, 'right_s1': 0.947, 'right_w0': -2.742287838168007, 'right_w1': 0.9780381055677475, 'right_w2': 1.1313534973875081, 'right_e0': 1.881106510746034, 'right_e1': 1.9022031896138627}
# initial_right = {'right_s0': 0.800466048527356, 'right_s1': 0.3491576638939382, 'right_w0': -2.959, 'right_w1': 1.5025484131118265, 'right_w2': 2.935518625428416, 'right_e0': 1.559681467805579, 'right_e1': 1.7002814644295619}

# #v5 - (38, 16)
initial_left = {'left_w0': -2.8091023178151637, 'left_w1': -1.088359369004701, 'left_w2': -1.3874856226423566, 'left_e0': -1.3798157187029296, 'left_e1': 0.9993884833073471, 'left_s0': 0.5702573578964025, 'left_s1': -0.705247667230319}
initial_right = {'right_s0': 0.877122385358828, 'right_s1': 0.38881368501200375, 'right_w0': -2.959, 'right_w1': 1.1286424474647607, 'right_w2': 2.932984543928958, 'right_e0': 1.4928188728981575, 'right_e1': 1.5984069969052348}


#####################################################################
#####################################################################

class TrialInfo:
    def __init__(self, num):
        self.num = num
        self.traj_jnt = [[],[]] 
        self.traj_cart = [[],[]]


def cleanup_on_shutdown():
    # cleanup, close any open windows
    BI.RobotEnable().disable()
    cv2.destroyAllWindows()


def sqdist(x,y):
    return np.sqrt(x**2 + y**2)


def checkStickConstraints(left_dx, left_dy, right_dx, right_dy, _):
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


def getNewPose(left_dx, left_dy, right_dx, right_dy, speed):   
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

    joint_values_left, joint_values_right, speed_left, speed_right, new_pos_left, new_pos_right = getNewPose(*params) 
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
    raw_input(">>> Ready to execute configuration: "+str((trial_params))+"?\n") 
    # Set the speeds
    limb_left.set_joint_position_speed(speed_left)
    limb_right.set_joint_position_speed(speed_right)
    # EXECUTE MOTION and save/track progress
    while not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) and \
        not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS)):
        # send joint commands
        limb_left.set_joint_positions(joint_values_left)
        limb_right.set_joint_positions(joint_values_right)
        # save joint movements
        trial.traj_jnt[0].append(limb_left.joint_angles())
        trial.traj_jnt[1].append(limb_right.joint_angles())
        # save end-effector movements
        trial.traj_cart[0].append(limb_left.endpoint_pose()['position'][0:3])
        trial.traj_cart[1].append(limb_right.endpoint_pose()['position'][0:3])


    # CHECK 3) PHYSICAL EFFECT
    while True:
        try:
            resp = input("\n> TRIAL CHECK 3): What happened? [(0): All OK; (1): Fuse broke; (2): No effect] ")
            if not(resp==0 or resp==1 or resp==2):
                continue
        except:
            continue
        else:
            break
    if resp==0:
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
limb_left.move_to_joint_positions(initial_left, timeout=3)
limb_right.move_to_joint_positions(initial_right, timeout=3)
# time.sleep(5)
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Clear'\"")  
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Ready my master!'\"") 

params_list = []
info_list = []
labels_list = []

model = updf.PDFoperations()

#################################################################
# for tr in range(100):
tr = 0
while True:
    tr+=1
    limb_left.move_to_joint_positions(initial_left, timeout=5)
    limb_right.move_to_joint_positions(initial_right, timeout=5)

##### GENERATE SAMPLE
    print "\n==================="
    print "===== Step", tr,"====="
    avar, lvar = model.returnUncertainty()
    print "- Generating new trial parameters...\n- Current entropy [alpha:"+str(round(avar,4))+"; L:"+str(round(lvar,4))+"]"
    trial_params = model.generateSample(np.array(params_list), np.array(labels_list))
    params_list.append(trial_params)
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
    else:
        labels_list.append([None, None])
        ### if task has failed, remember where it failed
        model.updatePDF(trial_params)


##### SAVE FEATURES and LABELS
    with open("DATA_HCK_trial_checkpoint"+model.date+".dat", "wb") as f:
        pickle.dump((params_list, labels_list, info_list, model.failed_list), f)

    if not tr%100:
        quit = raw_input("\n\n=============== Continue 100 more iterations? [x to quit]\n")
        if quit == 'x' or  quit == 'X':
            break

    if not tr<len(model.param_space):
        print "========================================"
        print " ALL POSSIBLE COMBINATIONS TRIED. EXIT."
        print "========================================"
#################################################################

print '\nDONE! saving..'
with open("DATA_HCK_trial_checkpoint"+model.date+".dat", "wb") as f:
    pickle.dump((params_list, labels_list, info_list, model.failed_list), f)


rospy.on_shutdown(cleanup_on_shutdown)
# rate.sleep()
# rospy.spin()

