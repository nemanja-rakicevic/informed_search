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
import matplotlib.pyplot as plt
import ik_solver
import baxter_interface as BI

from geometry_msgs.msg import  (PoseStamped,Pose,Point,Quaternion)
from std_msgs.msg import (Float32MultiArray,UInt64MultiArray)
from sensor_msgs.msg import Image

import utils_pdf as updf

#####################################################################

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
# CONSTANTS - cartesian
SPEED_MIN = 0.3
SPEED_MAX = 1

LEFT_X_MIN = -0.3
LEFT_X_MAX = 0.1

LEFT_Y_MIN = -0.8
LEFT_Y_MAX = 0.05

RIGHT_X_MIN = -0.05
RIGHT_X_MAX = 0.15

RIGHT_Y_MIN = -0.8
RIGHT_Y_MAX = 0.1

# INITIAL POSE v3 - best conf
initial_left = {'left_w0': -2.6461168591023387, 'left_w1': -0.9595049828223263, 'left_w2': -0.5668059011236604, 'left_e0': -1.9493060862053895, 'left_e1': 1.2202817167628466, 'left_s0': 0.9437816797465008, 'left_s1': -0.08321845774278369}
initial_right = {'right_s0': 0.9594823450680383, 'right_s1': 0.29484998388802847, 'right_w0': -2.098514074519654, 'right_w1': 0.8602590333104184, 'right_w2': 1.9667979410452, 'right_e0': 1.6474396266164186, 'right_e1': 1.4020464458749478}

#####################################################################

class TrialInfo:
    def __init__(self, num):
        self.num = num


def cleanup_on_shutdown():
    # cleanup, close any open windows
    BI.RobotEnable().disable()
    cv2.destroyAllWindows()


def generateDisplacement(tmp_left, tmp_right):
    bad_limit = True
    while bad_limit:
        # generate samples
        left_dx, left_dy, right_dx, right_dy, speed_left, speed_right = samplePDF(pdf, lower=True)
        # left_dx = np.random.uniform(LEFT_X_MIN, LEFT_X_MAX)
        # left_dy = np.random.uniform(LEFT_Y_MIN, LEFT_Y_MAX)
        # right_dx = np.random.uniform(RIGHT_X_MIN, RIGHT_X_MAX)
        # right_dy = np.random.uniform(RIGHT_Y_MIN, RIGHT_Y_MAX)
        # check constraints
        dx = abs((tmp_left.x + left_dx) - (tmp_right.x + right_dx))
        dy = abs((tmp_left.y + left_dy) - (tmp_right.y + right_dy))
        if dx < STICK_X_MAX and dy < STICK_Y_MAX:
             # abs(left_dx)>10*THRSH_POS and abs(left_dy)>10*THRSH_POS and\
             # abs(right_dx)>10*THRSH_POS and abs(right_dy)>10*THRSH_POS:
            bad_limit = False
            pdf = updatePDF(pdf, [left_dx, left_dy, right_dx, right_dy, speed_left, speed_right], cov)
        else:
            bad_limit = True
    # assign speeds
    # speed_left = np.random.uniform(SPEED_MIN, SPEED_MAX)
    # speed_right = np.clip(speed_left + np.random.uniform(-0.2,0.2), SPEED_MIN, SPEED_MAX)

    return left_dx, left_dy, right_dx, right_dy, speed_left, speed_right


def getNewPose():   
    # Get current position
    pose_tmp_left = limb_left.endpoint_pose()
    pose_tmp_right = limb_right.endpoint_pose()
    # Generate offsets
    left_dx, left_dy, right_dx, right_dy, speed_left, speed_right = generateDisplacement(pose_tmp_left['position'], pose_tmp_right['position'])
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
    limb_left.set_joint_position_speed(speed_left)
    limb_right.set_joint_position_speed(speed_right)

    return joint_values_left, joint_values_right, new_pos_left, new_pos_right, [left_dx, left_dy, right_dx, right_dy, speed_left, speed_right]



def executeTrial(trialnum):  
    trial = TrialInfo(trialnum)
    jnt_vals = [[],[]] 
    end_vals = [[],[]] 
    force_graph = [[],[]]
    puck_positions = [[ball_x, ball_y,0],[ball_x, ball_y,0]]
    puck_speeds = []
    get_joints = True
    cnt = 1
    # get new joint positions based on generated displacements
    while get_joints:
        joint_values_left, joint_values_right, new_pos_left, new_pos_right, params = getNewPose()
        if joint_values_left == -1 or joint_values_right == -1:
            print 'bad joint configuration...retrying...', cnt
            cnt+=1
        else:
            get_joints = False
    # print joint_values_left
    # print joint_values_right
    print 'Left:',params[0], params[1]
    print 'Right:',params[2], params[3]
    raw_input("Press ENTER to continue.")

    # Execute motion and track progress
    time_start = time.time()
    while not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) and \
        not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS)):
        # send joint commands
        limb_left.set_joint_positions(joint_values_left)
        limb_right.set_joint_positions(joint_values_right)
        # save joint movements
        jnt_vals[0].append(joint_values_left)
        jnt_vals[1].append(joint_values_right)
        # save end-effector movements
        end_vals[0].append(limb_left.endpoint_pose()['position'])
        end_vals[1].append(limb_right.endpoint_pose()['position'])
        # check feedback force
        # forces_left = getForces('left')
        # forces_right = getForces('right')
        force_graph[0].append(getForces('left'))
        force_graph[1].append(getForces('right'))
        # track puck positions
        puck_positions.append([ball_x, ball_y, time.time()-time_start])
        # puck_speeds.append(getPuckSpeed(puck_positions))


    # CHECK IF TRIAL FAILED
    trial.status = False
    # Save features and results
    # Calculate puck direction
    trial.puck_direction = getPuckDirection(puck_positions)
    print 'Angle:',trial.puck_direction
    # Calculate puck speed
    trial.puck_speed = getPuckSpeed(puck_positions)
    print 'Speed:',trial.puck_speed
    # other
    trial.forces = force_graph
    trial.trajectory_joints = jnt_vals
    trial.trajectory_cartesian = end_vals
    trial.position_final = params
    trial.puck_positions = np.asarray(puck_positions)
    print 'Trial:',trialnum,'- DONE'
    plt.figure(1)
    plt.subplot(211)
    plt.plot(force_graph[0]), plt.title('Left force norm')
    plt.subplot(212)
    plt.plot(force_graph[1]), plt.title('Rigth force norm')
    # plt.show()
    plt.figure(2)
    plt.subplot(211)
    plt.plot(end_vals[0]), plt.title('Left cartesian')
    plt.subplot(212)
    plt.plot(end_vals[1]), plt.title('Rigth cartesian')
    #
    plt.figure(3)
    plt.plot(puck_speeds)
    plt.show()

    return trial

#####################################################################
rospy.init_node('HCK_PC_main_node')

pub_ctrl_base = rospy.Publisher('HCK_control_base', Float32MultiArray, queue_size=1)
pub_ctrl_rArm = rospy.Publisher('HCK_control_rArm', JointCommand, queue_size=1)
pub_ctrl_lArm = rospy.Publisher('HCK_control_lArm', JointCommand, queue_size=1)
sub_ball_img = rospy.Subscriber('HCK_ball_img', Image, callback=callback_cam)

control_base = Float32MultiArray()
msg_bridge = CvBridge()
stationary_mode = [0, 0]
control_base.data = stationary_mode

rate = rospy.Rate(100)

#####################################################################
# Baxter initialisation
if not BI.RobotEnable().state().enabled:
    print("Enabling robot... ")
    BI.RobotEnable().enable()

limb_left = BI.Limb("left")
limb_right = BI.Limb("right")
limb_left.set_joint_position_speed(0.3)
limb_right.set_joint_position_speed(0.3)

#####################################################################
#####################################################################
# Main loop
#####################################################################
#####################################################################

limb_left.move_to_joint_positions(initial_left, timeout=3)
limb_right.move_to_joint_positions(initial_right, timeout=3)
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'System is ready!'\"") 
# time.sleep(10)


while not rospy.is_shutdown():
#     # Get into initial position
    limb_left.move_to_joint_positions(initial_left, timeout=3)
    limb_right.move_to_joint_positions(initial_right, timeout=3)
    # time.sleep(5)
    # os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Clear'\"")   

    labels_list = []
    trialList = []
    trial_sess = []

    for trnum in range(1):
        limb_left.move_to_joint_positions(initial_left, timeout=3)
        limb_right.move_to_joint_positions(initial_right, timeout=3)
        print "\nStep", i+1
        raw_input("Ready to execute configuration: "+str((left_dx, left_dy, right_dx, right_dy, speed))+"?")
        
        # (left_dx, left_dy, right_dx, right_dy, speed)
        fail_flag = executeTrial(left_dx, left_dy, right_dx, right_dy, speed)
        # MANUALLY ENTER LABELS when ready
        labels_list.append(input('\nEnter angle, distance: '))
        # # give voice signal to start
        # print 'Ready for trial:',ep
        # os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Ready my master!'\"")
        raw_input("Press ENTER to continue.")
        # generate sample motion parameters
        trial = executeTrial(trnum)
        # save results
        if trial:
            trialList = append(trial)
            with open('TrialInfo.dat', "wb") as f:
                pickle.dump(trialList, f)

    print '\nDONE! saving..'
    with open('TrialInfo.dat', "wb") as f:
                pickle.dump(trialList, f)


rospy.on_shutdown(cleanup_on_shutdown)
# rate.sleep()
rospy.spin()

