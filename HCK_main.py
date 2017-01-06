#!/usr/bin/env python

#################################################
# MOBILE BASE
# ssh pi@192.168.2.105
# cd ~/ros_catkin_ws/src/mob_base_controls/src
# python movement_control.py 
#################################################
# LAPTOP
# ssh petar@192.168.0.2
# cd ~/ros_catkin_ws/src/.......
# python get_camera.py 
#################################################

import os
import math
import time
import rospy
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

from cv_bridge import CvBridge, CvBridgeError

import baxter_interface as BI

import ik_solver

from geometry_msgs.msg import  (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from std_msgs.msg import (
    Float32MultiArray,
    UInt64MultiArray,
)

from sensor_msgs.msg import Image

from baxter_core_msgs.msg import (
    CameraControl,
    CameraSettings,
    EndpointState,
    JointCommand,
)
from baxter_core_msgs.srv import (
    CloseCamera,
    ListCameras,
    OpenCamera,
)

#####################################################################

# CONSTANTS
THRSH_START = 10
THRSH_FORCE = 40
THRSH_POS = 0.01
THRSH_SPEED = 0.1

# GLOBAL
fx_left, fy_left, fz_left = 0, 0, 0
fx_right, fy_right, fz_right = 0, 0, 0
ball_x, ball_y = None, None

# Green filter - obtained using range-detector.py
# colorLower = (29, 86, 6)
# colorUpper = (64, 255, 255)
# Blue filter
colorLower = (73, 100, 190)
colorUpper = (106, 255, 255)

# INITIAL POSE (recorded)
# initial_left = {'left_w0': 0.1499466220157992, 'left_w1': 0.9836651802315216, 'left_w2': -0.6239466854723921, 'left_e0': -1.3173060015965992, 'left_e1': 1.349903093339164, 'left_s0': 0.9828981898375788, 'left_s1': -1.2524953133084402}
# initial_right = {'right_s0': 0.7760595517077958, 'right_s1': 0.5187698944592852, 'right_w0': -1.8506648712234506, 'right_w1': 1.0262244939226002, 'right_w2': 1.1394899859079073, 'right_e0': 1.8801431484869309, 'right_e1': 1.486003064569564}

initial_left = {'left_w0': 0.6599952339876992, 'left_w1': 0.7343933022001419, 'left_w2': -0.7202039799122018, 'left_e0': -1.77941771394708, 'left_e1': 1.485276897870052, 'left_s0': 1.0599807244288209, 'left_s1': -0.603237944835939}
initial_right = {'right_s0': 0.8895058321345872, 'right_s1': 0.5979099682041336, 'right_w0': -2.011470908979235, 'right_w1': 1.1274321217175909, 'right_w2': 1.1836823915603576, 'right_e0': 2.002373427888003, 'right_e1': 1.4542642968942092}

#####################################################################

# class EpisodeInfo():
#     def __init__(self,in,in2):
#         self.


def cleanup_on_shutdown():
    # cleanup, close any open windows
    # BI.RobotEnable().disable()
    cv2.destroyAllWindows()


def callback_cam(msg):
    global ball_x, ball_y
    # Read image
    try:
        frame = msg_bridge.imgmsg_to_cv2(msg)
    except CvBridgeError as e:
        print(e)
    # Extract blob position
    height, width, _ = frame.shape
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        ball_x = center[0]-width/2.0
        ball_y = -(center[1]-height/2.0)
        if radius > 1:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    else:
        ball_x = None
        ball_y = None
    # Show augmented image
    # cv2.imshow("Front Camera", frame)
    # cv2.waitKey(1)


def getForces(hand):
    if hand == 'left':
        f = limb_left.endpoint_effort()['force']
    elif hand == 'right':
        f = limb_right.endpoint_effort()['force']
    return [f.x, f.y, f.z]


def getForceOver():
    forces_left = getForces('left')
    forces_right = getForces('right')
    return any([-THRSH_FORCE<np.linalg.norm(forces_left)<THRSH_FORCE,-THRSH_FORCE<np.linalg.norm(forces_right)<THRSH_FORCE])


def getNewPose(dx_left=0,dy_left=0,dx_right=0,dy_right=0, speed_left=0.5, speed_right=0.5):   
    # Get current position
    pose_tmp_left = limb_left.endpoint_pose()
    pose_tmp_right = limb_right.endpoint_pose()
    # Set new position
    new_pos_left = limb_left.Point( 
        x = pose_tmp_left['position'].x + dx_left, 
        y = pose_tmp_left['position'].y + dy_left, 
        z = pose_tmp_left['position'].z ) 
    new_pos_right = limb_right.Point( 
        x = pose_tmp_right['position'].x + dx_right, 
        y = pose_tmp_right['position'].y + dy_right, 
        z = pose_tmp_right['position'].z ) 
    # Get Joint positions
    joint_values_left = ik_solver.ik_solve('left', new_pos_left, pose_tmp_left['orientation'], limb_left.joint_angles())
    joint_values_right = ik_solver.ik_solve('right', new_pos_right, pose_tmp_right['orientation'], limb_right.joint_angles()) 
    # Set joint speed
    limb_left.set_joint_position_speed(speed_left)
    limb_right.set_joint_position_speed(speed_right)

    return joint_values_left, joint_values_right

def getPuckDirection(puck_positions):
    # empty array
    if not pos_array:
        return None
    pos_array = np.asarray(puck_positions)
    # more than 20% failed
    if sum(pos_array == np.array(None)) > 0.2*len(pos_array):
        return None
    # calculate angle
    pos_array[pos_array != np.array(None)]
    #
    # LINEAR REGRESSION
    #


    return angle

def getPuckSpeed(puck_positions):
    if len(puck_positions)<2:
        return 0
    dt = puck_positions[-1][2] - puck_positions[-2][2]
    dl = np.sqrt((puck_positions[-1][0] - puck_positions[-2][0])**2 + (puck_positions[-1][1] - puck_positions[-2][1])**2)
    speed = dl/dt

    return speed

def executeTrial(dx_left=0,dy_left=0,dx_right=0,dy_right=0, speed_left=0.5, speed_right=0.5):  
    jnt_vals = [[],[]] 
    force_graph = [[],[]]
    puck_positions = []
    puck_speeds = []
    # get new joint positions based on displacements
    joint_values_left, joint_values_right = getNewPose(dx_left, dy_left, dx_right, dy_right, speed_left, speed_right)
    # print joint_values_left
    # print joint_values_right
    # Execute motion and track progress
    while (not tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) \ 
        and (getPuckSpeed(puck_positions) < THRSH_SPEED):
        # send joint commands
        limb_left.set_joint_positions(joint_values_left)
        limb_right.set_joint_positions(joint_values_right)
        # save joint movements
        jnt_vals[0].append(joint_values_left)
        jnt_vals[1].append(joint_values_right)
        # check feedback force
        # forces_left = getForces('left')
        # forces_right = getForces('right')
        frc_graph[0].append(getForces('left'))
        frc_graph[1].append(getForces('right'))
        # track puck positions
        puck_positions.append([ball_x, ball_y, time_step])
        puck_speeds.append(getPuckSpeed(puck_positions))

    # Calculate puck direction
    trial.puck_direction = getPuckDirection(puck_positions)
    # Calculate puck speed
    trial.puck_speed = np.mean(puck_speeds)
    # Save information
    trial.forces = frc_graph
    trial.joints = jnt_vals
    trial.puck_positions = puck_positions



# for i in range(0,10):
#     # p[0].append(0+0.1*i)
#     # p[1].append(0)
#     # p[2].append(i)

#     d.append([0+0.1*i,0,i])

    print 'Trial X - DONE'
    # if stuck, fix arms? and move forward with base, add rotation if necessarry
    # if completely stuck  - stop ?
    ### alternativelly:
    ###? move the obstacle out of the way
    ###? get feedback about successfull manipulaiton
    plt.figure(1)
    plt.subplot(211)
    plt.plot(frc_graph[0]), plt.title('Left force norm')
    plt.subplot(212)
    plt.plot(frc_graph[1]), plt.title('Rigth force norm')
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
limb_left.set_joint_position_speed(0.5)
limb_right.set_joint_position_speed(0.5)

#####################################################################
#####################################################################
# Main loop
#####################################################################
#####################################################################

limb_left.move_to_joint_positions(initial_left, timeout=3)
limb_right.move_to_joint_positions(initial_right, timeout=3)
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'System is ready!'\"") 
# time.sleep(10)

# while not rospy.is_shutdown():
#     # Get into initial position
#     limb_left.move_to_joint_positions(initial_left, timeout=3)
#     limb_right.move_to_joint_positions(initial_right, timeout=3)
    # time.sleep(5)
    # os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Clear'\"")   
    # executeTrial(-0.4,0,0.2,0)
    # executeTrial(0,0,0.2,0)
    # executeTrial(-0.4,0,0,0)
    # executeTrial(0,0,0,0)
#################
    # control_signals.data = [-20, 20, 0]
    # control_base.data = executeTrial()
    # pub_ctrl_base.publish(control_base)
    # print ball_x, ball_y

for ep in range(0,1):
    print ep
    # return to initial pose
    limb_left.move_to_joint_positions(initial_left, timeout=3)
    limb_right.move_to_joint_positions(initial_right, timeout=3)
    # check if ball in place ball_x in range...
    # while not (-THRSH_START < (ball_x, ball_y) < THRSH_START):
    #     print 'waiting..'
    # give voice signal 
    # os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Clear'\"")
    # generate sample motion parameters
    # executeTrial(-0.4,0,0.2,0,0.5,0.5)
    # execute motion
    # monitor effect
    # calculate speed
    # save pair

    # rospy.on_shutdown(cleanup_on_shutdown)
    # rate.sleep()

