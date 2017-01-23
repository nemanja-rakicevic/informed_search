import os
import math
import time
import rospy
import cv2
import imutils
import pickle
import numpy as np
import matplotlib.pyplot as plt

import ik_solver
import baxter_interface as BI

from cv_bridge import CvBridge, CvBridgeError

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

# mu_conv = pickle.load( open( "mu_cam640x480toM.dat", "rb" ) )

#####################################################
# Blue filter
# colorLower = (75, 138, 228)
# colorUpper = (125, 255, 255)
#
colorLower = (73, 100, 190)
colorUpper = (106, 255, 255)
#daylight
# colorLower = (54, 73, 255)
# colorUpper = (93, 150, 255)

### INITIAL POSES
# INITIAL POSE v3 - best conf
initial_left = {'left_w0': -2.6461168591023387, 'left_w1': -0.9595049828223263, 'left_w2': -0.5668059011236604, 'left_e0': -1.9493060862053895, 'left_e1': 1.2202817167628466, 'left_s0': 0.9437816797465008, 'left_s1': -0.08321845774278369}
initial_right = {'right_s0': 0.9594823450680383, 'right_s1': 0.29484998388802847, 'right_w0': -2.098514074519654, 'right_w1': 0.8602590333104184, 'right_w2': 1.9667979410452, 'right_e0': 1.6474396266164186, 'right_e1': 1.4020464458749478}

### KINECT PARAMS
CameraPosition = {
    "x": 0, # actual position in meters of kinect sensor relative to the viewport's center.
    "y": 0, # actual position in meters of kinect sensor relative to the viewport's center.
    "z": 1.57, # height in meters of actual kinect sensor from the floor.
    "roll": 0, # angle in degrees of sensor's roll (used for INU input - trig function for this is commented out by default).
    "azimuth": 0, # sensor's yaw angle in degrees.
    "elevation": -45, # sensor's pitch angle in degrees.
}

#####################################################

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
        ball_x = center[0]
        ball_y = center[1]
        # ball_y = -(center[0]-width/2.0)
        # ball_x = center[0]-width/2.0
        # ball_y = -(center[1]-height/2.0)
        if radius > 1:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    else:
        ball_x = None
        ball_y = None
    # Add text for puck position
    cv2.putText(frame, "x _pos: {}, y_pos: {}".format(ball_x, ball_y),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)
    # Show augmented image
    cv2.imshow("Front Camera", frame)
    cv2.waitKey(1)


def callback_depth(msg):
    # global ball_x, ball_y
    # Read image
    try:
        frame = msg_bridge.imgmsg_to_cv2(msg)
    except CvBridgeError as e:
        print(e)
    # Extract blob position
    # height, width, dval = frame.shape
    height= frame.shape

    print frame[ball_x,ball_y]


# def find_nearest(array,value):
#     idx = (np.abs(array-value)).argmin()
#     return array[idx]

### CONVERSION
def px2m_original(px_x, px_y):
    try:
        x_out = mu_conv[np.abs(mu_conv[:,0]-px_x/100.).argmin(), 1]
    except:
        x_out = None

    try:
        y_out = 1.2*px_y
    except:
        y_out = None

    return x_out, y_out


#####################################################################
rospy.init_node('HCK_PC_main_node')

pub_ctrl_base = rospy.Publisher('HCK_control_base', Float32MultiArray, queue_size=1)
pub_ctrl_rArm = rospy.Publisher('HCK_control_rArm', JointCommand, queue_size=1)
pub_ctrl_lArm = rospy.Publisher('HCK_control_lArm', JointCommand, queue_size=1)
sub_ball_img = rospy.Subscriber('/kinect2/sd/image_color_rect', Image, callback=callback_cam)
sub_ball_depth = rospy.Subscriber('/kinect2/sd/image_depth_rect', Image, callback=callback_depth)

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

# limb_left.move_to_joint_positions(initial_left, timeout=3)
# limb_right.move_to_joint_positions(initial_right, timeout=3)
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'System is ready!'\"") 
# time.sleep(10)


while not rospy.is_shutdown():
#     # Get into initial position
    # limb_left.move_to_joint_positions(initial_left, timeout=3)
    # limb_right.move_to_joint_positions(initial_right, timeout=3)
    # time.sleep(5)
    # os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Clear'\"")   
    # executeTrial(-0.4,0,0.2,0)
    # executeTrial(0,0,0.2,0)
    # executeTrial(-0.4,0,0,0)
    # executeTrial(0,0,0,0)
    # x_curr, y_curr = px2m_original(ball_x, ball_y)
    pass
    # if x_curr and y_curr:
    #     print "X:",round(x_curr,2)," Y:", round(y_curr,2)
    # else:
    #     print "Puck not found"





rospy.on_shutdown(cleanup_on_shutdown)
# rate.sleep()
rospy.spin()

