
# Import stuff
import time
import rospy
import numpy as np

import ik_solver
import baxter_interface as BI

# from dynamic_reconfigure.server import (
#     Server,
# )

# from baxter_examples.cfg import (
#     JointSpringsExampleConfig,
# )

#####################################################################

# CONSTANTS - thresholds
THRSH_START = 10
THRSH_FORCE = 40
THRSH_POS = 0.0051
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
#####################################################################

# Check range

def sqdist(x,y):
    return np.sqrt(x**2 + y**2)

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
        speed_right = speed*right_dL/left_dL
    else:
        speed_right = speed
        speed_left = speed*left_dL/right_dL

    # print speed_left, speed_right
    limb_left.set_joint_position_speed(speed_left)
    limb_right.set_joint_position_speed(speed_right)

    return joint_values_left, joint_values_right, new_pos_left, new_pos_right


def executeTrial(left_dx, left_dy, right_dx, right_dy, speed=0.9):  
    joint_values_left, joint_values_right, new_pos_left, new_pos_right = getNewPose(left_dx, left_dy, right_dx, right_dy, speed)

    ## For incremental tests
    limb_left.set_joint_positions(joint_values_left)
    limb_right.set_joint_positions(joint_values_right)

    curr_left = limb_left.endpoint_pose()['position']
    print "\nnow LEFT ( x:", round(curr_left[0],2),', y:', round(curr_left[1],2),', z:', round(curr_left[2],2),')'
    curr_right = limb_right.endpoint_pose()['position']
    print "now RIGHT ( x:", round(curr_right[0],2),', y:', round(curr_right[1],2),', z:', round(curr_right[2],2),')'

    # # Execute motion and track progress
    # while not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) and \
    #     not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS)):
    #     # send joint commands
    #     limb_left.set_joint_positions(joint_values_left)
    #     limb_right.set_joint_positions(joint_values_right)
    #     # save joint movements
    #     # puck_speeds.append(getPuckSpeed(puck_positions))

    return 1

#####################################################################
rospy.init_node('HCK_PC_main_node')
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

#v4 - 35,20
# initial_left = {'left_w0': -2.89577223233069, 'left_w1': -1.1600729708383442, 'left_w2': -1.6582332317041322, 'left_e0': -1.448461358960802, 'left_e1': 0.7669903939427068, 'left_s0': 0.27113110425874687, 'left_s1': -0.7060146576242616}
# # initial_right = {'right_s0': 0.4415068194922188, 'right_s1': 0.947, 'right_w0': -2.742287838168007, 'right_w1': 0.9780381055677475, 'right_w2': 1.1313534973875081, 'right_e0': 1.881106510746034, 'right_e1': 1.9022031896138627}
# initial_right = {'right_s0': 0.800466048527356, 'right_s1': 0.3491576638939382, 'right_w0': -2.959, 'right_w1': 1.5025484131118265, 'right_w2': 2.935518625428416, 'right_e0': 1.559681467805579, 'right_e1': 1.7002814644295619}

# #v3 - 30, 5
# initial_left = {'left_w0': -2.6461168591023387, 'left_w1': -0.9595049828223263, 'left_w2': -0.5668059011236604, 'left_e0': -1.9493060862053895, 'left_e1': 1.2202817167628466, 'left_s0': 0.9437816797465008, 'left_s1': -0.08321845774278369}
# initial_right = {'right_s0': 1.032759546843563, 'right_s1': 0.4353110245074654, 'right_w0': -2.248856247084555, 'right_w1': 1.018974816124536, 'right_w2': 1.9044125747349738, 'right_e0': 1.8281707804907916, 'right_e1': 1.3464587499877072}

# v5 38, 16
initial_left = {'left_w0': -2.8091023178151637, 'left_w1': -1.088359369004701, 'left_w2': -1.3874856226423566, 'left_e0': -1.3798157187029296, 'left_e1': 0.9993884833073471, 'left_s0': 0.5702573578964025, 'left_s1': -0.705247667230319}
initial_right = {'right_s0': 0.877122385358828, 'right_s1': 0.38881368501200375, 'right_w0': -2.959, 'right_w1': 1.1286424474647607, 'right_w2': 2.932984543928958, 'right_e0': 1.4928188728981575, 'right_e1': 1.5984069969052348}


limb_left.move_to_joint_positions(initial_left, timeout=5)
limb_right.move_to_joint_positions(initial_right, timeout=5)
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'System is ready!'\"") 
# time.sleep(5)

init_left = limb_left.endpoint_pose()['position']
print "\nINIT LEFT ( x:", round(init_left[0],2),', y:', round(init_left[1],2),', z:', round(init_left[2],2),')'
init_right = limb_right.endpoint_pose()['position']
print "INIT RIGHT ( x:", round(init_right[0],2),', y:', round(init_right[1],2),', z:', round(init_right[2],2),')'
print "-"*50




raw_input("Execute trial?")



while not rospy.is_shutdown():
    # (left_dx, left_dy, right_dx, right_dy, speed_left=0.3, speed_right=0.3)

    executeTrial(0, 0, 0.1 ,0)

    # executeTrial(-0.1, 0, 0 ,0)

    # raw_input("Execute trial?")
    # s = 0.5
    # executeTrial(-0.4,0, 0.3,0, s)


rospy.on_shutdown(cleanup_on_shutdown)
# rate.sleep()
rospy.spin()
