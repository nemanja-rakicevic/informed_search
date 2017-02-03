
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

#v4 - 35,20
# initial_left = {'left_w0': -2.89577223233069, 'left_w1': -1.1600729708383442, 'left_w2': -1.6582332317041322, 'left_e0': -1.448461358960802, 'left_e1': 0.7669903939427068, 'left_s0': 0.27113110425874687, 'left_s1': -0.7060146576242616}
# # initial_right = {'right_s0': 0.4415068194922188, 'right_s1': 0.947, 'right_w0': -2.742287838168007, 'right_w1': 0.9780381055677475, 'right_w2': 1.1313534973875081, 'right_e0': 1.881106510746034, 'right_e1': 1.9022031896138627}
# initial_right = {'right_s0': 0.800466048527356, 'right_s1': 0.3491576638939382, 'right_w0': -2.959, 'right_w1': 1.5025484131118265, 'right_w2': 2.935518625428416, 'right_e0': 1.559681467805579, 'right_e1': 1.7002814644295619}

# #v3 - 30, 5
# initial_left = {'left_w0': -2.6461168591023387, 'left_w1': -0.9595049828223263, 'left_w2': -0.5668059011236604, 'left_e0': -1.9493060862053895, 'left_e1': 1.2202817167628466, 'left_s0': 0.9437816797465008, 'left_s1': -0.08321845774278369}
# initial_right = {'right_s0': 1.032759546843563, 'right_s1': 0.4353110245074654, 'right_w0': -2.248856247084555, 'right_w1': 1.018974816124536, 'right_w2': 1.9044125747349738, 'right_e0': 1.8281707804907916, 'right_e1': 1.3464587499877072}

# # v5 38, 16
# initial_left = {'left_w0': -2.8091023178151637, 'left_w1': -1.088359369004701, 'left_w2': -1.3874856226423566, 'left_e0': -1.3798157187029296, 'left_e1': 0.9993884833073471, 'left_s0': 0.5702573578964025, 'left_s1': -0.705247667230319}
# initial_right = {'right_s0': 0.877122385358828, 'right_s1': 0.38881368501200375, 'right_w0': -2.959, 'right_w1': 1.1286424474647607, 'right_w2': 2.932984543928958, 'right_e0': 1.4928188728981575, 'right_e1': 1.5984069969052348}

# # v6, bigger angle span
# initial_left = {'left_w0': 1.0987137393229276, 'left_w1': 1.0270001374892845, 'left_w2': -2.8604906742093252, 'left_e0': -2.3324177879797716, 'left_e1': 1.7422186798408588, 'left_s0': 0.4452379236837413, 'left_s1': 0.9322768238373602}
# initial_right = {'right_s0': -0.24191704699435934, 'right_s1': 0.9320225083599076, 'right_w0': -2.634182591662538, 'right_w1': 0.9931456535434887, 'right_w2': 1.1775742356463146, 'right_e0': 1.3751312891539729, 'right_e1': 1.9915054135984016}


# # v6B, bigger angle span
# initial_left = {'left_w0': 0.6576942628058712, 'left_w1': 1.737233242280231, 'left_w2': -2.292917782691722, 'left_e0': -2.3776702212223912, 'left_e1': 1.609145846491799, 'left_s0': 0.2933738256830854, 'left_s1': -0.19941750242510378}
# initial_right = {'right_s0': -0.5409801757359759, 'right_s1': 0.947, 'right_w0': -2.163408540995058, 'right_w1': 1.3872383604059964, 'right_w2': 0.9530894510188792, 'right_e0': 1.5557911285801003, 'right_e1': 2.2906723192741483}

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
# CONSTANTS - left arm
LEFT_X_MIN = -0.35
LEFT_X_MAX = 0.12
LEFT_Y_MIN = -0.8
LEFT_Y_MAX = 0.27
# CONSTANTS - right arm
RIGHT_X_MIN = 0.0
RIGHT_X_MAX = 0.17
RIGHT_Y_MIN = -0.5
RIGHT_Y_MAX = 0.5
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
    # print joint_values_left
    # print joint_values_right
    # Set joint speed
    left_dL = sqdist(left_dx,left_dy)
    right_dL = sqdist(right_dx,right_dy) 
    if left_dL>=right_dL:
        speed_left = speed
        try:
            speed_right = speed*right_dL/left_dL
        except:
            speed_right = 0
    else:
        speed_right = speed
        speed_left = speed*left_dL/right_dL
    # print left_dL, right_dL
    # print speed_left, speed_right, speed
    # limb_left.set_joint_position_speed(1)
    # limb_right.set_joint_position_speed(1)
    limb_left.set_joint_position_speed(speed_left)
    limb_right.set_joint_position_speed(speed_right)

    return joint_values_left, joint_values_right, new_pos_left, new_pos_right


def executeTest(left_dx, left_dy, right_dx, right_dy, w=0, speed=0.7):  
    # Calculate joint angles
    joint_values_left, joint_values_right, new_pos_left, new_pos_right = getNewPose(left_dx, left_dy, right_dx, right_dy, speed)
    joint_values_left['left_w2'] = w
    # print joint_values_left
    # print limb_left.joint_angles()
    ## For incremental tests
    limb_left.set_joint_positions(joint_values_left)
    limb_right.set_joint_positions(joint_values_right)
    # Print progress
    curr_left = limb_left.endpoint_pose()['position']
    print "\nnow LEFT ( x:", round(curr_left[0],2),', y:', round(curr_left[1],2),', z:', round(curr_left[2],2),')'
    curr_right = limb_right.endpoint_pose()['position']
    print "now RIGHT ( x:", round(curr_right[0],2),', y:', round(curr_right[1],2),', z:', round(curr_right[2],2),')'

    return 1


def executeTrial(left_dx, left_dy, right_dx, right_dy, w=0, speed=0.5):  
    # Set tip hit angle
    angle_left = limb_left.joint_angles()
    angle_left['left_w2'] = w
    limb_left.set_joint_position_speed(1)
    limb_left.move_to_joint_positions(angle_left, timeout=1)
    # Calculate joint angles
    joint_values_left, joint_values_right, new_pos_left, new_pos_right = getNewPose(left_dx, left_dy, right_dx, right_dy, speed)
    # joint_values_left['left_w2'] = w
    # Execute motion and track progress
    # while (not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) or \
    #     not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS))) and cnt <100000:
    cnt = 0
    while (not (tuple(np.array(joint_values_left.values())-THRSH_POS) <= tuple(limb_left.joint_angles().values()) <= tuple(np.array(joint_values_left.values())+THRSH_POS)) or \
        not (tuple(np.array(joint_values_right.values())-THRSH_POS) <= tuple(limb_right.joint_angles().values()) <= tuple(np.array(joint_values_right.values())+THRSH_POS))) and cnt <50000:
        cnt+=1
        # send joint commands
        limb_left.set_joint_positions(joint_values_left)
        limb_right.set_joint_positions(joint_values_right)

    raw_input("Done?")

    return 1


def executeTrial_separate(left_dx, left_dy, right_dx, right_dy, w=0, speed=0.9):  
    # Set tip hit angle
    angle_left = limb_left.joint_angles()
    angle_left['left_w2'] = w
    limb_left.set_joint_position_speed(1)
    limb_left.move_to_joint_positions(angle_left, timeout=1)
    # Calculate joint angles
    joint_values_left, joint_values_right, new_pos_left, new_pos_right = getNewPose(left_dx, left_dy, right_dx, right_dy, speed)
    # Execute motion and track progress
    cnt = 0
    while not (tuple(np.array(joint_values_left.values())-THRSH_POS) <= tuple(limb_left.joint_angles().values()) <= tuple(np.array(joint_values_left.values())+THRSH_POS)) and cnt <100000:
        limb_left.set_joint_positions(joint_values_left)
        cnt+=1
        # print 'LEFT',cnt
    cnt=0
    while not (tuple(np.array(joint_values_right.values())-THRSH_POS) <= tuple(limb_right.joint_angles().values()) <= tuple(np.array(joint_values_right.values())+THRSH_POS)) and cnt <100000:
        limb_right.set_joint_positions(joint_values_right)
        cnt+=1
        # print 'RIGHT',cnt
    raw_input("Done?")

    return 1



def executeForces(left_dx, left_dy, right_dx, right_dy, w=0, speed=0.5):  
   # Calculate joint angles
    joint_values_left, joint_values_right, new_pos_left, new_pos_right = getNewPose(left_dx, left_dy, right_dx, right_dy, speed)

    default_spring = np.array([10.0, 15.0, 5.0, 5.0, 3.0, 2.0, 1.5])/10.
    # default_damping = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    # default_spring = (1., 1., 1., 1., 1., 1., 1.)
    default_damping = np.array([10.0, 15.0, 5.0, 5.0, 3.0, 2.0, 1.5])/10.

    curr_left = limb_left.endpoint_pose()['position']
    print "\nnow LEFT ( x:", round(curr_left[0],2),', y:', round(curr_left[1],2),', z:', round(curr_left[2],2),')'
    curr_right = limb_right.endpoint_pose()['position']
    print "now RIGHT ( x:", round(curr_right[0],2),', y:', round(curr_right[1],2),', z:', round(curr_right[2],2),')'

    # Execute motion and track progress
    cnt=0
    while (not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) or \
        not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS))) and cnt <20000:     
        cnt+=1
        print cnt
        # create our command dict
        torques_left = dict()
        torques_right = dict()
        # record current angles/velocities
        cur_pos_left = limb_left.joint_angles()
        cur_pos_right = limb_right.joint_angles()
        cur_vel_left = limb_left.joint_velocities()
        cur_vel_right = limb_right.joint_velocities()

        # calculate desired forces, using spring and damping
        for idx, joint in enumerate(cur_pos_left.keys()):
            torques_left[joint] = default_spring[idx] * (joint_values_left[joint] - cur_pos_left[joint])
            torques_left[joint] -= default_damping[idx] * cur_vel_left[joint]
        for idx, joint in enumerate(cur_pos_right.keys()):
            torques_right[joint] = default_spring[idx] * (joint_values_right[joint] - cur_pos_right[joint])
            torques_right[joint] -= default_damping[idx] * cur_vel_right[joint]
            print np.sum(joint_values_right[joint] - cur_pos_right[joint])
        # send joint commands
        # limb_left.set_joint_torques(torques_left)
        limb_right.set_joint_torques(torques_right)
        # save joint movements
        # puck_speeds.append(getPuckSpeed(puck_positions))
    raw_input("Done?")

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

# # v6C, bigger angle span
initial_left = {'left_w0': -1.9017526817809416, 'left_w1': -1.5098205904762185, 'left_w2': 0.0, 'left_e0': -2.111141059327301, 'left_e1': 1.5926555530220308, 'left_s0': 0.3861796633501529, 'left_s1': 0.25349032519806464}
# initial_right = {'right_s0': -0.37155194764034827, 'right_s1': 0.947, 'right_w0': -1.9226134104693988, 'right_w1': 1.2720872783823325, 'right_w2': 0.9336995649992517, 'right_e0': 1.4905379406793826, 'right_e1': 1.8962337509333822}
initial_right = {'right_s0': -0.18604712804257897, 'right_s1': 0.7366307125790943, 'right_w0': -1.6419219445615647, 'right_w1': 1.4053479234536634, 'right_w2': 1.0948321182211216, 'right_e0': 1.5301695974547347, 'right_e1': 1.8269783292751778}



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
time.sleep(2)

while not rospy.is_shutdown():

    # (left_dx, left_dy, right_dx, right_dy, w, speed_right=0.3)

    ### TEST RANGE
    # executeTest(0, 0.1, 0 ,0)    # LEFT Y
    # executeTest(0.1, 0, 0 ,0)       # LEFT X
    # executeTest(0, 0 , 0, 0.1)    # RIGHT Y
    # executeTest(0, 0, 0.1 , 0)    # RIGHT X
    # executeTest(-0.35 , 0.3, 0., 0., w=-0.97)    # LEFT WRIST


    ### TEST SWING
    # # raw_input("Execute trial?")
    WRIST_MIN = -0.97    #(max = -3.) lean front
    WRIST_MAX = 0.4     #(max = +3.) lean back
    s = 1
    # executeTrial(-0.3, 0.1, 0.05, 0.4, w=-0.97, speed=s)  # left angle
    # executeTrial(0.1, -0.1, 0.2, 1.2, w=0.3, speed=s)   # straight angle
    # executeTrial_separate(0.25, -0.2, 0.25, -0.10, w=0.45, speed=s)   # straight angle

    ### TEST FORCES
    # executeForces(0.0, 0.0, 0.3, -0.25)   # left angle


rospy.on_shutdown(cleanup_on_shutdown)
# rate.sleep()
rospy.spin()
