
# Import stuff
import time
import rospy
import pickle
import numpy as np
import matplotlib.pyplot as pl

import ik_solver
import baxter_interface as BI

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

LEFT_X_MIN = -0.4
LEFT_X_MAX = 0.4

LEFT_Y_MIN = -0.4
LEFT_Y_MAX = 0.4

RIGHT_X_MIN = -0.4
RIGHT_X_MAX = 0.4

RIGHT_Y_MIN = -0.4
RIGHT_Y_MAX = 0.4


### INITIAL POSES
# INITIAL POSE v3 - best conf
initial_left = {'left_w0': -2.6461168591023387, 'left_w1': -0.9595049828223263, 'left_w2': -0.5668059011236604, 'left_e0': -1.9493060862053895, 'left_e1': 1.2202817167628466, 'left_s0': 0.9437816797465008, 'left_s1': -0.08321845774278369}
initial_right = {'right_s0': 0.9594823450680383, 'right_s1': 0.29484998388802847, 'right_w0': -2.098514074519654, 'right_w1': 0.8602590333104184, 'right_w2': 1.9667979410452, 'right_e0': 1.6474396266164186, 'right_e1': 1.4020464458749478}

# # INITIAL POSE v2
# initial_left = {'left_w0': 0.6895243641544935, 'left_w1': 1.864170152477749, 'left_w2': 2.879665434057893, 'left_e0': -1.7299468335377755, 'left_e1': 1.3936215457938983, 'left_s0': 0.5211699726840693, 'left_s1': -0.9012137128826806}
# initial_right = {'right_s0': 0.37964739807565584, 'right_s1': 0.14352143114051552, 'right_w0': -1.8527256523443292, 'right_w1': 1.420931037716502, 'right_w2': 1.685053641346243, 'right_e0': 1.4377696804678948, 'right_e1': 2.0279224340086435}

# INITIAL POSE v1 
# initial_left = {'left_w0': 0.642354454927017, 'left_w1': 1.3690778531877317, 'left_w2': 3.01197127701301, 'left_e0': -1.4871943738549087, 'left_e1': 1.259781722050896, 'left_s0': 0.9698593531405528, 'left_s1': -1.1792477306869118}
# initial_right = {'right_s0': 1.1888384467948654, 'right_s1': 0.10362679558375552, 'right_w0': -1.754988626921679, 'right_w1': 1.4634793712939942, 'right_w2': 1.734155015491007, 'right_e0': 1.6475973989727597, 'right_e1': 1.1819367115504478}

# Some old one...
# initial_left = {'left_w0': 0.8590292412158317, 'left_w1': 0.7850146682003605, 'left_w2': 2.8324955248304167, 'left_e0': -1.831573060735184, 'left_e1': 1.4860438882639946, 'left_s0': 1.0530778108833365, 'left_s1': -0.5840631849873713}
# initial_right = {'right_s0': 0.8942580596047225, 'right_s1': 0.5936894423811188, 'right_w0': -2.0064777590922307, 'right_w1': 1.1135462654472854, 'right_w2': 1.1819992394524164, 'right_e0': 1.9901836920417224, 'right_e1': 1.4520039296340554}

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


def executeTrial(left_dx, left_dy, right_dx, right_dy, speed=0.3):  
    joint_values_left, joint_values_right, new_pos_left, new_pos_right = getNewPose(left_dx, left_dy, right_dx, right_dy, speed)

    ### For incremental tests
    # limb_left.set_joint_positions(joint_values_left)
    # limb_right.set_joint_positions(joint_values_right)

    # curr_left = limb_left.endpoint_pose()['position']
    # print "\nnow LEFT ( x:", round(curr_left[0],2),', y:', round(curr_left[1],2),', z:', round(curr_left[2],2),')'
    # curr_right = limb_right.endpoint_pose()['position']
    # print "now RIGHT ( x:", round(curr_right[0],2),', y:', round(curr_right[1],2),', z:', round(curr_right[2],2),')'

    # Execute motion and track progress
    while not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) and \
        not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS)):
        # send joint commands
        limb_left.set_joint_positions(joint_values_left)
        limb_right.set_joint_positions(joint_values_right)
        # save joint movements
        # puck_speeds.append(getPuckSpeed(puck_positions))

    return 1

# # Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

# # Taken from intro to bayes opt
# def kernel(a, b):
#     """ GP squared exponential kernel """
#     kernelParameter = 1
#     sqdist = (1/kernelParameter) * np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#     return (1+np.sqrt(5*sqdist)+5*sqdist/3.) * np.exp(-np.sqrt(5*sqdist))

#####################################################################
# ROS initialisation
rospy.init_node('HCK_PC_main_node')
rate = rospy.Rate(100)
# Baxter initialisation
if not BI.RobotEnable().state().enabled:
    print("Enabling robot... ")
    BI.RobotEnable().enable()

limb_left = BI.Limb("left")
limb_right = BI.Limb("right")
# limb_left.set_joint_position_speed(0.3)
# limb_right.set_joint_position_speed(0.3)

#####################################################################
#####################################################################
# Main loop
#####################################################################

# limb_left.move_to_joint_positions(initial_left, timeout=3)
# limb_right.move_to_joint_positions(initial_right, timeout=3)
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'System is ready!'\"") 
time.sleep(5)
print "-"*50


# Create distribution with ONE TRAINING POINT!
N = 1
# Import stuff
import time
import rospy
import pickle
import numpy as np
import matplotlib.pyplot as pl

import ik_solver
import baxter_interface as BI

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

LEFT_X_MIN = -0.4
LEFT_X_MAX = 0.4

LEFT_Y_MIN = -0.4
LEFT_Y_MAX = 0.4

RIGHT_X_MIN = -0.4
RIGHT_X_MAX = 0.4

RIGHT_Y_MIN = -0.4
RIGHT_Y_MAX = 0.4


### INITIAL POSES
# INITIAL POSE v3 - best conf
initial_left = {'left_w0': -2.6461168591023387, 'left_w1': -0.9595049828223263, 'left_w2': -0.5668059011236604, 'left_e0': -1.9493060862053895, 'left_e1': 1.2202817167628466, 'left_s0': 0.9437816797465008, 'left_s1': -0.08321845774278369}
initial_right = {'right_s0': 0.9594823450680383, 'right_s1': 0.29484998388802847, 'right_w0': -2.098514074519654, 'right_w1': 0.8602590333104184, 'right_w2': 1.9667979410452, 'right_e0': 1.6474396266164186, 'right_e1': 1.4020464458749478}

# # INITIAL POSE v2
# initial_left = {'left_w0': 0.6895243641544935, 'left_w1': 1.864170152477749, 'left_w2': 2.879665434057893, 'left_e0': -1.7299468335377755, 'left_e1': 1.3936215457938983, 'left_s0': 0.5211699726840693, 'left_s1': -0.9012137128826806}
# initial_right = {'right_s0': 0.37964739807565584, 'right_s1': 0.14352143114051552, 'right_w0': -1.8527256523443292, 'right_w1': 1.420931037716502, 'right_w2': 1.685053641346243, 'right_e0': 1.4377696804678948, 'right_e1': 2.0279224340086435}

# INITIAL POSE v1 
# initial_left = {'left_w0': 0.642354454927017, 'left_w1': 1.3690778531877317, 'left_w2': 3.01197127701301, 'left_e0': -1.4871943738549087, 'left_e1': 1.259781722050896, 'left_s0': 0.9698593531405528, 'left_s1': -1.1792477306869118}
# initial_right = {'right_s0': 1.1888384467948654, 'right_s1': 0.10362679558375552, 'right_w0': -1.754988626921679, 'right_w1': 1.4634793712939942, 'right_w2': 1.734155015491007, 'right_e0': 1.6475973989727597, 'right_e1': 1.1819367115504478}

# Some old one...
# initial_left = {'left_w0': 0.8590292412158317, 'left_w1': 0.7850146682003605, 'left_w2': 2.8324955248304167, 'left_e0': -1.831573060735184, 'left_e1': 1.4860438882639946, 'left_s0': 1.0530778108833365, 'left_s1': -0.5840631849873713}
# initial_right = {'right_s0': 0.8942580596047225, 'right_s1': 0.5936894423811188, 'right_w0': -2.0064777590922307, 'right_w1': 1.1135462654472854, 'right_w2': 1.1819992394524164, 'right_e0': 1.9901836920417224, 'right_e1': 1.4520039296340554}

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


def executeTrial(left_dx, left_dy, right_dx, right_dy, speed=0.3):  
    joint_values_left, joint_values_right, new_pos_left, new_pos_right = getNewPose(left_dx, left_dy, right_dx, right_dy, speed)

    ### For incremental tests
    # limb_left.set_joint_positions(joint_values_left)
    # limb_right.set_joint_positions(joint_values_right)

    # curr_left = limb_left.endpoint_pose()['position']
    # print "\nnow LEFT ( x:", round(curr_left[0],2),', y:', round(curr_left[1],2),', z:', round(curr_left[2],2),')'
    # curr_right = limb_right.endpoint_pose()['position']
    # print "now RIGHT ( x:", round(curr_right[0],2),', y:', round(curr_right[1],2),', z:', round(curr_right[2],2),')'

    # Execute motion and track progress
    while not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) and \
        not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS)):
        # send joint commands
        limb_left.set_joint_positions(joint_values_left)
        limb_right.set_joint_positions(joint_values_right)
        # save joint movements
        # puck_speeds.append(getPuckSpeed(puck_positions))

    return 1

# # Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

# # Taken from intro to bayes opt
# def kernel(a, b):
#     """ GP squared exponential kernel """
#     kernelParameter = 1
#     sqdist = (1/kernelParameter) * np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#     return (1+np.sqrt(5*sqdist)+5*sqdist/3.) * np.exp(-np.sqrt(5*sqdist))

#####################################################################
# ROS initialisation
rospy.init_node('HCK_PC_main_node')
rate = rospy.Rate(100)
# Baxter initialisation
if not BI.RobotEnable().state().enabled:
    print("Enabling robot... ")
    BI.RobotEnable().enable()

limb_left = BI.Limb("left")
limb_right = BI.Limb("right")
# limb_left.set_joint_position_speed(0.3)
# limb_right.set_joint_position_speed(0.3)

#####################################################################
#####################################################################
# Main loop
#####################################################################

# limb_left.move_to_joint_positions(initial_left, timeout=3)
# limb_right.move_to_joint_positions(initial_right, timeout=3)
# os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'System is ready!'\"") 
time.sleep(5)
print "-"*50


# Create distribution with ONE TRAINING POINT!
N = 1
n = 100         # number of test points.
s = 0.00005    # noise variance.
Xtrain = np.random.uniform(SPEED_MIN, SPEED_MAX, size=(N,1)).reshape(-1,1)
y = np.empty([0,1])

Xtest = np.linspace(SPEED_MIN, SPEED_MAX, n).reshape(-1,1)
Kss = kernel(Xtest, Xtest)


while not rospy.is_shutdown():

    for i in range(5):
        limb_left.move_to_joint_positions(initial_left, timeout=3)
        limb_right.move_to_joint_positions(initial_right, timeout=3)
        print "\nStep", i+1
        raw_input("Ready to execute speed: "+str(Xtrain[-1])+"?")
        # time.sleep(15)
        # (left_dx, left_dy, right_dx, right_dy, speed)
        executeTrial(-0.1,0, 0.2,0, Xtrain[-1])
        y = np.append(y, np.array(input('Enter distance: ')))

        # get posterior MU and SIGMA
        K = kernel(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + s*np.eye(N))
        Ks = kernel(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # get posterior MU and SIGMA
        mu = np.dot(Lk.T, np.linalg.solve(L, y))
        var_post = np.sqrt(np.diag(Kss) - np.sum(Lk**2, axis=0))
        # SAMPLE a NEW POINT based on HIGHEST UNCERTAINTY and add it to Xtrain
        # make it uniform from the list
        new_point = Xtest[np.argmax(var_post)].reshape(-1,1)
        Xtrain = np.append(Xtrain, new_point, axis=0)
        print '\n'
        print Xtrain, y
        # PLOT stuff
        # pl.figure(1)
        pl.clf()
        pl.plot(Xtrain[:-1], y, 'r+', ms=20)
        pl.gca().fill_between(Xtest.flat, mu-3*var_post, mu+3*var_post, color="#dddddd")
        pl.plot(Xtest, mu, 'r--', lw=2)
        pl.plot(Xtest, var_post, 'k-', lw=2)
        pl.scatter(new_point, np.max(var_post), c='g', s=50)
        pl.title('[Step: '+str(i+1)+'] speed: '+str(round(Xtrain[-2][0],2))+'; distance: '+str(y[-1])+'m')
        pl.axis([SPEED_MIN, SPEED_MAX, -1, 3])
        pl.savefig('vtest_step_'+str(i)+'.png', bbox_inches='tight')
        pl.pause(1)

    with open('vtest_mu.dat', "wb") as f:
            pickle.dump(mu, f)

    print '\nDONE!'

rospy.on_shutdown(cleanup_on_shutdown)
# rate.sleep()
rospy.spin()


# Execute motion to check repeatability

# implement GPR
n = 100         # number of test points.
s = 0.00005    # noise variance.
Xtrain = np.random.uniform(SPEED_MIN, SPEED_MAX, size=(N,1)).reshape(-1,1)
y = np.empty([0,1])

Xtest = np.linspace(SPEED_MIN, SPEED_MAX, n).reshape(-1,1)
Kss = kernel(Xtest, Xtest)


while not rospy.is_shutdown():

    for i in range(5):
        limb_left.move_to_joint_positions(initial_left, timeout=3)
        limb_right.move_to_joint_positions(initial_right, timeout=3)
        print "\nStep", i+1
        raw_input("Ready to execute speed: "+str(Xtrain[-1])+"?")
        # time.sleep(15)
        # (left_dx, left_dy, right_dx, right_dy, speed)
        executeTrial(-0.1,0, 0.2,0, Xtrain[-1])
        y = np.append(y, np.array(input('Enter distance: ')))

        # get posterior MU and SIGMA
        K = kernel(Xtrain, Xtrain)
        L = np.linalg.cholesky(K + s*np.eye(N))
        Ks = kernel(Xtrain, Xtest)
        Lk = np.linalg.solve(L, Ks)
        # get posterior MU and SIGMA
        mu = np.dot(Lk.T, np.linalg.solve(L, y))
        var_post = np.sqrt(np.diag(Kss) - np.sum(Lk**2, axis=0))
        # SAMPLE a NEW POINT based on HIGHEST UNCERTAINTY and add it to Xtrain
        # make it uniform from the list
        new_point = Xtest[np.argmax(var_post)].reshape(-1,1)
        Xtrain = np.append(Xtrain, new_point, axis=0)
        print '\n'
        print Xtrain, y
        # PLOT stuff
        # pl.figure(1)
        pl.clf()
        pl.plot(Xtrain[:-1], y, 'r+', ms=20)
        pl.gca().fill_between(Xtest.flat, mu-3*var_post, mu+3*var_post, color="#dddddd")
        pl.plot(Xtest, mu, 'r--', lw=2)
        pl.plot(Xtest, var_post, 'k-', lw=2)
        pl.scatter(new_point, np.max(var_post), c='g', s=50)
        pl.title('[Step: '+str(i+1)+'] speed: '+str(round(Xtrain[-2][0],2))+'; distance: '+str(y[-1])+'m')
        pl.axis([SPEED_MIN, SPEED_MAX, -1, 3])
        pl.savefig('vtest_step_'+str(i)+'.png', bbox_inches='tight')
        pl.pause(1)

    with open('vtest_mu.dat', "wb") as f:
            pickle.dump(mu, f)

    print '\nDONE!'

rospy.on_shutdown(cleanup_on_shutdown)
# rate.sleep()
rospy.spin()


# Execute motion to check repeatability

# implement GPR