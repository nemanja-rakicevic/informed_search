
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
##########################################################################3333
##########################################################################3333
##########################################################################3333
##########################################################################3333
# implement GPR 2D sample

import itertools
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

f = lambda x: np.sum(np.sin(np.pi*x)/(np.pi*x), axis=1)
f1 = lambda x: np.sum(np.sin(np.pi*x)/(np.pi*x), axis=0)

# Define the kernel
# def kernel(a, b):
#     """ GP squared exponential kernel """
#     kernelParameter = 1
#     sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#     return np.exp(-.5 * (1/kernelParameter) * sqdist)


# Create distribution with ONE TRAINING POINT!
N = 60
n = 100         # number of test points.
s = 0.00005    # noise variance.
Xtrain = np.random.uniform(-5, 5, size=(N,2)).reshape(-1,2)
# Xtest = np.array([np.linspace(-5, 5, n), np.linspace(-5, 5, n)]).T
Xtest = np.array([ss for ss in itertools.product(np.linspace(-5, 5, n),np.linspace(-5, 5, n))])
Kss = kernel(Xtest, Xtest)

# for i in range(10):
y = f(Xtrain).reshape(-1,1)# + s*np.random.randn(Xtrain.shape[0])
# get posterior MU and SIGMA
K = kernel(Xtrain, Xtrain)
L = np.linalg.cholesky(K + s*np.eye(N))
Ks = kernel(Xtrain, Xtest)
Lk = np.linalg.solve(L, Ks)
# get posterior MU and SIGMA
mu = np.dot(Lk.T, np.linalg.solve(L, y))
var_post = np.sqrt(np.diag(Kss) - np.sum(Lk**2, axis=0))
# # SAMPLE a NEW POINT based on HIGHEST UNCERTAINTY and add it to Xtrain
# # make it uniform from the list
# new_point = Xtest[np.argmax(var_post)].reshape(-1,1)
# Xtrain = np.append(Xtrain, new_point, axis=0)
# print Xtrain, mu.shape, y.shape
# # PLOT stuff
# # pl.figure(1)
# pl.clf()
# pl.plot(Xtrain, y, 'r+', ms=20)
# # pl.plot(Xtrain[:-1], y, 'r+', ms=20)
# pl.plot(Xtest, f(Xtest), 'b-')
# pl.gca().fill_between(Xtest[:,0], mu-3*var_post, mu+3*var_post, color="#dddddd")
# pl.plot(Xtest, mu, 'r--', lw=2)
# pl.plot(Xtest, var_post, 'k-', lw=2)
# pl.scatter(new_point, np.max(var_post), c='g')
# # pl.savefig('predictive.png', bbox_inches='tight')
# pl.title('Step:'+str(i))
# pl.axis([-5, 5, -3, 3])
# pl.pause(1)
#     # REPEAT 10 times (ERROR MEASURE???)


fig = pl.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.linspace(-5, 5, n), np.linspace(-5, 5, n))
Z=mu.reshape(100,100)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#
# fig2 = pl.figure()
# ax2 = fig2.gca(projection='3d')
# Ztrue = f1(np.array([X,Y]))
# surf = ax2.plot_surface(X, Y, Ztrue, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# pl.show()
#
fig3 = pl.figure()
ax2 = fig3.gca(projection='3d')
Zvar = var_post.reshape(100,100)
surf = ax2.plot_surface(X, Y, Zvar, cmap=cm.coolwarm,linewidth=0, antialiased=False)
pl.show()


# pl.figure()
# pl.scatter(Xtest[:,0], f(Xtest))
# pl.figure()
# pl.scatter(Xtest[:,1], f(Xtest))
# pl.show()

fig3 = pl.figure()
ax2 = fig3.gca(projection='3d')
X, Y = np.meshgrid(range_l_dx, range_l_dy)
Z = cr.reshape(7,10)
surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
pl.show()


##########################################################################3333
##########################################################################3333
##########################################################################3333
##########################################################################3333
# REAL WORLD EXAMPLE
# implement GPR 2D sample

import itertools
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle

# f = lambda x: np.sum(np.sin(np.pi*x)/(np.pi*x), axis=1)
# f1 = lambda x: np.sum(np.sin(np.pi*x)/(np.pi*x), axis=0)

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

# def kernel(a, b):
#     """ GP Matern 5/2 kernel: """
#     kernelParameter = 1
#     sqdist = (1/kernelParameter) * np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    # return (1+np.sqrt(5*sqdist)+5*sqdist/3.) * np.exp(-np.sqrt(5*sqdist))

# Create distribution with ONE TRAINING POINT!
s = 0.00005    # noise variance.
# N = 60
# n = 100         # number of test points.

# measurements in pixels
x_train = np.array( [[0., 0.], 
                    [140., 0], 
                    [340., 0], 
                    [540., 0], 
                    [740., 0], 
                    [940., 200], 
                    [140., 200], 
                    [340., 200], 
                    [540., 200], 
                    [740., 200], 
                    [940., 200], 
                    [140., 400], 
                    [340., 400], 
                    [540., 400], 
                    [740., 400], 
                    [940., 400]])/100
# distances in meters
y_train = np.array( [[0., 0.], 
                [20., 0], 
                [56., 0], 
                [106., 0], 
                [180., 0], 
                [298.,0],
                [20., 30], 
                [56.5, 36], 
                [107.5, 43], 
                [182., 52], 
                [298., 67],
                [20., 61], 
                [57.5, 70.5], 
                [111., 85], 
                [187., 105], 
                [307., 137]])



# Xtrain = x_train[:,0].reshape(-1,1)/100
Xtrain = x_train
# x -coordinate mapping
y1 = y_train[:,0]
# y -coordinate mapping
y2 = y_train[:,1]

Xtest = np.array([ss for ss in itertools.product(np.linspace(-0.5, 10.80, 200),np.linspace(0, 8, 100))])

# Xtest = np.linspace(0, 10.80, 20).reshape(-1,1)

Kss = kernel(Xtest, Xtest)

# for i in range(10):
# y = f(Xtrain).reshape(-1,1)# + s*np.random.randn(Xtrain.shape[0])
# get posterior MU and SIGMA
K = kernel(Xtrain, Xtrain)
L = np.linalg.cholesky(K + s*np.eye(len(Xtrain)))
Ks = kernel(Xtrain, Xtest)
Lk = np.linalg.solve(L, Ks)
# get posterior MU and SIGMA
mu1 = np.dot(Lk.T, np.linalg.solve(L, y1))
mu2 = np.dot(Lk.T, np.linalg.solve(L, y2))

var_post = np.sqrt(np.diag(Kss) - np.sum(Lk**2, axis=0))




X, Y = np.meshgrid(np.linspace(0, 8.00, 100), np.linspace(-0.5, 10.80, 200))
fig1 = pl.figure()
ax1 = fig1.gca(projection='3d')
Z1 = mu1.reshape(200,100)
surf = ax1.plot_surface(X, Y, Z1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#
X1, Y1 = np.meshgrid(np.linspace(0, 19.20, 200), np.linspace(-0.5, 10.80, 200))
fig2 = pl.figure()
ax2 = fig2.gca(projection='3d')
Z_x = np.hstack((np.fliplr(Z1), Z1))
surf = ax2.plot_surface(X1, Y1, Z_x, cmap=cm.coolwarm,linewidth=0, antialiased=False)
pl.show()
#
#
fig3 = pl.figure()
ax3 = fig3.gca(projection='3d')
Z2 = mu2.reshape(200,100)
surf = ax3.plot_surface(X, Y, Z2, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#
fig4 = pl.figure()
ax4 = fig4.gca(projection='3d')
Z_y = np.hstack((np.fliplr(Z2), -Z2))
surf = ax4.plot_surface(X1, Y1, Z_y, cmap=cm.coolwarm,linewidth=0, antialiased=False)
pl.show()


x_range = np.linspace(-0.5, 10.80, 200)
y_range = np.linspace(0, 19.20, 200)

with open('kinect_mapping.dat', "wb") as f:
    pickle.dump((Z_x, Z_y, x_range, y_range), f)


ball_x = 557
ball_y = 745

px_x = ball_x
px_y = 1080 - ball_y
matx = np.abs(x_range-px_y/100.).argmin()
maty = np.abs(y_range-px_x/100.).argmin()

x_coord = Z_x[matx, maty]
y_coord = Z_y[matx, maty]

print x_coord, y_coord
print matx, maty

pl.plot(x_range, Z_y[:,maty])
pl.show()


pl.plot(Xtest, mu1)
pl.scatter(Xtrain, y1)
pl.show()
