
"""
Author:         Nemanja Rakicevic
Date:           November 2017
Description:
                                Classes for creating and running experiments in MuJoCo and on 
                                the real robot.
"""

import time
import pickle
import logging
import numpy as np 

logger = logging.getLogger(__name__)



class SimulationExperiment(object):
    import envs
    import gym

    def __init__(self, environment, resolution, 
                       animation_steps=100, verbose=False, display=False, 
                       **kwargs):

        if environment=='sim2link':
            self.env = self.gym.make('ReacherOneShot-v0', resolution=resolution)
        elif environment=='sim5link':
            self.env = self.gym.make('ReacherOneShot-v1', resolution=resolution)
        
        self.parameter_list = self.env.unwrapped.parameter_list
        self._num_links = len(self.parameter_list)
        self.verbose = verbose
        self.display = display

        self.type = 'SIMULATION'
        self.info_list = []
        self._num_steps = animation_steps




    def _log_trial(self, fail_status, ball_polar, target_dist, **kwargs):
        if self.verbose:
            error_string = ''
            if fail_status:
                outcome_string = "FAIL ({})".format(fail_status)
            else:
                outcome_string = "SUCCESS\t> ball_polar: {}".format(ball_polar)
                if test_params is not None:
                    error_string = "; test error: {}".format(target_dist)
            logger.info("--- trial executed: {}{}{}".format(outcome_string,
                                                            error_string))


    def save_trial_data(self, trial_dirname):  
        self.env.close()
        with open(trial_dirname + "/data_training_info.dat", "wb") as m:
                pickle.dump(self.info_list, m, protocol=pickle.HIGHEST_PROTOCOL)



    def execute_trial(self, param_coords, param_vals, 
                      num_trial=None, test_params=None):
        """ Execute a trial defined by parameters """
        # Set up sequence of intermediate positions
        param_seq = np.array([np.linspace(0, p, self._num_steps) \
                                                     for p in param_vals]).T
        # Place target for testing or just hide it
        if test_params is not None:
            self.env.unwrapped.init_qpos[-2] = \
                    -test_params[1] * np.sin(np.deg2rad(test_params[0])) / 100.
            self.env.unwrapped.init_qpos[-1] = \
                    test_params[1] * np.cos(np.deg2rad(test_params[0])) / 100.
        else:
            self.env.unwrapped.init_qpos[-2] = 0.0
            self.env.unwrapped.init_qpos[-1] = 1.3

        # Execute trial
        init_pos = self.env.reset()
        init_pos = init_pos[:self._num_links]
        obs_list = []
        for i in range(self._num_steps):
            if self.display and not isinstance(test_params, bool):
                self.env.render()
            control = init_pos + param_seq[i]
            observation, _, done, info_dict = self.env.step(control) 
            obs_list.append(observation)
            # Check collision
            if done:
                fail_status = 1
                break
        if self.display and test_params is not None:
            self.env.close()

        # Check ball movement and calculate polar coords
        ball_xy = info_dict['ball_xy']
        if np.linalg.norm(ball_xy) > 0:
            fail_status = 0  
            ball_polar = np.array([np.rad2deg(np.arctan2(-ball_xy[0], 
                                                          ball_xy[1])), 
                                              np.linalg.norm(ball_xy) * 100]) 
        else:
            fail_status = 2  
            ball_polar = np.array([0,0])  

        # Compile trial info
        trial_info = {'trial_num': num_trial,
                      'parameters': param_vals,
                      'coordinates': param_coords,
                      'fail_status': fail_status, 
                      'ball_polar': ball_polar,
                      'target_dist': observation[-1],
                      'observations': np.vstack(obs_list)}
        self._log_trial(**trial_info)

        return trial_info
























##################################################################
#  FINISH THIS AND test_params ON ROBOT
##################################################################
class RobotExperiment:

    # import rospy
    # import baxter_interface as BI
    # CONSTANTS - speed 
    SPEED_MIN = 0.5
    SPEED_MAX = 1
    # CONSTANTS - left arm
    LEFT_X_MIN = -0.3   #(-0.35)
    LEFT_X_MAX = 0.1    #(0.12)
    LEFT_Y_MIN = -0.1   #(-0.8)
    LEFT_Y_MAX = 0.1   #(0.30)
    # CONSTANTS - left wrist
    WRIST_MIN = -0.97   #(max = -3.) lean front
    WRIST_MAX = 0.4     #(max = +3.) lean back
    # CONSTANTS - right arm
    RIGHT_X_MIN = 0.0   #(-0.05)
    RIGHT_X_MAX = 0.17  #(0.20)
    RIGHT_Y_MIN = -0.1  #(-0.5)
    RIGHT_Y_MAX = 0.5   #(0.5)
    ##################################################################
    ## max length of combination vector should be 25000 - 8/7/8/7/8
    # # ### FULL MOTION SPACE
    __range_l_dx  = np.round(np.linspace(LEFT_X_MIN, LEFT_X_MAX, 5), 3)
    __range_l_dy  = np.round(np.linspace(LEFT_Y_MIN, LEFT_Y_MAX, 5), 3)
    __range_r_dx  = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 5), 3)
    __range_r_dy  = np.round(np.linspace(RIGHT_Y_MIN, RIGHT_Y_MAX, 5), 3)
    __range_wrist = np.round(np.linspace(WRIST_MIN, WRIST_MAX, 6), 3)
    __range_speed = np.round(np.linspace(SPEED_MIN, SPEED_MAX, 5), 3)
    ################################################################(-0.3, 0.1, 0.05, 0.4, w=-0.97, speed=s) #(-0.1,0, 0.2,0, s)
    # # ### PARTIAL JOINT SPACE
    # __range_l_dx = np.round(np.linspace(-0.3, -0.3, 1), 3)
    # __range_l_dy = np.round(np.linspace(0.1, 0.1, 1), 3)
    # __range_r_dx = np.round(np.linspace(RIGHT_X_MIN, RIGHT_X_MAX, 5), 3)
    # __range_r_dy = np.round(np.linspace(0.4, 0.4, 1), 3)
    # __range_wrist = np.round(np.linspace(WRIST_MIN, WRIST_MAX, 6), 3)
    # __range_speed = np.round(np.linspace(1, 1, 1), 3)
    # ##################################################################

    def __init__(self):
        self.parameter_list = np.array([self.__range_l_dx, self.__range_l_dy, self.__range_r_dx, self.__range_r_dy, self.__range_wrist, self.__range_speed])
        self.type = 'ROBOT'
        self.info_list = []


    # def executeTrial_ROBOT(params):  
    #     # CHECK 1) Stick constraints
    #     if checkStickConstraints(*params):
    #         print '>> FAILED (Error 1: Stick constraints)'   
    #         print "Repeating...\n"
    #         return 0
    #     print "> TRIAL CHECK 1): OK Stick constraint"

    #     # GET IK SOLUTION
    #     joint_values_left, joint_values_right, speed_left, speed_right, new_pos_left, new_pos_right = getNewPose(*params) 
    #     # joint_values_left['left_w2'] = params[4]
    #     # CHECK 2) Inverse Kinematic solution
    #     if joint_values_left == -1 or joint_values_right == -1:
    #         print '>> TRIAL # - FAILED (Error 2: No IK solution)'
    #         print "Repeating...\n"
    #         return 0
    #     print "> TRIAL CHECK 2): OK IK solution"

    #     # Passed constraint check ready to execute
    #     raw_input(">>> Ready to execute configuration: "+str((params))+"?\n")
    #     # os.system("ssh petar@192.168.0.2 \"espeak -v fr -s 95 'Stand clear'\"")   
    #     time.sleep(1)
    #     # EXECUTE MOTION
    #     # Set tip hit angle
    #     angle_left = limb_left.joint_angles()
    #     angle_left['left_w2'] = params[4]
    #     limb_left.set_joint_position_speed(1)
    #     limb_left.move_to_joint_positions(angle_left, timeout=2)
    #     # Set the speeds
    #     limb_left.set_joint_position_speed(speed_left)
    #     limb_right.set_joint_position_speed(speed_right)
    #     #
    #     # joint_values_left['left_w2'] = params[4]
    #     # EXECUTE MOTION and save/track progress
    #     # while not (tuple(np.asarray(new_pos_left)-THRSH_POS) <= tuple(limb_left.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_left)+THRSH_POS)) and \
    #     #     not (tuple(np.asarray(new_pos_right)-THRSH_POS) <= tuple(limb_right.endpoint_pose()['position']) <= tuple(np.asarray(new_pos_right)+THRSH_POS)):
    #     cnt = 0
    #     while (not (tuple(np.array(joint_values_left.values())-THRSH_POS) <= tuple(limb_left.joint_angles().values()) <= tuple(np.array(joint_values_left.values())+THRSH_POS)) or \
    #         not (tuple(np.array(joint_values_right.values())-THRSH_POS) <= tuple(limb_right.joint_angles().values()) <= tuple(np.array(joint_values_right.values())+THRSH_POS))) and cnt <30000:
    #         cnt+=1
    #         # send joint commands
    #         limb_left.set_joint_positions(joint_values_left)
    #         limb_right.set_joint_positions(joint_values_right)

    #     return 1




    def executeTrial(num_trial, params):
        theta_list = np.array([np.linspace(0, params[0], NUM_STEPS), np.linspace(0, params[1], NUM_STEPS)]).T
        init_pos = env.reset()
        init_pos = init_pos[:2]
        # EXECUTE TRIAL
        contact_cnt = 0
        obs_list = []
        for i in range(NUM_STEPS):
            env.render()
            control = init_pos + theta_list[i]
            observation, _, _, _ = env.step(control) 
            obs_list.append(observation)
            ball_xy = observation[2:4]
            # Check collision
            if env.unwrapped.data.ncon:
                contact_cnt+=1
                if contact_cnt > 5 and not sum(ball_xy)>0:
                    # contact with wall
                    fail_status = 1
                    break
        # Check ball movement and calculate polar coords
        if not sum(ball_xy)>0.:
            # ball did not move
            fail_status = 2  
            ball_polar = np.array([0,0])   
        else:
            fail_status = 0  
            ball_polar = np.array([ np.rad2deg(np.arctan2(-ball_xy[0], ball_xy[1])), np.sqrt(ball_xy[0]**2 + ball_xy[1]**2) * 100])
        # Compile trial info
        all_info = {'trial_num':  num_trial+1,
                    'parameters':   params,
                    'coordinates':  coords,
                    'fail':     fail_status, 
                    'ball_polar':   ball_polar,
                    'observations': np.array(obs_list) }
        # INFO
        if fail_status:
            print("--- trial executed: FAIL ({})".format(fail_status))
        else:
            print("--- trial executed: SUCCESS -> labels: {}".format(ball_polar))
        return all_info


    # def checkStickConstraints(left_dx, left_dy, right_dx, right_dy, *_k):
 #    # Check if the parameters comply with stick dimension constraints  
 #    tmp_left = limb_left.endpoint_pose()['position']
 #    tmp_right = limb_right.endpoint_pose()['position']
 #    dx = abs((tmp_left.x + left_dx) - (tmp_right.x + right_dx))
 #    dy = abs((tmp_left.y + left_dy) - (tmp_right.y + right_dy))
 #    # print "/// STICK ", dx < STICK_X_MAX, dx < STICK_X_MAX and STICK_Y_MIN < dy < STICK_Y_MAX
 #    # print "/// STICK ", round(dx,2), round(dy,2)
 #    if dx <= STICK_X_MAX and dy <= STICK_Y_MAX:
 #         # abs(left_dx)>10*THRSH_POS and abs(left_dy)>10*THRSH_POS and\
 #         # abs(right_dx)>10*THRSH_POS and abs(right_dy)>10*THRSH_POS:
 #        return False
 #    else:
 #        return True



    # def getNewPose(left_dx, left_dy, right_dx, right_dy, w, speed):   
    #     # Get current position
    #     pose_tmp_left = limb_left.endpoint_pose()
    #     pose_tmp_right = limb_right.endpoint_pose()
    #     # Set new position
    #     new_pos_left = limb_left.Point( 
    #         x = pose_tmp_left['position'].x + left_dx, 
    #         y = pose_tmp_left['position'].y + left_dy, 
    #         z = pose_tmp_left['position'].z ) 
    #     new_pos_right = limb_right.Point( 
    #         x = pose_tmp_right['position'].x + right_dx, 
    #         y = pose_tmp_right['position'].y + right_dy, 
    #         z = pose_tmp_right['position'].z ) 
    #     # Get Joint positions
    #     joint_values_left = ik_solver.ik_solve('left', new_pos_left, pose_tmp_left['orientation'], limb_left.joint_angles())
    #     joint_values_right = ik_solver.ik_solve('right', new_pos_right, pose_tmp_right['orientation'], limb_right.joint_angles()) 
    #     # Set joint speed
    #     left_dL = sqdist(left_dx,left_dy)
    #     right_dL = sqdist(right_dx,right_dy) 
    #     if left_dL>right_dL:
    #         speed_left = speed
    #         try:
    #             speed_right = speed*right_dL/left_dL
    #         except:
    #             speed_right = 0
    #     else:
    #         speed_right = speed
    #         speed_left = speed*left_dL/right_dL
    #     # print speed_left, speed_right
    #     # return the joint values
    #     return joint_values_left, joint_values_right, speed_left, speed_right, new_pos_left, new_pos_right


    def saveData(self, trial_dirname):  
        with open(trial_dirname + "/data_training_info.dat", "wb") as m:
            pickle.dump(self.info_list, m, protocol=pickle.HIGHEST_PROTOCOL)

