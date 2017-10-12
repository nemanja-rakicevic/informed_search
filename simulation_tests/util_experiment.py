
import numpy as np 


class SimulationExperiment:

	import gym

	__NUM_STEPS = 100
	__range1 = np.linspace(0, 2, 50)
	__range2 = np.linspace(0, -1, 50)

	def __init__(self):
		self.parameter_list = np.array([self.__range1, self.__range2])
		self.info_list = []
		self.env = self.gym.make('ReacherOneShot-v0')

	def executeTrial(self, t, coords, params):
	    theta_list = np.array([np.linspace(0, params[0], self.__NUM_STEPS), np.linspace(0, params[1], self.__NUM_STEPS)]).T
	    init_pos = self.env.reset()
	    init_pos = init_pos[:2]
	    # EXECUTE TRIAL
	    contact_cnt = 0
	    obs_list = []
	    for i in range(self.__NUM_STEPS):
	        self.env.render()
	        control = init_pos + theta_list[i]
	        observation, _, _, _ = self.env.step(control) 
	        obs_list.append(observation)
	        ball_xy = observation[2:4]
	        # Check collision
	        if self.env.unwrapped.data.ncon:
	            contact_cnt+=1
	            if contact_cnt > 5 and not sum(ball_xy)>0:
	                # contact with wall
	                fail_status = 1     
	                break
	    # Check ball movement and calculate polar coords
	    if not sum(ball_xy)>0:
	        # ball did not move
	        fail_status = 2  
	        ball_polar = np.array([0,0])   
	    else:
	        fail_status = 0  
	        ball_polar = np.array([ np.rad2deg(np.arctan2(-ball_xy[0], ball_xy[1])), np.sqrt(ball_xy[0]**2 + ball_xy[1]**2) * 100])
	    # Compile trial info
	    all_info = {'trial_num': t+1,
	                'parameters': params,
	                'coordinates': coords,
	                'fail': fail_status, 
	                'ball_polar': ball_polar,
	                'observations': np.array(obs_list) }
	    # INFO
	    # print("\nTRIAL #", t+1,"params",[param1,param2])
	    # print("CTRL:\t",  control)
	    # print("Q_POS:\t", ob[:2])
	    # print("Q_VEL:\t", ob[4:6])
	    # print("BALL:\t",  ball_xy, ball_polar)
	    self.env.render(close=True)
	    return all_info


	def saveData(self):	
		with open("./DATA/Simulation_data.dat", "wb") as m:
			pickle.dump(self.info_list, m, protocol=pickle.HIGHEST_PROTOCOL)




class RobotExperiment:

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
		self.info_list = []
		self.parameter_list = np.array([self.__range_l_dx, self.__range_l_dy, self.__range_r_dx, self.__range_r_dy, self.__range_wrist, self.__range_speed])

	def executeTrial(t, params):
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
	    if not sum(ball_xy)>0:
	        # ball did not move
	        fail_status = 2  
	        ball_polar = np.array([0,0])   
	    else:
	        fail_status = 0  
	        ball_polar = np.array([ np.rad2deg(np.arctan2(-ball_xy[0], ball_xy[1])), np.sqrt(ball_xy[0]**2 + ball_xy[1]**2) * 100])
	    # Compile trial info
	    all_info = {'trial_num': t+1,
	                'parameters': params,
	                'coordinates': coords,
	                'fail': fail_status, 
	                'ball_polar': ball_polar,
	                'observations': np.array(obs_list) }
	    # INFO
	    # print("\nTRIAL #", t+1,"params",[param1,param2])
	    # print("CTRL:\t",  control)
	    # print("Q_POS:\t", ob[:2])
	    # print("Q_VEL:\t", ob[4:6])
	    # print("BALL:\t",  ball_xy, ball_polar)
	    return all_info


	def saveData(self):	
	    with open("./DATA/Simulation_data.dat", "wb") as m:
	        pickle.dump(self.info_list, m, protocol=pickle.HIGHEST_PROTOCOL)