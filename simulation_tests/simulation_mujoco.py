

import numpy as np

import util_pdf as updf
import util_experiment as uexp

#################################################################
NUM_TRIALS = 10

# # LOAD ENVIRONMENT
# # if 'env' in locals(): 
# #     env.render(close=True)
# env = gym.make('ReacherOneShot-v0')

# # DEFINE PARAMETER RANGE
# range1 = np.linspace(0, 2, 50)
# range2 = np.linspace(0, -1, 50)
# parameter_list = np.array([range1, range2])


#################################################################
# def executeTrial(t, params):
#     theta_list = np.array([np.linspace(0, params[0], NUM_STEPS), np.linspace(0, params[1], NUM_STEPS)]).T
#     init_pos = env.reset()
#     init_pos = init_pos[:2]
#     # EXECUTE TRIAL
#     contact_cnt = 0
#     obs_list = []
#     for i in range(NUM_STEPS):
#         env.render()
#         control = init_pos + theta_list[i]
#         observation, _, _, _ = env.step(control) 
#         obs_list.append(observation)
#         ball_xy = observation[2:4]
#         # Check collision
#         if env.unwrapped.data.ncon:
#             contact_cnt+=1
#             if contact_cnt > 5 and not sum(ball_xy)>0:
#                 # contact with wall
#                 fail_status = 1     
#                 break
#     # Check ball movement and calculate polar coords
#     if not sum(ball_xy)>0:
#         # ball did not move
#         fail_status = 2  
#         ball_polar = np.array([0,0])   
#     else:
#         fail_status = 0  
#         ball_polar = np.array([ np.rad2deg(np.arctan2(-ball_xy[0], ball_xy[1])), np.sqrt(ball_xy[0]**2 + ball_xy[1]**2) * 100])
#     # Compile trial info
#     all_info = {'trial_num': t+1,
#                 'parameters': params,
#                 'coordinates': coords,
#                 'fail': fail_status, 
#                 'ball_polar': ball_polar,
#                 'observations': np.array(obs_list) }
#     # INFO
#     # print("\nTRIAL #", t+1,"params",[param1,param2])
#     # print("CTRL:\t",  control)
#     # print("Q_POS:\t", ob[:2])
#     # print("Q_VEL:\t", ob[4:6])
#     # print("BALL:\t",  ball_xy, ball_polar)
#     return all_info
# 
#################################################################
# INITIALISE MODEL
print("INITIALISING MODEL\n")
experiment = uexp.SimulationExperiment()
model      = updf.InformedModel(experiment.parameter_list)

# RUN TRIALS
for t in range(5):
    ### generate sample
    # param1 = np.random.choice(range1)
    # param2 = np.random.choice(range2)
    model_uncertainty = model.returnUncertainty()
    # print "- Generating new trial parameters...\n- Current model entropy:"+str(round(avar,4))
    trial_coords, trial_params = model.generateInformedSample(info_list)
    print("\nTRIAL #", t+1,"params", trial_params)

    #### execute trial
    trial_info = experiment.executeTrial(t, trial_coords, trial_params)
    info_list.append(trial_info)

    #### update model
    model.updateModel(info_list)

    #### save data
    experiment.saveData()

    #### print progress
    model.plotModel(t+1, [0,1], ['joint_0', 'joint_1'])
    
    print(trial_info)