

import numpy as np

import util_pdf as updf
import util_experiment as uexp

NUM_TRIALS = 10

# INITIALISE MODEL
print("INITIALISING MODEL\n")
experiment = uexp.SimulationExperiment(display=True)
model      = updf.InformedModel(experiment.parameter_list)

# RUN TRAINING
for t in range(150):
   
    ### generate sample
    print("\n\nTRIAL #", t+1, "(failed: ", len(model.failed_coords),", successful: ",len(model.coord_explored)-len(model.failed_coords),")")
    print("--- Current model uncertainty:", model.returnUncertainty())

    trial_coords, trial_params = model.generateInformedSample(experiment.info_list)
    # trial_coords, trial_params = model.generateRandomSample()

    #### execute trial
    trial_info = experiment.executeTrial(t, trial_coords, trial_params)
    experiment.info_list.append(trial_info)
    #### update model
    model.updateModel(experiment.info_list)
    #### print progress
    if trial_info['fail']==0:
        model.plotModel(t+1, [0,1], ['joint_0', 'joint_1'])
    #### save data
    # experiment.saveData(model.trial_dirname)
model.plotModel(t+1, [0,1], ['joint_0', 'joint_1'])
    

# RUN TESTS
while True:
    ### Do paired Cartesian sqrt distance but check that it's not in the failed trials!!!
    # Select (angle, L) pair which is closest to the desired one
    angle_s, dist_s = input("\nEnter GOAL angle, distance: ")
    trial_coords, trial_params = model.testModel(angle_s, dist_s)
    experiment.display = True
    trial_info = experiment.executeTrial(0, trial_coords, trial_params)

    test_q = raw_input("\nEXECUTION DONE. Enter to try again, or (x) to quit ")
    if test_q=='x':
        break