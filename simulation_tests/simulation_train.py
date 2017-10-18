
import numpy as np
import util_modelling as umodel
import util_experiment as uexp

NUM_TRIALS = 50

# INITIALISE MODEL
print("INITIALISING MODEL\n")
experiment = uexp.SimulationExperiment(display=False)
model      = umodel.InformedModel(experiment.parameter_list, experiment.type)

# RUN TRAINING
for t in range(NUM_TRIALS):
    # Generate sample trial
    print("\n\nTRIAL #", t+1, "(failed: ", len(model.failed_coords),", successful: ",len(model.coord_explored)-len(model.failed_coords),")")
    print("--- Current model uncertainty:", model.returnUncertainty())
    trial_coords, trial_params = model.generateInformedSample(experiment.info_list)
    # trial_coords, trial_params = model.generateRandomSample()
    # Execute trial
    trial_info = experiment.executeTrial(t, trial_coords, trial_params)
    experiment.info_list.append(trial_info)
    # Update model
    model.updateModel(experiment.info_list)
    # Plot model progress
    if (t+1)%10 == 0:
        model.plotModel(t+1, [0,1], ['joint_0', 'joint_1'], show=False, top_view=True)
    # Save experiment data
    experiment.saveData(model.trial_dirname)
model.plotModel('final_{}_top'.format(t+1), [0,1], ['joint_0', 'joint_1'], show=False, top_view=True)
model.plotModel('final_{}_3d'.format(t+1),  [0,1], ['joint_0', 'joint_1'], show=False, top_view=False)


# RUN TESTS
input("\nRUN TESTS?")  
experiment.display = True
while True:
    # Input goal position
    angle_s, dist_s = input("\nEnter GOAL angle, distance: ").split(",")
    trial_coords, trial_params = model.testModel(float(angle_s), float(dist_s))
    trial_info = experiment.executeTrial(0, trial_coords, trial_params)
    # Continue
    if input("\nEXECUTION DONE. Enter to try again, or (x) to quit ") == 'x':
        break