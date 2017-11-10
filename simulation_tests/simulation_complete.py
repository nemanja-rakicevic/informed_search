
import numpy as np
import util_modelling as umodel
import util_experiment as uexp
import util_testing as utest

NUM_TRIALS = 300
# INITIALISE MODEL
print("INITIALISING MODEL\n")
experiment = uexp.SimulationExperiment(display=False)
model      = umodel.InformedModel(experiment.parameter_list, experiment.type)
testing    = utest.FullTest(show=False, verbose=False)

# RUN FULL EXPERIMENT
for t in range(NUM_TRIALS):
    ##### TRAINING STEP #####
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
    # Save experiment data
    experiment.saveData(model.trial_dirname)
    # Plot model progress
    if (t+1)%10 == 0:
        model.plotModel(t+1, [0,1], ['joint_0', 'joint_1'], show=True, show_points=False)

        ##### TESTING STEP #####
        print("\n\nTESTING {} cases...".format(len(testing.test_cases)))
        testing.runFullTests(t+1, experiment, model)

# FINAL MODEL PLOT
# model.plotModel('final_{}_top'.format(t+1), [0,1], ['joint_0', 'joint_1'], show=False, top_view=True)
# model.plotModel('final_{}_3d'.format(t+1),  [0,1], ['joint_0', 'joint_1'], show=False, top_view=False)


print("\n\nDONE.")