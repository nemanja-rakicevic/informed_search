
import numpy as np
import itertools
import pickle
import util_modelling as umodel
import util_experiment as uexp

# INITIALISE MODEL
print("INITIALISING MODEL\n")
experiment = uexp.SimulationExperiment(display=False, display_steps=100)
model      = umodel.InformedModel(experiment.parameter_list, test=True)
model.loadModel()

# DEFINE TESTING POINS
test_angles = np.arange(-25, 16, 5)
test_dist   = np.arange(0, 21, 5)
test_cases  = np.array([xs for xs in itertools.product(test_angles, test_dist)])

# RUN TESTS
input("\nRUN {} TESTS?".format(len(test_cases)))
statistics = []
for t in range(len(test_cases)):
    # Input goal position
    angle_s, dist_s = test_cases[t]
    print("\nTEST #{} > angle, distance: ({},{})".format(t+1, angle_s, dist_s))
    # Generate movement parameter vector
    trial_coords, trial_params = model.testModel(float(angle_s), float(dist_s), verbose=0)
    trial_info = experiment.executeTrial(0, trial_coords, trial_params)
    # Compile test statistics
    sq_error = np.sqrt((trial_info['ball_polar'] - test_cases[t])**2)
    statistics.append({ 'trial_num':    t+1,
                        'target_polar': test_cases[t],
                        'ball_polar':   trial_info['ball_polar'],
                        'fail':         trial_info['fail'],
                        'error':        sq_error })
   
# Calculate error
num_fails  = len([1 for x in statistics if x['fail']>0])
errors_all = np.array([x['error'] for x in statistics])
error_mean = errors_all.mean()
error_std  = errors_all.std()

print("\nTESTING COMPLETE.\nPERFORMANCE ERRORS\t> mean: {}, std: {}".format(error_mean, error_std))
print("\t\t\t> failed: {}".format(num_fails))

print("saving results...")
with open(model.trial_dirname + "/data_test_statistics.dat", "wb") as m:
    pickle.dump([statistics, error_mean, error_std, num_fails], m, protocol=pickle.HIGHEST_PROTOCOL)
