
import numpy as np
import util_modelling as umodel
import util_experiment as uexp

# INITIALISE MODEL
print("INITIALISING MODEL\n")
experiment = uexp.SimulationExperiment(display=True, display_steps=100)
model      = umodel.InformedModel(experiment.parameter_list, test=True)
model.loadModel()
# RUN TESTS
input("\nRUN TESTS?")  
experiment.display = True
while True:
    # Input goal position
    while True:
         try:
             angle_s, dist_s = input("\nEnter GOAL angle, distance: ").split(",")
             break
         except Exception as i:
             print(i)
             continue
    trial_coords, trial_params = model.testModel(float(angle_s), float(dist_s))
    trial_info = experiment.executeTrial(0, trial_coords, trial_params)
    # Continue
    if input("\nEXECUTION DONE. Enter to try again, or (x) to quit ") == 'x':
        break