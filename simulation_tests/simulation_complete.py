
import argparse
import numpy as np
import util_modelling as umodel
import util_experiment as uexp
import util_testing as utest

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-v',   '--verbose',    
                    dest='verb',       
                    help="Define verbose level\n"
                         "0: basic info\n"
                         "1: training detailed info\n"
                         "2: test detailed info\n"
                         "3: training and test detailed info\n",
                    default=0)
parser.add_argument('-pl',   '--plots',    
                    dest='plots',       
                    help="Define plots to show\n"
                         "0: no plots\n"
                         "1: model plots\n"
                         "2: test plots\n"
                         "3: both model and test plots\n",
                    default=0)
parser.add_argument('-tr',  '--numtrial',   
                    dest='num_trial',  
                    help="Number of trials to run",
                    default=300)
parser.add_argument('-m',   '--model',      
                    dest='model_type', 
                    help="Select which model to use 'random' or 'informed'",
                    default='random')
parser.add_argument('-r',   '--resolution', 
                    dest='res',        
                    help="Select discretisation resolution",
                    default=150)
parser.add_argument('-o',   '--other', 
                    dest='other',        
                    help="Additional model specs",
                    default=[5, 0.01, 5])
args = parser.parse_args()

folder_name = '_'.join( args.model_type, 
                        'res'+str(args.res), 
                        'cov'+str(args.other[0]), 
                        'kernelSE'+str(args.other[1])+'seed'+str(args.other[2]))

# INITIALISE MODEL
print("INITIALISING MODEL: {}\n".format(folder_name))
print(args)
experiment = uexp.SimulationExperiment(resolution=args.res, animate=False, verbose=args.verb&1)
model      = umodel.InformedModel(experiment.parameter_list, experiment.type, show_plots=args.plots&1, other=args.other, folder_name=folder_name)
testing    = utest.FullTest(show_plots=args.plots&2, verbose=args.verb&2)

# RUN FULL EXPERIMENT
for t in range(args.num_trial):
    ##### TRAINING STEP #####
    print("\n\nTRIAL #", t+1, "(failed: ", len(model.failed_coords),", successful: ",len(model.coord_explored)-len(model.failed_coords),")")
    print("--- Current model uncertainty:", model.returnUncertainty())
    # Generate sample trial
    if args.model_type == 'informed':
        trial_coords, trial_params = model.generateInformedSample(experiment.info_list)
    elif args.model_type == 'random':
        trial_coords, trial_params = model.generateRandomSample()
    # Execute trial
    trial_info = experiment.executeTrial(t, trial_coords, trial_params)
    experiment.info_list.append(trial_info)
    # Update model
    model.updateModel(experiment.info_list)
    # Save experiment data
    experiment.saveData(model.trial_dirname)
    # Plot model progress
    if (t+1)%10 == 0:
        model.plotModel(t+1, [0,1], ['joint_0', 'joint_1'])

    ##### TESTING STEP #####
    if (t+1) > 1:
        print("\n\nTESTING {} cases...".format(len(testing.test_cases)))
        testing.runFullTests(t+1, experiment, model, save_progress=(not (t+1)%50), heatmap=(not (t+1)%10))

# FINAL MODEL PLOT
# model.plotModel('final_{}_top'.format(t+1), [0,1], ['joint_0', 'joint_1'], show=False, top_view=True)
# model.plotModel('final_{}_3d'.format(t+1),  [0,1], ['joint_0', 'joint_1'], show=False, top_view=False)


print("\n\nDONE.")