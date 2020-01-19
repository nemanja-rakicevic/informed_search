
"""
Author: Nemanja Rakicevic
Date  : January 2018
Description:
            Main file to run simulation experiments

"""

import os
import json
import logging
import argparse
import datetime
import numpy as np

import util_modelling as umodel
import utils.environments as uenv
import utils.testing as utest


logger = logging.getLogger(__name__) 


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-m',   '--model_type',
                    default='reviewer', 
                    help="Select which model to use: "
                         "random\n"
                         "informed\n"
                         "uidf\n"
                         "entropy\n"
                         "reviewer")
parser.add_argument('-r',   '--resolution',
                    default=7,        
                    help="Select discretisation resolution")

parser.add_argument('-pcov',   '--pidf_cov', type=float,
                    default=0.1,      
                    help="Penalisation function covariance coefficient.")

parser.add_argument('-k',   '--kernel_name',
                    default='RBF',      
                    help="Gaussian Process Regression kernel function.")
parser.add_argument('-sig',   '--kernel_sigma', nargs='+', type=float,
                    default=0.01,      
                    help="Sigma coefficient of the kernel")


parser.add_argument('-nt',  '--num_trial', type=int,
                    default=10,  
                    help="Number of trials to run")

parser.add_argument('-e',   '--environment',
                    default='sim5link', 
                    help="Select which environment to use: "
                         "'sim2link'\n"
                         "'sim5link'\n"
                         "'robot'")

parser.add_argument('-s',   '--seed', type=int,
                    default=100,      
                    help="Experiment seed.")
parser.add_argument('-pl',   '--plots',
                    default=0,     
                    help="Define plots to show\n"
                         "0: no plots\n"
                         "1: model plots\n"
                         "2: test plots\n"
                         "3: both model and test plots\n")
parser.add_argument('-v',   '--verbose',
                    default=0,       
                    help="Define verbose level\n"
                         "0: basic info\n"
                         "1: training detailed info\n"
                         "2: test detailed info\n"
                         "3: training and test detailed info\n")


def _start_logging():
    args = parser.parse_args()
    dirname = "experiment_data/simulation/"\
              "ENV_{}__MODEL_{}_res{}_cov{}_kernelSE_sl{}__{}_{}".format(
              args.environment,
              args.model_type, 
              args.resolution, 
              args.pidf_cov, args.kernel_sigma, args.seed,
              datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(dirname)
    task_kwargs = vars(args)
    task_kwargs['dirname'] = dirname
    # Start logging info
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-10s '
                               '%(levelname)-8s %(message)s',
                        handlers=[
                            logging.FileHandler(
                                '{}/logging_data.log'.format(dirname)),
                            logging.StreamHandler()
                        ])
    # Save the modified arguments
    filename = '{}/experiment_metadata.json'.format(dirname)
    with open(filename, 'w') as outfile:  
        json.dump(task_kwargs, outfile, sort_keys=True, indent=4)
    # print(task_kwargs)
    logger.info('Starting session: {}\n'.format(dirname))
    return task_kwargs


# INITIALISE MODEL

def main_run():
    # Initialise stuff
    task_kwargs = _start_logging()
    # experiment = mm.ExperimentManager(task_kwargs)


    experiment = uenv.SimulationExperiment(**task_kwargs)
    model = umodel.InformedModel(experiment.parameter_list, 
                                 experiment.type, 
                                 show_plots=0, 
                                 other=[0.1, 0.01, 1], 
                                 folder_name=task_kwargs['dirname'])

    testing = utest.FullTest(experiment, model, 
                             show_plots=0, 
                             verbose=1)

    # RUN FULL EXPERIMENT
    for t in range(task_kwargs['num_trial']):
        # Display trial information
        logger.info("\n\nTRIAL #{} (failed: {}; successful: {})".format(
                        t+1, len(model.failed_coords),
                        len(model.coord_explored)-len(model.failed_coords)))
        logger.info("--- Current model uncertainty: {}".format(
                        model.returnUncertainty()))

        # Generate sample trial
        if task_kwargs['model_type'] == 'informed':
            trial_coords, trial_params = model.generateInformedSample(experiment.info_list)
        elif task_kwargs['model_type'] == 'random':
            trial_coords, trial_params = model.generateRandomSample()
        elif task_kwargs['model_type'] == 'uidf':
            trial_coords, trial_params = model.generateUIDFSample(experiment.info_list)
        elif task_kwargs['model_type'] == 'entropy':
            trial_coords, trial_params = model.generateEntropySample(experiment.info_list)
        elif task_kwargs['model_type'] == 'reviewer':
            trial_coords, trial_params = model.generateInformedSample_reviewer(experiment.info_list)
     
        # Execute trial
        trial_info = experiment.execute_trial(t, trial_coords, trial_params)
        experiment.info_list.append(trial_info)

        # Update model
        # model.updateModel(experiment.info_list, save_progress=(not (t+1)%100))
        model.updateModel_reviewer(experiment.info_list, save_progress=(not (t+1)%100))
        # Save experiment data and plot  progress
        experiment.save_data(model.trial_dirname)
        if (t+1)%10 == 0:
            logger.info("\n\nPLOTTING")
            model.plotModelFig(t+1, [0,1], ['joint_1', 'joint_0'])

        # Model evaluation
        if (t+1) > 1:
            logger.info("\n\nTESTING {} cases...".format(len(testing.test_cases)))
            # testing.runFullTests(t+1, experiment, model, save_progress=(not (t+1)%10), heatmap=(not (t+1)%10))
            testing.runFullTests(t+1, save_progress=(not (t+1)%10), heatmap=(not (t+1)%10))

    # FINAL MODEL PLOT
    # model.plotModel('final_{}_top'.format(t+1), [0,1], ['joint_0', 'joint_1'], show=False, top_view=True)
    # model.plotModel('final_{}_3d'.format(t+1),  [0,1], ['joint_0', 'joint_1'], show=False, top_view=False)


    print("\n\n\t>>> TRAINING DONE.")




if __name__ == "__main__":
    try:
        main_run()
    except Exception as e:
        logging.fatal(e, exc_info=True)

