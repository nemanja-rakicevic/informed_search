"""
Author: Nemanja Rakicevic
Date  : January 2018
Description:
            Main run file (rename to run_experiments)

"""

import os
import logging
import argparse
import datetime
import numpy as np

import util_modelling as umodel
import util_experiment as uexp
import util_testing as utest


# TODO use config
# TODO put stuff in main()
# TODO add logging

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
parser.add_argument('-e',   '--env_type',
                    default='sim5link', 
                    help="Select which environment to use 'sim2link','sim5link' or 'robot'")
parser.add_argument('-r',   '--resolution',
                    default=7,        
                    help="Select discretisation resolution")
parser.add_argument('-v',   '--verbose',
                    default=0,       
                    help="Define verbose level\n"
                         "0: basic info\n"
                         "1: training detailed info\n"
                         "2: test detailed info\n"
                         "3: training and test detailed info\n")
parser.add_argument('-pl',   '--plots',
                    default=0,     
                    help="Define plots to show\n"
                         "0: no plots\n"
                         "1: model plots\n"
                         "2: test plots\n"
                         "3: both model and test plots\n")
parser.add_argument('-tr',  '--num_trial',
                    default=300,  
                    help="Number of trials to run")
parser.add_argument('-o',   '--other', nargs='+', type=float,
                    default=[0.1, 0.01, 1],      
                    help="Additional model specs list\n"
                         "[COV, siglensq, seed]\n")



# INITIALISE MODEL

def main_run():
    args = parser.parse_args()
    # dirname = '_'.join([args.env_type,
    #                     args.model_type, 
    #                     'res'+str(args.res), 
    #                     'cov'+str(int(args.other[0])), 
    #                     'kernelSE_sl'+str(args.other[1])+'-seed'+str(int(args.other[2]))])
    dirname = "experiment_data/simulation"\
              "ENV_{}__MODEL_{}_res{}_cov{}_kernelSE_sl{}__{}_{}".format(
              args.env_type,
              args.model_type, 
              args.resolution, 
              int(args.other[0]), args.other[1], int(args.other[2]),
              datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(dirname)
    # Start logging info
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-52s '
                               '%(levelname)-8s %(message)s',
                        handlers=[
                            logging.FileHandler(
                                '{}/logging_data.log'.format(dirname)),
                            logging.StreamHandler()
                        ])
    logger.info('Starting session: {}\n'.format(dirname))
    print(args)



    experiment = uexp.SimulationExperiment(agent=args.env_type, 
                                           resolution=args.resolution, 
                                           animate=False, 
                                           verbose=args.verbose&1)
    model = umodel.InformedModel(experiment.parameter_list, 
                                 experiment.type, 
                                 show_plots=args.plots&1, 
                                 other=args.other, 
                                 folder_name=dirname)
    testing = utest.FullTest(experiment, model, 
                             show_plots=args.plots&2, 
                             verbose=args.verbose&2)

    # RUN FULL EXPERIMENT
    for t in range(args.num_trial):
        # Display trial information
        logger.info("\n\nTRIAL #{} (failed: {}; successful: {})".format(
                        t+1, len(model.failed_coords),
                        len(model.coord_explored)-len(model.failed_coords)))
        logger.info("--- Current model uncertainty: {}".format(
                        model.returnUncertainty()))

        # Generate sample trial
        if args.model_type == 'informed':
            trial_coords, trial_params = model.generateInformedSample(experiment.info_list)
        elif args.model_type == 'random':
            trial_coords, trial_params = model.generateRandomSample()
        elif args.model_type == 'uidf':
            trial_coords, trial_params = model.generateUIDFSample(experiment.info_list)
        elif args.model_type == 'entropy':
            trial_coords, trial_params = model.generateEntropySample(experiment.info_list)
        elif args.model_type == 'reviewer':
            trial_coords, trial_params = model.generateInformedSample_reviewer(experiment.info_list)
     
        # Execute trial
        trial_info = experiment.executeTrial(t, trial_coords, trial_params)
        experiment.info_list.append(trial_info)

        # Update model
        # model.updateModel(experiment.info_list, save_progress=(not (t+1)%100))
        model.updateModel_reviewer(experiment.info_list, save_progress=(not (t+1)%100))
        # Save experiment data and plot  progress
        experiment.saveData(model.trial_dirname)
        if (t+1)%10 == 0:
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

