
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Main file for running experiments
"""

import os
import json
import time
import logging
import argparse
import datetime

import utils.experiment_manage as expm

logger = logging.getLogger(__name__) 

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# Experiment
parser.add_argument('-s', '--seed', default=100,      
                    type=int, help="Experiment seed.")
parser.add_argument('-ntrial', '--num-trial', default=300,  
                    type=int, help="Number of trials to run.")
parser.add_argument('-efreq', '--eval-freq', default=10,  
                    type=int, help="How often to evaluate on test set.")

# Task
parser.add_argument('-r', '--resolution', default=7,
                    help="Select discretisation resolution")

parser.add_argument('-e', '--environment', default='sim5link', 
                    help="Select which environment to use: "
                         "'sim2link'\n"
                         "'sim5link'\n"
                         "'robot'")
# Modelling
parser.add_argument('-m', '--search-type', default='UIDF', 
                    help="Select which model to use: "
                         "random\n"
                         "informed\n"
                         "UIDF\n"
                         "entropy\n"
                         "BO")

parser.add_argument('-pcov', '--pidf-cov', default=0.1, 
                    type=float,  
                    help="Penalisation function covariance coefficient.")

parser.add_argument('-kname', '--kernel-name', default='se',      
                    help="Gaussian Process Regression kernel function.")

parser.add_argument('-kls', '--kernel-lenscale', default=0.01,      
                    type=float,
                    help="Sigma coefficient of the kernel")

parser.add_argument('-ksig', '--kernel-sigma', default=1.,      
                    type=float, 
                    help="Sigma coefficient of the kernel")

# Utils
parser.add_argument('-p', '--show_plots', default=0,     
                    help="Define plots to show\n"
                             "0: no plots\n"
                             "1: model plots\n"
                             "2: test plots\n"
                             "3: both model and test plots\n")
parser.add_argument('-v', '--verbose', default=0,       
                    help="Define verbose level\n"
                             "0: basic info\n"
                             "1: training detailed info\n"
                             "2: test detailed info\n"
                             "3: training and test detailed info\n")


def _start_logging():
    """
        Create the experiment directory and start logging
    """
    args = parser.parse_args()
    dirname = "experiment_data/simulation/" \
                "ENV_{}__SEARCH_{}_res{}_cov{}_kernelSE_sl{}__{}_{}".format(
                args.environment,
                args.search_type, 
                args.resolution, 
                args.pidf_cov, args.kernel_sigma, args.seed,
                datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(dirname)
    task_kwargs = vars(args)
    task_kwargs['dirname'] = dirname
    # Start logging info
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-25s '
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
    logger.info('Starting session: {}\n'.format(dirname))
    return task_kwargs


def main_run():
    # Initialise
    task_kwargs = _start_logging()
    experiment = expm.ExperimentManager(task_kwargs)
    # Run main loop
    for ntrial_ in range(1, task_kwargs['num_trial']+1):         
        experiment.execute_trial(ntrial_)
        if ntrial_%task_kwargs['eval_freq'] == 0:
            experiment.evaluate_test_cases(ntrial_)
    # Training done
    logger.info("\t>>> TRAINING DONE [successful: {}; failed: {}]".format(
                                            experiment.environment.n_success, 
                                            experiment.environment.n_fail))


if __name__ == "__main__":
    try:
            main_run()
    except Exception as e:
            logging.fatal(e, exc_info=True)

