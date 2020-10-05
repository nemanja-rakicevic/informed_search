
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Main file for running experiments
"""

import os
import sys
import json
import logging
import argparse
import datetime

import informed_search.tasks.experiment_manage as expm


logger = logging.getLogger(__name__)


def _load_args():
    """Load experiment args from config, overwrite the ones from cmd line."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--config_file', required=True)
    # Experiment
    parser.add_argument('--seed', default=100,
                        type=int, help="Experiment seed.")
    parser.add_argument('--num_trial', default=300,
                        type=int, help="Number of trials to run.")
    parser.add_argument('--eval_freq', default=10,
                        type=int, help="How often to evaluate on test set.")
    # Task
    parser.add_argument('--resolution', default=7,
                        help="Select discretisation resolution")
    parser.add_argument('--environment', default='sim2link',
                        help="Select which environment to use: "
                             "'sim2link'\n"
                             "'sim5link'\n"
                             "(TODO)'robot'")
    # Modelling
    parser.add_argument('--search_type', default='UIDF',
                        help="Select which model to use: "
                             "random\n"
                             "informed\n"
                             "UIDF\n"
                             "entropy\n"
                             "BO")
    parser.add_argument('--pidf_cov', default=0.1,
                        type=float,
                        help="Penalisation function covariance coefficient.")
    parser.add_argument('--kernel_name', default='se',
                        help="Gaussian Process Regression kernel function: "
                             "se\n"
                             "mat\n"
                             "rq")
    parser.add_argument('--kernel_lenscale', default=0.01,
                        type=float,
                        help="Sigma coefficient of the kernel")
    parser.add_argument('--kernel_sigma', default=1.,
                        type=float,
                        help="Sigma coefficient of the kernel")
    # Utils
    parser.add_argument('--show_plots', default=0,
                        help="Define plots to show\n"
                             "0: no plots\n"
                             "1: model plots\n"
                             "2: test plots\n"
                             "3: both model and test plots\n")
    parser.add_argument('--verbose', default=0,
                        help="Define verbose level\n"
                             "0: basic info\n"
                             "1: training detailed info\n"
                             "2: test detailed info\n"
                             "3: training and test detailed info\n")
    args = parser.parse_args()
    # Load default arguments from config file
    with open(args.config_file) as json_file:
        metadata = json.load(json_file)
    metadata['config_file'] = args.config_file
    # Additional cmd line args to overwrite file
    if len(sys.argv) > 3:
        args = vars(args)
        for a in sys.argv[3:]:
            if '--' in a:
                k = a[2:]  # k is the actual key
                if k in metadata.keys():
                    metadata[k] = args[k]
    return metadata


def _start_logging(taskargs):
    """Create the experiment directory and start logging."""
    dirname = os.path.join(os.getcwd(), 'experiment_data', 'simulation')
    dirname = os.path.join(
        dirname,
        "env__{}".format(taskargs['environment']),
        "ENV_{}__SEARCH_{}__res{}_cov{}_kernel{}_sl{}__{}_{}".format(
            taskargs['environment'],
            taskargs['search_type'],
            taskargs['resolution'],
            taskargs['pidf_cov'],
            taskargs['kernel_name'].upper(),
            taskargs['kernel_lenscale'],
            taskargs['seed'],
            datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(dirname)
    taskargs['dirname'] = dirname
    # Start logging info
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-40s '
                               '%(levelname)-8s %(message)s',
                        handlers=[
                            logging.FileHandler(
                                '{}/logging_data.log'.format(dirname)),
                            logging.StreamHandler()
                        ])
    # Save the modified arguments
    filename = '{}/experiment_metadata.json'.format(dirname)
    with open(filename, 'w') as outfile:
            json.dump(taskargs, outfile, sort_keys=True, indent=4)
    logger.info('Starting session: {}\n'.format(dirname))
    return taskargs


def main_run():
    """Main experiment execution."""
    # Initialise
    task_kwargs = _load_args()
    task_kwargs = _start_logging(task_kwargs)
    experiment = expm.ExperimentManager(task_kwargs)
    # Run main loop
    for ntrial_ in range(1, task_kwargs['num_trial'] + 1):
        if experiment.run_trial(ntrial_) is not None:
            break
        if task_kwargs['eval_freq'] and ntrial_ % task_kwargs['eval_freq'] == 0:
            experiment.evaluate_test_cases(ntrial_)
    # Training done
    logger.info("\t>>> TRAINING DONE [successful: {}; failed: {}]\n\n\n".format(
        experiment.environment.n_success,
        experiment.environment.n_fail))


if __name__ == "__main__":
    try:
        main_run()
    except Exception as e:
        logging.fatal(e, exc_info=True)
