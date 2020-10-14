
"""
Author:         Nemanja Rakicevic
Date:           October 2020
Description:
                Transfer the collection of explored points from a source
                environment, to a target environment, to help learn the forward
                model efficiently.
"""

import os
import json
import pickle
import logging
import argparse
import datetime

import informed_search.tasks.experiment_manage as expm

from informed_search.utils.misc import _TAB


logger = logging.getLogger(__name__)


def load_metadata(source_path):
    """Extract class experiment metadata."""
    with open(source_path + '/experiment_metadata.json', 'r') as f:
        args_dict = json.load(f)
    return args_dict


def load_trial_data(source_path):
    """Extract class experiment parameter data."""
    with open(source_path + '/statistics_trials.pkl', "rb") as f:
        info_list = pickle.load(f)
    return info_list


def _start_logging(taskargs):
    """Create the experiment directory and start logging."""
    dirname = os.path.join(os.getcwd(), 'experiment_data', 'simulation')
    dirname = os.path.join(
        dirname,
        "env__{}".format(taskargs['environment']),
        "ENV_{}__SEARCH_{}-transfer__res{}_pidf{}_kernel{}_sl{}__{}_{}".format(
            taskargs['environment'],
            taskargs['search_algo'],
            taskargs['resolution'],
            taskargs['pidf_coeff'],
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
    logger.info("Transferring environments: {} >> {}\n".format(
        taskargs['environment_source'], taskargs['environment']))
    return taskargs


def main_test(source_path):
    """Run source environments point in new environment and evaluate."""
    # Apply to the corresponding target environment (same parameterisation)
    source_task = load_metadata(source_path)
    source_env = source_task['environment']
    source_task['environment_source'] = source_task['environment']
    if 'nl' in source_env:
        source_task['environment'] = source_task['environment_source'][:-3]
    else:
        source_task['environment'] = source_task['environment_source'] + '_nl'
    # Start new transfer experiment
    task_kwargs = _start_logging(source_task)
    experiment = expm.ExperimentManager(task_kwargs)
    # Load successful points from source experiment
    info_list = load_trial_data(source_path)
    succ_params = [pp['parameters'] for pp in info_list
                   if pp['trial_outcome'] == 'SUCCESS']
    succ_coords = [pp['coordinates'] for pp in info_list
                   if pp['trial_outcome'] == 'SUCCESS']
    # Loop through points, execute and update model
    transfer_info_list = []
    for ntrial_, (sc_, sp_) in enumerate(zip(succ_coords, succ_params)):
        experiment.model.coord_explored.append(sc_)
        trial_info = experiment.environment.execute_trial(
            sc_, sp_, num_trial=ntrial_)
        if trial_info['trial_outcome'] == 'FAIL':
            experiment.model.coord_failed.append(sc_)
        transfer_info_list.append(trial_info)
        experiment.model.update_model(transfer_info_list, num_trial=ntrial_)
        logger.info("Transferring TRIAL {} / {}:"
                    "\n{} - Trial coords: {} params: {}"
                    "\n{} - Fail_status: {}; Distance to target: {:4.2f}"
                    "\n{} - Total (failed: {}; successful: {})"
                    "\n{} - Updated model uncertainty: {:4.2}".format(
                        ntrial_ + 1, len(succ_coords),
                        _TAB, sc_, sp_, _TAB,
                        trial_info['fail_status'], trial_info['target_dist'],
                        _TAB, experiment.environment.n_fail,
                        experiment.environment.n_success, _TAB,
                        experiment.model.uncertainty))
        experiment.evaluate_test_cases(ntrial_)
    logger.info("\t>>> TRANSFER EXPERIMENT DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-source', '--source_path',
                        default=None, required=True,
                        help="Path to the source model file.")
    args = parser.parse_args()
    main_test(source_path=args.source_path)
