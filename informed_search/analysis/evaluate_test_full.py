
"""
Author:         Nemanja Rakicevic
Date  :         June 2020
Description:
                Load a model from the experiment directory and evaluate it on
                the full test set.
"""

import json
import argparse
import numpy as np

import informed_search.tasks.experiment_manage as expm


def load_metadata(datapath):
    """Extract class experiment metadata."""
    with open(datapath + '/experiment_metadata.json', 'r') as f:
        args_dict = json.load(f)
    return args_dict


def main_test(load_path):
    """Run model evaluation on full test set."""
    load_task_kwargs = load_metadata(load_path)
    experiment = expm.ExperimentManager(
        load_task_kwargs,
        load_model_path=load_path)
    test_stats = experiment.evaluate_test_cases(
        num_trial=None,
        save_model=False,
        save_test_progress=False,
        verbose=True,
        save_plots=False,
        save_data=False)
    # Extract information
    errors_all = np.array([[x['euclid_error'], x['polar_error']]
                           for x in test_stats])
    errors_mean = errors_all.mean(axis=0)
    errors_std = errors_all.std(axis=0)
    num_total = len(test_stats)
    num_fails = sum([x['fail_status'] > 0 for x in test_stats])
    num_success = num_total - num_fails
    model_error_mean = np.mean([x['model_polar_error'] for x in test_stats])
    # Print outcome info
    print("\n{} TESTING {} cases {}"
          "\n - Sucessful/Failed:       {} / {} ({})"
          "\n - Model polar error mean: {:4.2f}"
          "\n - Euclidian error mean:   {:4.2f}"
          "\n - Polar error norm mean:  {:4.2f}\n{}".format(
              '-' * 15, num_total, '-' * 15,
              num_success, num_fails,
              num_total, model_error_mean,
              errors_mean[0], errors_mean[1],
              '-' * 50))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-load', '--load_path',
                        default=None, required=True,
                        help="Path to the learned model file.")
    args = parser.parse_args()
    main_test(load_path=args.load_path)
