
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Load a model from the experiment directory and evaluate it on a
                user defined test.
"""

import sys
import json
import logging
import argparse

import informed_search.tasks.experiment_manage as expm


def load_metadata(datapath):
    """Extract class experiment metadata."""
    with open(datapath + '/experiment_metadata.json', 'r') as f:
        args_dict = json.load(f)
    return args_dict


def main_test(load_path, verbose, render):
    """Run model evaluation on test targets."""
    load_task_kwargs = load_metadata(load_path)
    load_task_kwargs.update(dict(render=render, verbose=verbose))
    experiment = expm.ExperimentManager(
        load_task_kwargs,
        load_model_path=load_path)
    # Allow evaluation of multiple test positions
    while True:
        # Input target position
        while True:
            try:
                angle_s, dist_s = input(
                    "\nEnter TARGET: angle [deg], distance[m] >> ").split(",")
                test_target = [float(angle_s), float(dist_s)]
                break
            except Exception as i:
                print(i)
                continue
        # Evaluate given test position
        _, _, test_stats = experiment.evaluate_single_test(test_target)
        # Print outcome info
        print("{} TEST TARGET (angle: {}; distance: {}) {}"
              "\n - Trial outcome:     {} [{}]; ball ({:4.2f},{:4.2f})"
              "\n - Model polar error: {:4.2f}"
              "\n - Euclidian error:   {:4.2f}"
              "\n - Polar error norm:  {:4.2f}\n{}".format(
                  '-' * 7, angle_s, dist_s, '-' * 7,
                  test_stats['trial_outcome'], test_stats['fail_status'],
                  *test_stats['ball_polar'],
                  test_stats['model_polar_error'],
                  test_stats['euclid_error'],
                  test_stats['polar_error'], '-' * 50))
        # Continue or exit
        if input("\nEXECUTION DONE. "
                 "Enter to try again, or (q) to quit ") == 'q':
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-load', '--load_path',
                        default=None, required=True,
                        help="Path to the learned model file.")
    parser.add_argument('-r', '--render',
                        default=False,
                        help="Show execution.")
    parser.add_argument('-v', '--verbose',
                        default=0,
                        type=int,
                        help="Define verbose level\n"
                             "0: basic info\n"
                             "1: training - trial info\n"
                             "2: testing - parameter selection info\n"
                             "3: training and testing detailed info\n")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-40s '
                               '%(levelname)-8s %(message)s',
                        handlers=[
                            logging.FileHandler(
                                '{}/test_target.log'.format(args.load_path)),
                            logging.StreamHandler()])
    main_test(
      load_path=args.load_path, render=args.render, verbose=args.verbose)
