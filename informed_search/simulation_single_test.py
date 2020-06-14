
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Load the model from a file and perform a user defined test
"""


import json
import argparse

import utils.experiment_manage as expm
from utils.misc import _TAB


parser = argparse.ArgumentParser()
parser.add_argument('-load', '--load_path',
                    default=None, required=True,
                    help="Path to the learned model file.")



def load_metadata(datapath):
    """Extract class experiment metadata """
    with open(datapath + '/experiment_metadata.json', 'r') as f:
        args_dict = json.load(f)
    return args_dict



def main_test(load_path, display=True):

    load_task_kwargs = load_metadata(load_path)
    experiment = expm.ExperimentManager(load_task_kwargs, 
                                        load_model_path=load_path)

    while True:
        # Input target position
        while True:
            try:
                angle_s, dist_s = input("\nEnter TARGET: "
                                    "angle [deg], distance[m] >> ").split(",")
                test_target = [float(angle_s), float(dist_s)]
                break
            except Exception as i:
                print(i)
                continue
        _, _, test_stats = experiment.evaluate_single_test(test_target, display)

        print("{} TEST TARGET angle: {}; distance: {} {}"
              "\n - Trial outcome:     {}"
              "\n - Model polar error: {:4.2f}"
              "\n - Euclidian error:   {:4.2f}"
              "\n - Polar error norm:  {:4.2f}\n{}".format(
                    '-'*10, angle_s, dist_s, '-'*10, 
                    test_stats['trial_outcome'], 
                    test_stats['model_polar_error'], 
                    test_stats['euclid_error'], 
                    test_stats['polar_error'], '-'*50))

        # Continue
        if input("\nEXECUTION DONE. "
                 "Enter to try again, or (x) to quit ") == 'x':
            break


if __name__ == "__main__":
    args = parser.parse_args()
    main_test(load_path=args.load_path)