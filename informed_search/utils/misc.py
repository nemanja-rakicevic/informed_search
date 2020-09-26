
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Helper functions and variables
"""


import numpy as np


_TAB = ' ' * 59
_EPS = 1e-8   # 0.00005


def elementwise_sqdist(x, y):
    return np.sqrt(x**2 + y**2)


def scaled_sqdist(M_angle, M_dist, angle_s, dist_s):
    diff_angle = M_angle - angle_s
    diff_angle = (diff_angle - diff_angle.min()) \
        / (diff_angle.max() - diff_angle.min())
    diff_dist = M_dist - dist_s
    diff_dist = (diff_dist - diff_dist.min()) \
        / (diff_dist.max() - diff_dist.min())
    return elementwise_sqdist(diff_angle, diff_dist)
