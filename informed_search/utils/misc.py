
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Helper functions and variables
"""


import numpy as np


# Useful variables

_TAB = ' ' * 75
_EPS = 1e-8   # 0.00005


# Useful functions

def elementwise_sqdist(x, y):
    """Standard squared distance calculation."""
    return np.sqrt(x**2 + y**2)


def scaled_sqdist(m_angle, m_dist, angle_s, dist_s):
    """Squared distance scaled to min-max range."""
    diff_angle = m_angle - angle_s
    diff_angle = (diff_angle - diff_angle.min()) \
        / (diff_angle.max() - diff_angle.min())
    diff_dist = m_dist - dist_s
    diff_dist = (diff_dist - diff_dist.min()) \
        / (diff_dist.max() - diff_dist.min())
    return elementwise_sqdist(diff_angle, diff_dist)
