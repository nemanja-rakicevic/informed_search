
"""
Author:         Nemanja Rakicevic
Date  :         January 2018
Description:
                Kernel function definitions
"""

import numpy as np

from scipy.spatial.distance import cdist


def se_kernel(a, b, kernel_lenscale, kernel_sigma):
    """ SE squared exponential kernel """
    # kernel_lenscale = 0.01 # 1 0.5 0.3 0.1 
    sqdist = (1. / kernel_lenscale) * cdist(a, b, 'sqeuclidean')
    return kernel_sigma * np.exp(-.5 * sqdist)


def mat_kernel(a, b, kernel_lenscale, kernel_sigma):
    """ MT Matern 5/2 kernel """
    # kernel_lenscale = 0.03 # 1
    sqdist = (1. / kernel_lenscale) * cdist(a, b, 'sqeuclidean')
    return kernel_sigma \
        * (1 + np.sqrt(5 * sqdist) + 5 * sqdist / 3.) \
        * np.exp(-np.sqrt(5. * sqdist))


def rq_kernel(a, b, kernel_lenscale, kernel_sigma):
    """ RQ rational quadratic kernel """
    # kernel_lenscale = 1
    alpha = a.shape[1] / 2.  # a.shape[1]/2. #np.exp(1) #len(a)/2.
    sqdist = (1. / kernel_lenscale) * cdist(a, b, 'sqeuclidean')
    return kernel_sigma * np.power(1 + 0.5 * sqdist / alpha, -alpha)
