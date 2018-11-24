
### KERNEL TESTING
# overview: http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/
# http://pythonhosted.org/GPy/GPy.kern.html
# https://uk.mathworks.com/help/stats/kernel-covariance-function-options.html

# Testing various implementatins of Kernels, taken from:
# GROUP 1) Nando's lectures
# GROUP 2) PyGPML: https://github.com/hadsed/PyGPML/blob/master/kernels.py
# GROUP 3) https://gist.github.com/stober/4964727
# GROUP 4) 



import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as pl



a = np.linspace(0,10,100).reshape(-1,1)
b = np.linspace(0,10,100).reshape(-1,1)

### GROUP 1 ###############################################################
def sqexp(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

def matern(a, b):
    """ GP Matern 5/2 kernel: """
    kernelParameter = 1
    sqdist = (1/kernelParameter) * np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return (1+np.sqrt(5*sqdist)+5*sqdist/3.) * np.exp(-np.sqrt(5*sqdist))
############################################################################

### GROUP 2 ###############################################################
def radial_basis(x=None, z=None, diag=False):
    """
    Radial-basis function (also known as squared-exponential and 
    Gaussian kernels) takes the following form,
    k(t) = \sigma^2_f * exp(-t^2/(2L^2))
    where \sigma and L are the adjustable hyperparameters giving
    hypcov = [ \sigma, L ].
    """
    sf2 = 1
    ell2 = 1
    if diag:
        K = np.zeros((x.shape[0],1))
    else:
        if x is z:
            K = sp.spatial.distance.cdist(x/ell2, x/ell2, 'sqeuclidean')
        else:
            K = sp.spatial.distance.cdist(x/ell2, z/ell2, 'sqeuclidean')
    K = sf2*np.exp(-K/2)
    return K

def rational_quadratic(x=None, z=None, diag=False):
    """
    Rational-quadratic kernel has the following form,
    k(t) = hscale^2 * (1 + t^2/(alpha*lamb^2))^{-alpha}
    where hscale, alpha, and lamb are hyperparameters that give
    hyp = [ hscale, alpha, lamb ].
    """
    hscale = np.exp(1)
    alpha = np.exp(1)
    lamb = np.exp(1)
    if diag:
        K = np.zeros((x.shape[0],1))
    else:
        if x is z:
            K = sp.spatial.distance.cdist(x/lamb**2, x/lamb**2, 'sqeuclidean')
        else:
            K = sp.spatial.distance.cdist(x/lamb**2, z/lamb**2, 'sqeuclidean')
    K = hscale**2 * np.power(1 + K/alpha, -alpha)
    return K

def periodic(x=None, z=None, diag=False):
    """
    The periodic kernel has a form
    k(x,x') = sigma^2 * exp(-2/ell^2 * sin^2(pi*|x-x'|/per))
    where sigma, ell, and per are hyperparameters giving
    hyp = [ sigma, ell, per ].
    """
    sigma = np.exp(1)
    ell = np.exp(1)
    per = np.exp(1)
    if diag:
        K = np.zeros((x.shape[0],1))
    else:
        if x is z:
            K = sp.spatial.distance.cdist(x, x, 'sqeuclidean')
        else:
            K = sp.spatial.distance.cdist(x, z, 'sqeuclidean')
    K = np.sqrt(K)  # need the absolute distance, not squared
    K = sigma**2 * np.exp(-2/ell**2 * np.power(np.sin(np.pi*K/per), 2))
    return K

def spectral_mixture(hypcov, x=None, z=None, diag=False):
    """
    Spectral Mixture kernel takes the following form [1],
    k(t) = \sum^Q_{q=0} w_q \prod^P_{p=0} exp(-2pi^2*v^2_{p,q}*t_p^2)
           * cos(2pi*\mu_{p,q}*t_p)
    It's corresponding hyperparameters are constructed according to
    [ [w_0, w_1, ..., w_q],
      [mu_0, mu_1, ..., mu_q],
      [v_0, v_1, ..., v_q] ]
    and then flattened to give hyp = [ w_0, ..., w_q, mu_0, ..., v_q ].
    So then P is the dimensionality of the data and Q is the number of
    Gaussians in the Gaussian mixture model (roughly speaking, Q is the
    number of peaks we attempt to model).
    [1] Wilson, A. G., & Adams, R. P. (2013). Gaussian process covariance
        kernels for pattern discovery and extrapolation. arXiv preprint 
        arXiv:1302.4245.
    """
    n, D = x.shape
    hypcov = np.array(hypcov).flatten()
    Q = hypcov.size/(1+2*D)
    w = np.exp(hypcov[0:Q])
    m = np.exp(hypcov[Q+np.arange(0,Q*D)]).reshape(D,Q)
    v = np.exp(2*hypcov[Q+Q*D+np.arange(0,Q*D)]).reshape(D,Q)
    d2list = []
    
    if diag:
        d2list = [np.zeros((n,1))]*D
    else:
        if x is z:
            d2list = [np.zeros((n,n))]*D
            for j in np.arange(0,D):
                xslice = np.atleast_2d(x[:,j])
                d2list[j] = sp.spatial.distance.cdist(xslice, xslice, 'sqeuclidean')
        else:
            d2list = [np.zeros((n,z.shape[0]))]*D
            for j in np.arange(0,D):
                xslice = np.atleast_2d(x[:,j])
                zslice = np.atleast_2d(z[:,j])
                d2list[j] = sp.spatial.distance.cdist(xslice, zslice, 'sqeuclidean')

    # Define kernel functions
    k = lambda d2v, dm: np.multiply(np.exp(-2*np.pi**2 * d2v),
                                    np.cos(2*np.pi * dm))
    # Calculate correlation matrix
    K = 0
    # Need the sqrt
    dlist = [ np.sqrt(dim) for dim in d2list ]
    # Now construct the kernel
    for q in range(0,Q):
        C = w[q]**2
        for j,(d,d2) in enumerate(zip(dlist, d2list)):
            C = np.dot(C, k(np.dot(d2, v[j,q]), 
                            np.dot(d, m[j,q])))
        K = K + C
    return K
### #########################################################################

### GROUP 3 ###############################################################
def ecl_kernel(a,b):
    # Kernel from Bishop's Pattern Recognition and Machine Learning pg. 307 Eqn. 6.63.
    thetas = [1.0, 64.0, 0.0, 0.0]
    tmp = np.reshape([sum( (x - y)**2 ) for x in a for y in b], (len(a),len(b)))
    exponential = thetas[0] * np.exp( -0.5 * thetas[1] * tmp )
    linear = thetas[3] * np.dot(x,y.T)
    constant = thetas[2]*np.ones((len(a),len(b)))
    return exponential + constant + linear

def OrnsteinKernel(a,b):
    # Ornstein-Uhlenbeck process kernel.
    theta = 1.0
    return np.reshape([np.exp(-theta * np.sum(abs(x-y))) for x in a for y in b], (len(a),len(b)))
############################################################################   
############################################################################

K1 = sqexp(a,b)
K2 = radial_basis(a,b)
K3 = matern(a,b)
K4 = rational_quadratic(a,b)
K5 = periodic(a,b)
K6 = ecl_kernel(a,b)
K7 = OrnsteinKernel(a,b)

pl.imshow(K7)
pl.show()



def kernel1( a, b):
    """ GP squared exponential kernel """
    sigsq = 1
    siglensq = 0.03
    sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
    return sigsq*np.exp(-.5 *sqdist)

def kernel2( a, b):
    """ GP Matern 5/2 kernel: """
    sigsq = np.exp(1)
    siglensq = 0.03
    sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
    return sigsq * (1 + np.sqrt(5*sqdist) + 5*sqdist/3.) * np.exp(-np.sqrt(5.*sqdist))

def kernel3( a, b):
    """ GP rational quadratic kernel """
    sigsq = 1
    siglensq = 0.03
    alpha = len(a)/2. #np.exp(1)
    sqdist = (1./siglensq) * sp.spatial.distance.cdist(a, b, 'sqeuclidean')
    return sigsq * np.power(1 + 0.5*sqdist/alpha, -alpha)

