
import numpy as np
import matplotlib.pyplot as pl


# This is the true unknown function we are trying to approximate
# f = lambda x: np.sin(0.9*x).flatten()
# f = lambda x: (0.25*(x**2)).flatten()
f = lambda x: (np.sin(np.pi*x)/(np.pi*x)).flatten()

# # Define the kernel
# def kernel(a, b):
#     """ GP squared exponential kernel """
#     kernelParameter = 1
#     sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#     return np.exp(-.5 * (1/kernelParameter) * sqdist)

# # Taken from intro to bayes opt
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 1
    sqdist = (1/kernelParameter) * np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return (1+np.sqrt(5*sqdist)+5*sqdist/3.) * np.exp(-np.sqrt(5*sqdist))


# Create distribution with ONE TRAINING POINT!
N = 1
n = 100         # number of test points.
s = 0.00005    # noise variance.
Xtrain = np.random.uniform(-5, 5, size=(N,1)).reshape(-1,1)
Xtest = np.linspace(-5, 5, n).reshape(-1,1)
Kss = kernel(Xtest, Xtest)

for i in range(10):
    y = f(Xtrain) + s*np.random.randn(N)
    # get posterior MU and SIGMA
    K = kernel(Xtrain, Xtrain)
    L = np.linalg.cholesky(K + s*np.eye(N))
    Ks = kernel(Xtrain, Xtest)
    Lk = np.linalg.solve(L, Ks)
    # get posterior MU and SIGMA
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    var_post = np.sqrt(np.diag(Kss) - np.sum(Lk**2, axis=0))
    # SAMPLE a NEW POINT based on HIGHEST UNCERTAINTY and add it to Xtrain
    # make it uniform from the list
    new_point = Xtest[np.argmax(var_post)].reshape(-1,1)
    Xtrain = np.append(Xtrain, new_point, axis=0)
    print Xtrain.shape, mu.shape, y.shape
    # PLOT stuff
    # pl.figure(1)
    pl.clf()
    pl.plot(Xtrain[:-1], y, 'r+', ms=20)
    pl.plot(Xtest, f(Xtest), 'b-')
    pl.gca().fill_between(Xtest.flat, mu-3*var_post, mu+3*var_post, color="#dddddd")
    pl.plot(Xtest, mu, 'r--', lw=2)
    pl.plot(Xtest, var_post, 'k-', lw=2)
    pl.scatter(new_point, np.max(var_post), c='g')
    # pl.savefig('predictive.png', bbox_inches='tight')
    pl.title('Step:'+str(i))
    pl.axis([-5, 5, -3, 3])
    pl.pause(1)
    # REPEAT 10 times (ERROR MEASURE???)







#########################################################################
### HELPER CODE ###
#########################################################################
# https://www.youtube.com/watch?v=MfHKW5z-OOA&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6&index=9
# np.random.seed(1)
# def kernel(a,b):
#     return sig**2*exp(-(1./2*L**2)*(a-b)**2
# def kernel(a,b, sig=1, Lsq=1):
#     sqdist = np.sum(a**2, 1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a,b.T)
#     return (sig**2)*np.exp(-0.5* (1./Lsq) * sqdist)
# n = 50
# x_t = np.linspace(-0.5,0.5,n).reshape(-1,1)
# K = kernel(x_t, x_t)
# L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
# # f_prior = np.dot(L, np.random.normal(loc=0, size=(n,100)))# + 2*np.sin(x_t)
# f_prior = np.dot(L, np.random.uniform(-1,1,size=(n,100))) + 1# + 2*np.sin(x_t)
# plt.plot(x_t, f_prior)
# plt.show()

#########################################################################

N = 5         # number of training points.
n = 100         # number of test points.
s = 0.00005    # noise variance.

# (TRAIN DATA)
# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + s*np.random.randn(N)

K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))

# (DOMAIN)
# points we're going to make predictions at. 
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# (CONDITION ON NEW POINTS)
# compute the mean at our test points.
Ks = kernel(X, Xtest)
Lk = np.linalg.solve(L, Ks)

mu = np.dot(Lk.T, np.linalg.solve(L, y))
# mu1 = np.dot(Ks.T, np.linalg.solve(L.T, np.linalg.solve(L, y)))

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

v = np.linalg.solve(L, Ks)
# var_post = K_ - np.dot(v.T,v)
var_post = np.sqrt((K_ - np.dot(v.T,v)).diagonal())

num_lines = 10

# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
# pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])
pl.plot(Xtest, var_post, 'k-', lw=2)
# pl.show()

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,num_lines)))
# f_prior = np.dot(L, np.random.uniform(-1,1,size=(n,num_lines)))
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])
# pl.savefig('prior.png', bbox_inches='tight')
# pl.show()

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,num_lines)))
# f_post = mu.reshape(-1,1) + np.dot(L, np.random.uniform(-1,1,size=(n,num_lines)))

pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post, var_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
# pl.savefig('post.png', bbox_inches='tight')
pl.show()



#########################################################################
### RESOURCES ###
# EX: http://sysbio.mrc-bsu.cam.ac.uk/group/images/6/6f/Infpy_gp.pdf    
# https://www.robots.ox.ac.uk/~mebden/reports/GPtutorial.pdf
# http://www.gaussianprocess.org/gpml
#########################################################################