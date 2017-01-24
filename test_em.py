import numpy as np
import math
import matplotlib.pyplot as plt

#### E-M Coin Toss Example as given in the EM tutorial paper by Do and Batzoglou* #### 

def get_binomial_log_likelihood(obs,probs):
    """ Return the (log)likelihood of obs, given the probs"""
    # Binomial Distribution Log PDF
    # ln (pdf)      =             Binomial Coeff            *   product of probabilities
    # ln[f(x|n, p)] =               comb(N,k)               *   num_heads*ln(pH) + (N-num_heads) * ln(1-pH) 

    N = sum(obs);#number of trials  
    k = obs[0] # number of heads
    binomial_coeff = math.factorial(N) / (math.factorial(N-k) * math.factorial(k))
    prod_probs = obs[0]*math.log(probs[0]) + obs[1]*math.log(1-probs[0])
    log_lik = binomial_coeff + prod_probs

    return log_lik

# 1st:  Coin B, {HTTTHHTHTH}, 5H,5T
# 2nd:  Coin A, {HHHHTHHHHH}, 9H,1T
# 3rd:  Coin A, {HTHHHHHTHH}, 8H,2T
# 4th:  Coin B, {HTHTTTHHTT}, 4H,6T
# 5th:  Coin A, {THHHTHHHTH}, 7H,3T
# so, from MLE: pA(heads) = 0.80 and pB(heads)=0.45

# represent the experiments
head_counts = np.array([5,9,8,4,7])
tail_counts = 10-head_counts
experiments = zip(head_counts,tail_counts)

# initialise the pA(heads) and pB(heads)
pA_heads = np.zeros(100); pA_heads[0] = 0.60
pB_heads = np.zeros(100); pB_heads[0] = 0.50

# E-M begins!
delta = 0.001  
j = 0 # iteration counter
improvement = float('inf')
while (improvement>delta):
    expectation_A = np.zeros((len(experiments),2), dtype=float) 
    expectation_B = np.zeros((len(experiments),2), dtype=float)
    for i in range(0,len(experiments)):
        e = experiments[i] # i'th experiment
        ll_A = get_mn_log_likelihood(e,np.array([pA_heads[j],1-pA_heads[j]])) # loglikelihood of e given coin A
        ll_B = get_mn_log_likelihood(e,np.array([pB_heads[j],1-pB_heads[j]])) # loglikelihood of e given coin B

        weightA = math.exp(ll_A) / ( math.exp(ll_A) + math.exp(ll_B) ) # corresponding weight of A proportional to likelihood of A 
        weightB = math.exp(ll_B) / ( math.exp(ll_A) + math.exp(ll_B) ) # corresponding weight of B proportional to likelihood of B                            

        expectation_A[i] = np.dot(weightA, e) 
        expectation_B[i] = np.dot(weightB, e)

    pA_heads[j+1] = sum(expectation_A)[0] / sum(sum(expectation_A)); 
    pB_heads[j+1] = sum(expectation_B)[0] / sum(sum(expectation_B)); 

    improvement = max( abs(np.array([pA_heads[j+1],pB_heads[j+1]]) - np.array([pA_heads[j],pB_heads[j]]) ))
    j = j+1

plt.figure();
plt.plot(range(0,j),pA_heads[0:j], 'r--')
plt.plot(range(0,j),pB_heads[0:j])
plt.show()


# http://dataconomy.com/2015/02/introduction-to-bayes-theorem-with-python/
# http://nbviewer.jupyter.org/github/tfolkman/learningwithdata/blob/master/Bayes_Primer.ipynb
def bern_post(n_params=100, n_sample=100, true_p=.8, prior_p=.5, n_prior=100):
    params = np.linspace(0, 1, n_params)
    sample = np.random.binomial(n=1, p=true_p, size=n_sample)
    likelihood = np.array([np.product(st.bernoulli.pmf(sample, p)) for p in params])
    #likelihood = likelihood / np.sum(likelihood)
    prior_sample = np.random.binomial(n=1, p=prior_p, size=n_prior)
    prior = np.array([np.product(st.bernoulli.pmf(prior_sample, p)) for p in params])
    prior = prior / np.sum(prior)
    posterior = [prior[i] * likelihood[i] for i in range(prior.shape[0])]
    posterior = posterior / np.sum(posterior)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8,8))
    axes[0].plot(params, likelihood)
    axes[0].set_title("Sampling Distribution")
    axes[1].plot(params, prior)
    axes[1].set_title("Prior Distribution")
    axes[2].plot(params, posterior)
    axes[2].set_title("Posterior Distribution")
    sns.despine()
    plt.tight_layout()

return posterior 


#############################
# 1D Bayesian rule example
#############################
x = np.linspace(-10,10,100)
prior = np.array([1./len(x)] * len(x))
mu1 = -3
sig1 = 2
likelihood1 = (1./(sig1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu1)/sig1)**2)
mu2 = 3
sig2 = 2
likelihood2 = (1./(sig2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu2)/sig2)**2)

posterior1 = likelihood1 * prior

posterior2 = likelihood2 * prior

post = posterior1+posterior2
post = post/np.sum(post)
posterior1 = posterior1/np.sum(posterior1)
posterior2 = posterior2/np.sum(posterior2)

plt.plot(x, prior)
# plt.plot(x, likelihood1)
# plt.plot(x, posterior1)
plt.plot(x, likelihood2)
# plt.plot(x, posterior2)
# plt.plot(x, post)
plt.plot(x, posterior1+posterior2)
plt.show()

###### SAMPLING 1D example ######
p = posterior1+posterior2
s=[]
for i in range(0,100):
    u = np.random.uniform(min(p),max(p))
    print u
    m = np.argwhere(p<=u)   # "<="" is because thos eare the good/unexplored parameters
    s.append(np.random.choice(m.reshape(len(m),)))

plt.plot(p)
plt.scatter(s,p[s])
plt.show()



#############################
# 2D Gaussian example
#############################
def generatePDF(x_sample, mu, cov):
    f = (1/np.sqrt(2*np.pi*np.linalg.det(cov)))*np.exp(-0.5*np.dot(np.dot((x_sample - mu), cov), (x_sample - mu).T))
    return f

nsampl = 100   
mu = np.array([-2, -2])
mu2 = np.array([1, 3])
cov = np.eye(2)
# cov = makeCovarianceMatrix(2, 0.2)
x1 = np.linspace(-5,5,num=nsampl)
x2 = np.linspace(-5,5,num=nsampl)
x = np.array([x1,x2])
# f = [generatePDF(c, mu, cov) for c in x.T]
# plt.plot(f)
# plt.show()

mm = np.ones((nsampl,nsampl))
for i, val_x1 in enumerate(x1):
    for j, val_x2 in enumerate(x1):
        mm[i,j]= generatePDF(np.array([val_x1, val_x2]), mu, cov) + generatePDF(np.array([val_x1, val_x2]), mu2, cov)

plt.imshow(mm)
plt.show()

###### SAMPLING 2D example ######
p = mm
s=[]
for i in range(0,100):
    u = np.random.uniform(np.min(mm),np.max(mm))
    m = np.argwhere(p<=u)   # "<="" is because thos eare the good/unexplored parameters
    # print m
    s.append(m[np.random.choice(len(m))])

s=np.array(s)
plt.imshow(p)
plt.scatter(s[:,1],s[:,0])
plt.show()